import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_node_id() -> str:
    raw = os.environ.get("LLMFORGE_CLUSTER_NODE_ID", "") or os.environ.get("COMPUTERNAME", "") or os.environ.get("HOSTNAME", "")
    raw = raw.strip().replace(" ", "_")
    return raw if raw else "node-unknown"


def _atomic_claim(src: Path, dst: Path) -> bool:
    try:
        src.replace(dst)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _run_stage(stage_name: str, state_path: Path, out: Path, attempt: int, fn):
    _write_json(state_path, {
        "status": "running",
        "stage": stage_name,
        "attempt": attempt,
        "updated_at_utc": _utc_now(),
    })
    started = time.time()
    fn()
    summary_path = out / f"pipeline_stage_{stage_name}.json"
    _write_json(summary_path, {
        "stage": stage_name,
        "status": "completed",
        "attempt": attempt,
        "duration_seconds": round(max(0.0, time.time() - started), 3),
        "updated_at_utc": _utc_now(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=False)
    args = parser.parse_args()

    role = (os.environ.get("LLMFORGE_CLUSTER_ROLE", "auto") or "auto").strip().lower()
    if args.job:
        job_path = Path(args.job)
        job = json.loads(job_path.read_text(encoding="utf-8"))
    else:
        if role != "worker":
            raise SystemExit("--job is required unless LLMFORGE_CLUSTER_ROLE=worker")
        job_path = Path(os.environ.get("LLMFORGE_CLUSTER_FALLBACK_JOB", "runs/default/job_spec.json"))
        job = {}

    out = Path(job.get("OutputDirectory", os.environ.get("LLMFORGE_CLUSTER_FALLBACK_RUN_DIR", "runs/default")))
    out.mkdir(parents=True, exist_ok=True)

    tcfg = job.get("Training", {})
    max_retries = max(0, int(tcfg.get("ClusterMaxRetries", 0)))
    heartbeat_seconds = max(1, int(tcfg.get("ClusterHeartbeatSeconds", 2)))
    orchestrated = bool(tcfg.get("OrchestratePipelineStages", False))
    run_data = bool(tcfg.get("PipelineRunDataStage", True))
    run_preprocess = bool(tcfg.get("PipelineRunPreprocessStage", True))
    run_train = bool(tcfg.get("PipelineRunTrainStage", True))
    run_eval = bool(tcfg.get("PipelineRunEvalStage", True))

    state_path = out / "cluster_run_state.json"
    heartbeat_path = out / "cluster_heartbeat.json"
    pipeline_state_path = out / "pipeline_stage_state.json"

    train_script = Path(__file__).resolve().parent / "train_stub.py"

    orchestrator = (tcfg.get("ClusterOrchestrator", os.environ.get("LLMFORGE_CLUSTER_ORCHESTRATOR", "local")) or "local").lower()
    if role == "worker" and orchestrator == "local":
        orchestrator = "sharedfs"
    if orchestrator == "sharedfs":
        return _run_sharedfs_cluster(job, job_path, out, max_retries, heartbeat_seconds, orchestrated, run_data, run_preprocess, run_train, run_eval)

    attempt = 0
    while True:
        _write_json(state_path, {
            "status": "running",
            "attempt": attempt,
            "max_retries": max_retries,
            "updated_at_utc": _utc_now(),
            "job_spec": str(job_path),
        })

        def _noop_stage():
            return

        def _run_train_stage():
            proc = subprocess.Popen(
                [sys.executable, str(train_script), "--job", str(job_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            while proc.poll() is None:
                _write_json(heartbeat_path, {
                    "status": "alive",
                    "attempt": attempt,
                    "updated_at_utc": _utc_now(),
                })
                time.sleep(heartbeat_seconds)

            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)

            if proc.returncode != 0:
                raise RuntimeError(f"train stage failed with exit code {proc.returncode}")

        code = 0
        try:
            if orchestrated:
                if run_data:
                    _run_stage("data", pipeline_state_path, out, attempt, _noop_stage)
                if run_preprocess:
                    _run_stage("preprocess", pipeline_state_path, out, attempt, _noop_stage)
                if run_train:
                    _run_stage("train", pipeline_state_path, out, attempt, _run_train_stage)
                if run_eval:
                    _run_stage("eval", pipeline_state_path, out, attempt, _noop_stage)
            else:
                _run_train_stage()
        except Exception:
            code = 1

        if code == 0:
            _write_json(state_path, {
                "status": "completed",
                "attempt": attempt,
                "max_retries": max_retries,
                "exit_code": code,
                "updated_at_utc": _utc_now(),
            })
            _write_json(heartbeat_path, {
                "status": "stopped",
                "attempt": attempt,
                "updated_at_utc": _utc_now(),
            })
            return

        if attempt < max_retries:
            attempt += 1
            _write_json(state_path, {
                "status": "retrying",
                "attempt": attempt,
                "max_retries": max_retries,
                "exit_code": code,
                "updated_at_utc": _utc_now(),
            })
            time.sleep(1.5)
            continue

        _write_json(state_path, {
            "status": "failed",
            "attempt": attempt,
            "max_retries": max_retries,
            "exit_code": code,
            "updated_at_utc": _utc_now(),
        })
        _write_json(heartbeat_path, {
            "status": "stopped",
            "attempt": attempt,
            "updated_at_utc": _utc_now(),
        })
        raise SystemExit(code)


def _execute_local_training(job_path: Path, out: Path, max_retries: int, heartbeat_seconds: int, orchestrated: bool, run_data: bool, run_preprocess: bool, run_train: bool, run_eval: bool):
    state_path = out / "cluster_run_state.json"
    heartbeat_path = out / "cluster_heartbeat.json"
    pipeline_state_path = out / "pipeline_stage_state.json"
    train_script = Path(__file__).resolve().parent / "train_stub.py"

    attempt = 0
    while True:
        _write_json(state_path, {
            "status": "running",
            "attempt": attempt,
            "max_retries": max_retries,
            "updated_at_utc": _utc_now(),
            "job_spec": str(job_path),
        })

        def _noop_stage():
            return

        def _run_train_stage():
            proc = subprocess.Popen(
                [sys.executable, str(train_script), "--job", str(job_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            while proc.poll() is None:
                _write_json(heartbeat_path, {
                    "status": "alive",
                    "attempt": attempt,
                    "updated_at_utc": _utc_now(),
                })
                time.sleep(heartbeat_seconds)

            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)

            if proc.returncode != 0:
                raise RuntimeError(f"train stage failed with exit code {proc.returncode}")

        code = 0
        try:
            if orchestrated:
                if run_data:
                    _run_stage("data", pipeline_state_path, out, attempt, _noop_stage)
                if run_preprocess:
                    _run_stage("preprocess", pipeline_state_path, out, attempt, _noop_stage)
                if run_train:
                    _run_stage("train", pipeline_state_path, out, attempt, _run_train_stage)
                if run_eval:
                    _run_stage("eval", pipeline_state_path, out, attempt, _noop_stage)
            else:
                _run_train_stage()
        except Exception:
            code = 1

        if code == 0:
            _write_json(state_path, {
                "status": "completed",
                "attempt": attempt,
                "max_retries": max_retries,
                "exit_code": code,
                "updated_at_utc": _utc_now(),
            })
            _write_json(heartbeat_path, {
                "status": "stopped",
                "attempt": attempt,
                "updated_at_utc": _utc_now(),
            })
            return 0

        if attempt < max_retries:
            attempt += 1
            _write_json(state_path, {
                "status": "retrying",
                "attempt": attempt,
                "max_retries": max_retries,
                "exit_code": code,
                "updated_at_utc": _utc_now(),
            })
            time.sleep(1.5)
            continue

        _write_json(state_path, {
            "status": "failed",
            "attempt": attempt,
            "max_retries": max_retries,
            "exit_code": code,
            "updated_at_utc": _utc_now(),
        })
        _write_json(heartbeat_path, {
            "status": "stopped",
            "attempt": attempt,
            "updated_at_utc": _utc_now(),
        })
        return code


def _run_sharedfs_cluster(job: dict, job_path: Path, out: Path, max_retries: int, heartbeat_seconds: int, orchestrated: bool, run_data: bool, run_preprocess: bool, run_train: bool, run_eval: bool):
    shared_root = Path(os.environ.get("LLMFORGE_CLUSTER_SHARED_ROOT", str(out / "_cluster_shared")))
    queue_pending = shared_root / "queue" / "pending"
    queue_claimed = shared_root / "queue" / "claimed"
    queue_result = shared_root / "queue" / "result"
    queue_heartbeats = shared_root / "queue" / "heartbeats"
    for p in [queue_pending, queue_claimed, queue_result, queue_heartbeats]:
        p.mkdir(parents=True, exist_ok=True)

    role = (os.environ.get("LLMFORGE_CLUSTER_ROLE", "auto") or "auto").strip().lower()
    node_id = _safe_node_id()
    ticket = f"{out.name}-{int(time.time())}"
    ticket_payload = {
        "ticket": ticket,
        "created_at_utc": _utc_now(),
        "job_spec_path": str(job_path),
        "output_directory": str(out),
        "max_retries": max_retries,
        "heartbeat_seconds": heartbeat_seconds,
        "orchestrated": orchestrated,
        "run_data": run_data,
        "run_preprocess": run_preprocess,
        "run_train": run_train,
        "run_eval": run_eval,
    }

    if role in {"coordinator", "auto"}:
        pending_path = queue_pending / f"{ticket}.json"
        _write_json(pending_path, ticket_payload)
        _write_json(out / "cluster_queue_ticket.json", {
            "ticket": ticket,
            "role": role,
            "shared_root": str(shared_root),
            "pending_path": str(pending_path),
            "node_id": node_id,
            "updated_at_utc": _utc_now(),
        })

    if role == "coordinator":
        return _wait_for_ticket_result(ticket, out, queue_result, queue_heartbeats, heartbeat_seconds)

    if role in {"worker", "auto"}:
        # Worker mode: claim one pending ticket and execute.
        claimed_ticket_path = _claim_pending_ticket(queue_pending, queue_claimed, node_id)
        if claimed_ticket_path is not None:
            payload = json.loads(claimed_ticket_path.read_text(encoding="utf-8"))
            target_job_spec = Path(payload.get("job_spec_path", str(job_path)))
            target_out = Path(payload.get("output_directory", str(out)))
            max_retries = int(payload.get("max_retries", max_retries))
            hb_secs = int(payload.get("heartbeat_seconds", heartbeat_seconds))
            result_code = _execute_local_training(
                target_job_spec,
                target_out,
                max_retries=max_retries,
                heartbeat_seconds=max(1, hb_secs),
                orchestrated=bool(payload.get("orchestrated", orchestrated)),
                run_data=bool(payload.get("run_data", run_data)),
                run_preprocess=bool(payload.get("run_preprocess", run_preprocess)),
                run_train=bool(payload.get("run_train", run_train)),
                run_eval=bool(payload.get("run_eval", run_eval)),
            )
            _write_json(queue_result / f"{payload.get('ticket', ticket)}.json", {
                "ticket": payload.get("ticket", ticket),
                "status": "completed" if result_code == 0 else "failed",
                "exit_code": result_code,
                "node_id": node_id,
                "claimed_path": str(claimed_ticket_path),
                "updated_at_utc": _utc_now(),
            })
            return result_code

    # Auto fallback: if no worker claimed ticket quickly, run locally and publish result.
    if role == "auto":
        deadline = time.time() + max(5, heartbeat_seconds * 3)
        result_path = queue_result / f"{ticket}.json"
        while time.time() < deadline:
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                return int(result.get("exit_code", 1))
            _write_json(queue_heartbeats / f"{ticket}.json", {
                "ticket": ticket,
                "role": "auto-coordinator",
                "node_id": node_id,
                "updated_at_utc": _utc_now(),
            })
            time.sleep(max(1, heartbeat_seconds))

        result_code = _execute_local_training(job_path, out, max_retries, heartbeat_seconds, orchestrated, run_data, run_preprocess, run_train, run_eval)
        _write_json(queue_result / f"{ticket}.json", {
            "ticket": ticket,
            "status": "completed" if result_code == 0 else "failed",
            "exit_code": result_code,
            "node_id": node_id,
            "mode": "auto-fallback-local",
            "updated_at_utc": _utc_now(),
        })
        return result_code

    return 1


def _claim_pending_ticket(queue_pending: Path, queue_claimed: Path, node_id: str):
    pendings = sorted(queue_pending.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in pendings:
        claimed = queue_claimed / f"{p.stem}__{node_id}.json"
        if _atomic_claim(p, claimed):
            return claimed
    return None


def _wait_for_ticket_result(ticket: str, out: Path, queue_result: Path, queue_heartbeats: Path, heartbeat_seconds: int):
    result_path = queue_result / f"{ticket}.json"
    while True:
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            _write_json(out / "cluster_run_state.json", {
                "status": result.get("status", "unknown"),
                "ticket": ticket,
                "exit_code": int(result.get("exit_code", 1)),
                "worker_node_id": result.get("node_id", ""),
                "updated_at_utc": _utc_now(),
            })
            _write_json(out / "cluster_heartbeat.json", {
                "status": "stopped",
                "ticket": ticket,
                "updated_at_utc": _utc_now(),
            })
            return int(result.get("exit_code", 1))

        _write_json(queue_heartbeats / f"{ticket}.json", {
            "ticket": ticket,
            "role": "coordinator",
            "updated_at_utc": _utc_now(),
        })
        _write_json(out / "cluster_heartbeat.json", {
            "status": "waiting-worker",
            "ticket": ticket,
            "updated_at_utc": _utc_now(),
        })
        time.sleep(max(1, heartbeat_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
