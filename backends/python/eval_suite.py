import json
from pathlib import Path


def _load_pack(pack_file: Path, eval_suite: str):
    if pack_file.exists():
        payload = json.loads(pack_file.read_text(encoding="utf-8"))
        if eval_suite in payload and isinstance(payload[eval_suite], list):
            return payload[eval_suite]

    fallback = 20 if eval_suite == "full-20" else 10 if eval_suite == "standard-10" else 5 if eval_suite == "quick-5" else 3
    return [{"id": f"task_{i+1:02d}", "category": "generic", "weight": 1.0} for i in range(fallback)]


def _category_bias(category: str):
    return {
        "reasoning": 0.0,
        "factuality": -1.5,
        "coding": -2.0,
        "safety": -1.0,
        "robustness": -2.5,
    }.get((category or "").lower(), -1.0)


def _band(score: float):
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "fair"
    return "poor"


def _pass_fail(score: float, eval_suite: str):
    threshold = {"quick-5": 60.0, "standard-10": 65.0, "full-20": 70.0}.get(eval_suite, 60.0)
    return score >= threshold, threshold


def _trend_from_logs(out_dir: Path):
    log_path = out_dir / "train_log.jsonl"
    if not log_path.exists():
        return []

    rows = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except Exception:
            continue

        step = int(row.get("step", 0))
        train_loss = float(row.get("train_loss", 10.0))
        val_loss = float(row.get("val_loss", 10.0))
        gap = max(0.0, val_loss - train_loss)
        score = max(0.0, min(100.0, 100.0 - val_loss * 18.0 - gap * 25.0))
        rows.append({"step": step, "estimated_eval_score": round(score, 2)})

    return rows


def _load_previous_eval(run_dir: Path):
    reg = run_dir / "eval_regression.json"
    if not reg.exists():
        return []
    try:
        payload = json.loads(reg.read_text(encoding="utf-8"))
        points = payload.get("history", [])
        if isinstance(points, list):
            return points
    except Exception:
        pass
    return []


def _build_release_candidate_scorecard(out_dir: Path, summary: dict):
    gate = bool(summary.get("release_gate_passed", False))
    avg = float(summary.get("average_score", 0.0))
    band = str(summary.get("band", "unknown"))
    suite = str(summary.get("eval_suite", "basic"))
    threshold = float(summary.get("release_gate_threshold", 0.0))
    tasks = summary.get("tasks", [])
    weak = sorted(tasks, key=lambda x: float(x.get("score", 0.0)))[:3]

    lines = [
        "# Release Candidate Scorecard",
        "",
        f"- Suite: `{suite}`",
        f"- Average: `{avg}`",
        f"- Band: `{band}`",
        f"- Gate threshold: `{threshold}`",
        f"- Candidate verdict: `{'PASS' if gate else 'FAIL'}`",
        "",
        "## Weakest Tasks",
    ]
    lines.extend([f"- {t.get('id','?')} ({t.get('category','generic')}): {t.get('score',0.0)}" for t in weak])
    lines.append("")
    lines.append("## Recommendation")
    if gate:
        lines.append("- Candidate can proceed to manual runtime validation.")
    else:
        lines.append("- Candidate blocked: improve weak tasks and rerun eval.")

    path = out_dir / "release_candidate_scorecard.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run_eval_suite(out_dir: Path, eval_suite: str, train_loss: float, val_loss: float, gap: float):
    pack_file = out_dir / "eval_benchmarks.json"
    if not pack_file.exists():
        local_pack_file = Path(__file__).resolve().parent / "eval_benchmarks.json"
        pack_file = local_pack_file

    tasks_spec = _load_pack(pack_file, eval_suite)

    base = max(0.0, min(100.0, 100.0 - val_loss * 18.0 - gap * 25.0))
    tasks = []

    for i, task in enumerate(tasks_spec):
        tid = task.get("id", f"task_{i+1:02d}")
        cat = task.get("category", "generic")
        weight = float(task.get("weight", 1.0))
        score = max(0.0, min(100.0, base + _category_bias(cat) - (i * 0.25)))
        tasks.append({
            "id": tid,
            "category": cat,
            "weight": weight,
            "score": round(score, 2),
        })

    total_w = sum(max(0.0, t["weight"]) for t in tasks) or 1.0
    weighted_sum = sum(t["score"] * max(0.0, t["weight"]) for t in tasks)
    avg = round(weighted_sum / total_w, 2)
    band = _band(avg)
    passed, threshold = _pass_fail(avg, eval_suite)

    summary = {
        "eval_suite": eval_suite,
        "num_benchmarks": len(tasks),
        "average_score": avg,
        "band": band,
        "release_gate_passed": passed,
        "release_gate_threshold": threshold,
        "tasks": tasks,
    }

    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trend = _trend_from_logs(out_dir)
    history = _load_previous_eval(out_dir)
    history.append({
        "suite": eval_suite,
        "average_score": avg,
        "band": band,
        "release_gate_passed": passed,
        "release_gate_threshold": threshold,
    })
    regression = {
        "history": history[-50:],
        "last_delta_vs_previous": round(avg - history[-2]["average_score"], 3) if len(history) > 1 else 0.0,
    }
    trend_path = out_dir / "eval_trend.json"
    trend_path.write_text(json.dumps({"points": trend}, indent=2), encoding="utf-8")
    regression_path = out_dir / "eval_regression.json"
    regression_path.write_text(json.dumps(regression, indent=2), encoding="utf-8")

    md_lines = [
        "# Eval Scorecard",
        "",
        f"- Suite: `{eval_suite}`",
        f"- Benchmarks: `{len(tasks)}`",
        f"- Weighted average score: `{avg}`",
        f"- Band: `{band}`",
        f"- Release gate threshold: `{threshold}`",
        f"- Release gate passed: `{passed}`",
        "",
        "## Task Scores",
    ]
    md_lines.extend([f"- {t['id']} ({t['category']}, w={t['weight']}): {t['score']}" for t in tasks])

    md_path = out_dir / "eval_scorecard.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    rc_path = _build_release_candidate_scorecard(out_dir, summary)

    return str(summary_path), str(md_path), str(trend_path), str(regression_path), str(rc_path), avg, band, passed, threshold
