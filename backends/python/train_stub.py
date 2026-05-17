import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_pipeline import clean_text, maybe_curriculum_view
from dataset_streaming import load_dataset_text
from training_runtime import (
    build_amp,
    build_optimizer,
    build_scheduler,
    eval_metrics,
    maybe_run_qat_foundation,
    maybe_quantize_dynamic,
)
from eval_suite import run_eval_suite

ERROR_DATASET_EMPTY = "DATASET_EMPTY"
ERROR_TRAIN_WINDOW_INVALID = "TRAIN_WINDOW_INVALID"
ERROR_TOKENIZER_STATE_REQUIRED = "TOKENIZER_STATE_REQUIRED"
ERROR_RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"


def write_quantization_report(out_dir: Path, quant_meta: dict, train_loss: float, val_loss: float):
    if not quant_meta:
        return None, None

    report_json = out_dir / "quantization_report.json"
    report_md = out_dir / "quantization_report.md"
    payload = {
        "profile": quant_meta.get("profile", "unknown"),
        "quant_mode": quant_meta.get("quant_mode", "unknown"),
        "dtype": quant_meta.get("dtype", "qint8"),
        "calibration_samples": quant_meta.get("calibration_samples", 0),
        "latency_ms_fp32": quant_meta.get("latency_ms_fp32", 0.0),
        "latency_ms_quantized": quant_meta.get("latency_ms_quantized", 0.0),
        "latency_gain_percent": quant_meta.get("latency_gain_percent", 0.0),
        "estimated_quality_delta_percent": quant_meta.get("estimated_quality_delta_percent", 0.0),
        "reference_train_loss": train_loss,
        "reference_val_loss": val_loss,
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(
        "\n".join([
            "# Quantization Report",
            "",
            f"- Profile: `{payload['profile']}`",
            f"- Mode: `{payload['quant_mode']}`",
            f"- Dtype: `{payload['dtype']}`",
            f"- Calibration samples: `{payload['calibration_samples']}`",
            f"- Latency FP32 (ms): `{payload['latency_ms_fp32']}`",
            f"- Latency Quantized (ms): `{payload['latency_ms_quantized']}`",
            f"- Latency gain (%): `{payload['latency_gain_percent']}`",
            f"- Estimated quality delta (%): `{payload['estimated_quality_delta_percent']}`",
            f"- Reference train/val loss: `{payload['reference_train_loss']}` / `{payload['reference_val_loss']}`",
        ]),
        encoding="utf-8",
    )
    return str(report_json), str(report_md)


def write_export_artifacts(out_dir: Path, model, cfg, tcfg: dict):
    export_onnx = bool(tcfg.get("ExportOnnx", False))
    export_gguf = bool(tcfg.get("ExportGguf", False))
    result = {
        "onnx_path": None,
        "onnx_status": "disabled",
        "gguf_path": None,
        "gguf_status": "disabled",
    }

    if export_onnx:
        onnx_path = out_dir / "model.onnx"
        try:
            dummy = torch.randint(0, max(2, cfg.vocab_size), (1, min(32, cfg.block_size)), dtype=torch.long)
            model_cpu = model.cpu().eval()
            class _OnnxWrapper(nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, input_ids):
                    logits, _ = self.inner(input_ids, None)
                    return logits

            wrapper = _OnnxWrapper(model_cpu)
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(onnx_path),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
                opset_version=17,
            )
            result["onnx_path"] = str(onnx_path)
            result["onnx_status"] = "exported"
        except Exception as ex:
            msg = str(ex)
            lowered = msg.lower()
            if "module onnx is not installed" in lowered or "no module named 'onnx'" in lowered:
                note_path = out_dir / "model.onnx.skipped.txt"
                note_path.write_text(
                    "ONNX export skipped: python module 'onnx' is not installed in backend environment.\n"
                    "Install optional dependency and retry if you need ONNX artifacts.",
                    encoding="utf-8",
                )
                result["onnx_status"] = "skipped_missing_dependency"
                result["onnx_path"] = str(note_path)
            else:
                failure_path = out_dir / "model.onnx.export_error.txt"
                failure_path.write_text(msg, encoding="utf-8")
                result["onnx_status"] = "failed"
                result["onnx_path"] = str(failure_path)

    if export_gguf:
        gguf_model_path = out_dir / "model.gguf"
        gguf_status_path = out_dir / "gguf_conversion_status.json"
        runtime_script = Path(__file__).resolve().parent / "gguf_converter_runtime.py"
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(runtime_script),
                    "--input-dir",
                    str(out_dir),
                    "--output",
                    str(gguf_model_path),
                    "--status",
                    str(gguf_status_path),
                    "--attempts",
                    "2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            status_payload = {}
            if gguf_status_path.exists():
                status_payload = json.loads(gguf_status_path.read_text(encoding="utf-8"))
            status = str(status_payload.get("status", "failed")).lower()
            if status == "completed" and gguf_model_path.exists():
                result["gguf_path"] = str(gguf_model_path)
                result["gguf_status"] = "exported"
            elif status in {"blocked", "failed"}:
                result["gguf_path"] = str(gguf_status_path)
                result["gguf_status"] = status
            else:
                result["gguf_path"] = str(gguf_status_path)
                result["gguf_status"] = "failed"
            if proc.returncode not in (0, 2, 3, 4, 5, 6, 7):
                result["gguf_status"] = "failed"
        except Exception as ex:
            failure_path = out_dir / "gguf_conversion_runtime_error.txt"
            failure_path.write_text(str(ex), encoding="utf-8")
            result["gguf_status"] = "failed"
            result["gguf_path"] = str(failure_path)

    return result


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        pos = torch.arange(0, t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def build_char_tokenizer(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi):
    return [stoi[ch] for ch in text if ch in stoi]


def try_load_ui_tokenization_state(out_dir: Path):
    """
    If UI tokenization state exists, prefer it over char-level fallback so backend
    training really follows the selected tokenizer pipeline (BPE/WordPiece/etc.).
    """
    state_path = out_dir / "tokenization_state.json"
    if not state_path.exists():
        return None

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    token_ids = payload.get("TokenIds") or payload.get("tokenIds") or []
    vocab = payload.get("Vocabulary") or payload.get("vocabulary") or []
    if not isinstance(token_ids, list) or not isinstance(vocab, list):
        return None
    if len(token_ids) == 0 or len(vocab) < 2:
        return None

    stoi = {}
    itos = {}
    for item in vocab:
        if not isinstance(item, dict):
            continue
        try:
            tid = int(item.get("Id"))
            tok = str(item.get("Token", ""))
        except Exception:
            continue
        if not tok:
            continue
        stoi[tok] = tid
        itos[tid] = tok

    if len(stoi) < 2:
        return None

    max_id = max(stoi.values())
    safe_ids = []
    for x in token_ids:
        try:
            v = int(x)
        except Exception:
            continue
        if 0 <= v <= max_id:
            safe_ids.append(v)

    if len(safe_ids) < 16:
        return None

    return {
        "token_ids": safe_ids,
        "stoi": stoi,
        "itos": itos,
    }


def get_batch(data, block_size, batch_size, device):
    n = int(len(data))
    if n < 3:
        raise RuntimeError(f"{ERROR_TRAIN_WINDOW_INVALID}: token stream too short for batching (len={n})")

    effective_block = max(1, min(int(block_size), n - 2))
    upper = n - effective_block - 1
    if upper <= 0:
        raise RuntimeError(
            f"{ERROR_TRAIN_WINDOW_INVALID}: invalid batch window (len={n}, block={block_size}, effective_block={effective_block})"
        )

    ix = torch.randint(upper, (batch_size,))
    x = torch.stack([data[i : i + effective_block] for i in ix])
    y = torch.stack([data[i + 1 : i + effective_block + 1] for i in ix])
    return x.to(device), y.to(device), effective_block


def resolve_device(force_cpu: bool):
    if force_cpu:
        return "cpu", "CPU (forced by user)"
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(torch.cuda.current_device())
    try:
        import torch_directml

        return torch_directml.device(), "DirectML"
    except Exception:
        return "cpu", "CPU fallback"


def maybe_init_distributed(device, tcfg):
    if not bool(tcfg.get("DistributedTraining", False)):
        return None
    strategy = str(tcfg.get("MultiGpuStrategy", "none")).lower()
    if strategy == "none":
        return None
    if device != "cuda":
        return None

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": world_size,
        "strategy": strategy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out = Path(job.get("OutputDirectory", "runs/default"))
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    mcfg = job.get("Model", {})
    tcfg = job.get("Training", {})
    dataset_path = Path(job.get("DatasetPath", ""))

    resume_enabled = bool(tcfg.get("ResumeDatasetFromState", True))
    shuffle_enabled = bool(tcfg.get("DeterministicShardShuffle", True))
    shuffle_seed = int(tcfg.get("DataShuffleSeed", 42))

    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    world_size = env_world if bool(tcfg.get("DistributedTraining", False)) else 1
    rank = env_rank if world_size > 1 else 0

    text = load_dataset_text(
        dataset_path,
        out,
        resume_enabled=resume_enabled,
        shuffle_enabled=shuffle_enabled,
        shuffle_seed=shuffle_seed,
        rank=rank,
        world_size=world_size,
    )

    text = clean_text(text, tcfg)
    if not text or not text.strip():
        raise RuntimeError(f"{ERROR_DATASET_EMPTY}: dataset text empty after loading/cleaning")

    ui_tok = try_load_ui_tokenization_state(out)
    if ui_tok is None:
        raise RuntimeError(
            f"{ERROR_TOKENIZER_STATE_REQUIRED}: tokenization_state.json is missing/invalid. "
            "Run Tokenization first and ensure state is saved before training."
        )

    stoi = ui_tok["stoi"]
    itos = ui_tok["itos"]
    ids = ui_tok["token_ids"]
    print(json.dumps({
        "event": "using_ui_tokenization_state",
        "token_count": len(ids),
        "vocab_size": len(stoi),
        "path": str(out / "tokenization_state.json"),
    }), flush=True)
    if len(ids) < 2048:
        ids = ids * (2048 // max(1, len(ids)) + 1)

    split = int(len(ids) * float(tcfg.get("TrainSplit", 0.9)))
    split = max(2, min(split, len(ids) - 2))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)

    requested_block_size = int(mcfg.get("BlockSize", 128))
    max_train_block = max(8, len(train_ids) - 2)
    safe_block_size = max(8, min(requested_block_size, max_train_block))
    if safe_block_size != requested_block_size:
        print(
            json.dumps(
                {
                    "event": "block_size_auto_clamped",
                    "requested_block_size": requested_block_size,
                    "effective_block_size": safe_block_size,
                    "train_token_count": int(len(train_ids)),
                    "message": "Block size reduced automatically because dataset token count is too small for requested context window.",
                }
            ),
            flush=True,
        )

    cfg = GPTConfig(
        vocab_size=max(2, len(stoi)),
        block_size=safe_block_size,
        n_layer=int(mcfg.get("Layers", 4)),
        n_head=int(mcfg.get("Heads", 4)),
        n_embd=int(mcfg.get("EmbeddingSize", 128)),
        dropout=float(mcfg.get("Dropout", 0.0)),
    )

    force_cpu = bool(tcfg.get("ForceCpu", False))
    device, device_label = resolve_device(force_cpu)
    dist = maybe_init_distributed(device if isinstance(device, str) else "directml", tcfg)

    model = GPT(cfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    max_steps = int(tcfg.get("MaxSteps", 300))
    scheduler = build_scheduler(optimizer, tcfg, max_steps)
    use_amp, scaler, amp_dtype = build_amp(device if isinstance(device, str) else "cpu", tcfg)

    eval_every = max(1, int(tcfg.get("EvalEvery", 20)))
    batch_size = int(tcfg.get("BatchSize", 16))
    grad_accum_steps = max(1, int(tcfg.get("GradientAccumulationSteps", 1)))
    checkpoint_every = max(0, int(tcfg.get("CheckpointEvery", 0)))

    last_loss = 0.0
    last_vloss = 0.0

    with log_path.open("w", encoding="utf-8") as f:
        start = time.time()
        for step in range(max_steps + 1):
            active_train, curriculum_frac = maybe_curriculum_view(train_ids, step, max_steps, tcfg)
            x, y, eff_block = get_batch(active_train, cfg.block_size, batch_size, device)

            optimizer.zero_grad(set_to_none=True)
            loss_acc = 0.0
            loss = None
            for _ in range(grad_accum_steps):
                x, y, eff_block = get_batch(active_train, cfg.block_size, batch_size, device)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        _, loss = model(x, y)
                        loss = loss / grad_accum_steps
                    scaler.scale(loss).backward()
                else:
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps
                    loss.backward()
                loss_acc += float(loss.item())

            if use_amp:
                if bool(tcfg.get("EnableGradientClipping", True)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("GradientClipNorm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                if bool(tcfg.get("EnableGradientClipping", True)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("GradientClipNorm", 1.0)))
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy, _ = get_batch(val_ids if len(val_ids) > cfg.block_size + 2 else train_ids, cfg.block_size, batch_size, device)
                    _, vloss = model(vx, vy)
                model.train()

                elapsed = max(1e-6, time.time() - start)
                tokens_done = (step + 1) * batch_size * max(1, int(eff_block))
                last_loss = float(loss_acc)
                last_vloss = float(vloss.item())
                metrics = eval_metrics(last_loss, last_vloss)

                row = {
                    "step": step,
                    "train_loss": round(last_loss, 6),
                    "val_loss": round(last_vloss, 6),
                    "tokens_per_second": round(tokens_done / elapsed, 2),
                    "train_perplexity": round(metrics["train_perplexity"], 4),
                    "val_perplexity": round(metrics["val_perplexity"], 4),
                    "generalization_gap": round(metrics["generalization_gap"], 6),
                    "curriculum_fraction": round(curriculum_frac, 3),
                    "gradient_accumulation_steps": grad_accum_steps,
                    "message": f"training on {device_label}",
                }
                if dist is not None:
                    row["distributed"] = dist

                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)

                if checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                    ckpt_path = out / f"checkpoint_step_{step}.pt"
                    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "step": step}, ckpt_path)

    model_path = out / "model.pt"
    tokenizer_path = out / "tokenizer.json"
    manifest_path = out / "checkpoint_manifest.json"

    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, model_path)
    tokenizer_type = "ui-tokenizer-state" if ui_tok is not None else "char"
    tokenizer_path.write_text(json.dumps({"type": tokenizer_type, "stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, indent=2), encoding="utf-8")

    quantized_path, quant_meta = maybe_quantize_dynamic(model, out / "model_int8.pt", tcfg, device if isinstance(device, str) else "cpu")
    qat_meta = maybe_run_qat_foundation(out, tcfg)
    metrics = eval_metrics(last_loss, last_vloss)
    eval_suite = str(tcfg.get("EvalSuite", "basic"))
    eval_json, eval_md, eval_trend, eval_regression, eval_release_md, eval_avg, eval_band, eval_passed, eval_threshold = run_eval_suite(
        out,
        eval_suite,
        last_loss,
        last_vloss,
        metrics["generalization_gap"],
    )

    manifest = {
        "formatVersion": "0.2",
        "modelWeightsPath": str(model_path),
        "tokenizerPath": str(tokenizer_path),
        "step": max_steps,
        "trainLoss": last_loss,
        "valLoss": last_vloss,
        "optimizer": str(tcfg.get("Optimizer", "adamw")),
        "scheduler": str(tcfg.get("Scheduler", "none")),
        "alignmentMode": str(tcfg.get("AlignmentMode", "none")),
        "fineTuningOrchestration": bool(tcfg.get("FineTuningOrchestration", False)),
        "rewardModelingEnabled": bool(tcfg.get("RewardModelingEnabled", False)),
        "safetyPolicyMode": str(tcfg.get("SafetyPolicyMode", "standard")),
        "rlhfFeedbackSource": str(tcfg.get("RlhfFeedbackSource", "inline")),
        "evalSuite": eval_suite,
        "evalSummaryPath": eval_json,
        "evalScorecardPath": eval_md,
        "evalTrendPath": eval_trend,
        "evalRegressionPath": eval_regression,
        "evalReleaseScorecardPath": eval_release_md,
        "evalAverageScore": eval_avg,
        "evalBand": eval_band,
        "evalReleaseGatePassed": eval_passed,
        "evalReleaseGateThreshold": eval_threshold,
        "distributedTraining": bool(tcfg.get("DistributedTraining", False)),
        "multiGpuStrategy": str(tcfg.get("MultiGpuStrategy", "none")),
        "gradientAccumulationSteps": grad_accum_steps,
        "autoDeviceMap": bool(tcfg.get("AutoDeviceMap", True)),
        "qatPathEnabled": bool(tcfg.get("EnableQatPath", False)),
    }
    if quantized_path:
        manifest["quantizedModelPath"] = quantized_path
    if quant_meta:
        manifest["quantization"] = quant_meta
        q_json, q_md = write_quantization_report(out, quant_meta, last_loss, last_vloss)
        manifest["quantizationReportPath"] = q_json
        manifest["quantizationReportMarkdownPath"] = q_md
    if qat_meta:
        manifest["qatReportPath"] = qat_meta.get("path")
        manifest["qatFineTuneSteps"] = qat_meta.get("steps")
        manifest["qatStatus"] = qat_meta.get("status")

    export_meta = write_export_artifacts(out, model, cfg, tcfg)
    manifest["exportTargets"] = {
        "onnx": {"status": export_meta["onnx_status"], "path": export_meta["onnx_path"]},
        "gguf": {"status": export_meta["gguf_status"], "path": export_meta["gguf_path"]},
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        msg = str(ex).strip() or repr(ex)
        code = ERROR_RUNTIME_EXCEPTION
        if msg.startswith(f"{ERROR_DATASET_EMPTY}:"):
            code = ERROR_DATASET_EMPTY
        elif msg.startswith(f"{ERROR_TRAIN_WINDOW_INVALID}:"):
            code = ERROR_TRAIN_WINDOW_INVALID
        payload = {"error_code": code, "message": msg}
        print("LLMFORGE_ERROR: " + json.dumps(payload, ensure_ascii=False), flush=True)
        sys.exit(1)
