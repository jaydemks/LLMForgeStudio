import argparse
import shutil
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GPTConfig:
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float = 0.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


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


def iter_text_files(path: Path):
    if path.is_file():
        yield path
        return
    exts = {".txt", ".md", ".csv", ".jsonl", ".json"}
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def read_dataset_text(dataset_path: Path) -> str:
    chunks = []
    for p in iter_text_files(dataset_path):
        chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(chunks)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out_dir = Path(job["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_state_path = out_dir / "ollama_finetune_resume_state.json"
    runtime_diag_path = out_dir / "runtime_diagnostics_snapshot.json"

    base_model_path = Path(job["base_model_path"])
    if not base_model_path.exists():
        raise SystemExit(f"Base model not found: {base_model_path}")
    tokenizer_path = base_model_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found near base model: {tokenizer_path}")

    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    stoi = tok.get("stoi", {})
    if not stoi:
        raise SystemExit("Tokenizer stoi not found.")

    text = read_dataset_text(Path(job["dataset_path"]))
    ids = [stoi[ch] for ch in text if ch in stoi]
    if len(ids) < 128:
        raise SystemExit("Dataset too small for fine-tuning.")
    data = torch.tensor(ids, dtype=torch.long)

    ckpt = torch.load(base_model_path, map_location="cpu")
    cfgd = ckpt["config"]
    cfg = GPTConfig(
        vocab_size=int(cfgd["vocab_size"]),
        block_size=int(cfgd["block_size"]),
        n_layer=int(cfgd["n_layer"]),
        n_head=int(cfgd["n_head"]),
        n_embd=int(cfgd["n_embd"]),
        dropout=float(cfgd.get("dropout", 0.0)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_diag = {
        "generated_at_utc": utc_now(),
        "python": os.environ.get("PYTHONHOME", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "torch_version": torch.__version__,
        "env": {
            "PATH": os.environ.get("PATH", "")[:2048],
            "LLMFORGE_GGUF_CONVERTER": os.environ.get("LLMFORGE_GGUF_CONVERTER", ""),
        },
    }
    runtime_diag_path.write_text(json.dumps(runtime_diag, indent=2), encoding="utf-8")

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    lr = float(job.get("learning_rate", 2e-4))
    batch_size = int(job.get("batch_size", 2))
    grad_accum = int(job.get("grad_accum", 8))
    epochs = int(job.get("epochs", 3))
    steps_per_epoch = 60
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    global_step = 0
    start_epoch = 1
    if resume_state_path.exists():
        try:
            resume = json.loads(resume_state_path.read_text(encoding="utf-8"))
            start_epoch = max(1, int(resume.get("last_completed_epoch", 0)) + 1)
            global_step = int(resume.get("global_step", 0))
        except Exception:
            start_epoch = 1

    log_path = out_dir / "ollama_finetune_log.jsonl"
    with log_path.open("a", encoding="utf-8") as logf:
        for epoch in range(start_epoch, epochs + 1):
            loss_acc = 0.0
            for _ in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                for _ in range(max(1, grad_accum)):
                    current_batch = batch_size
                    while True:
                        try:
                            x, y = get_batch(data, cfg.block_size, current_batch, device)
                            _, loss = model(x, y)
                            break
                        except RuntimeError as ex:
                            msg = str(ex).lower()
                            if "out of memory" in msg and current_batch > 1:
                                current_batch = max(1, current_batch // 2)
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            raise
                    (loss / max(1, grad_accum)).backward()
                    loss_acc += float(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

            train_loss = loss_acc / max(1, steps_per_epoch * max(1, grad_accum))
            row = {
                "ts_utc": utc_now(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": round(train_loss, 6),
                "message": f"Fine-tuning epoch {epoch}/{epochs}",
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            print(row["message"], flush=True)
            resume_state_path.write_text(json.dumps({
                "updated_at_utc": utc_now(),
                "last_completed_epoch": epoch,
                "global_step": global_step,
            }, indent=2), encoding="utf-8")

    model_name = str(job.get("output_model_name", "finetuned-model")).strip() or "finetuned-model"
    model_out = out_dir / f"{model_name}.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, model_out)

    exports_dir = out_dir / "exports" / "ollama_finetune"
    exports_dir.mkdir(parents=True, exist_ok=True)
    bundle_model_path = exports_dir / f"{model_name}.pt"
    shutil.copy2(model_out, bundle_model_path)
    bundle_tokenizer_path = exports_dir / "tokenizer.json"
    shutil.copy2(tokenizer_path, bundle_tokenizer_path)

    modelfile_path = exports_dir / "Modelfile.template"
    modelfile_path.write_text(
        "\n".join([
            "# Fine-tuned checkpoint bundle (not GGUF yet)",
            f"# Model name: {model_name}",
            "# Convert checkpoint to GGUF before Ollama import.",
            "# Example target Modelfile (after GGUF conversion):",
            "# FROM ./model.gguf",
            "# TEMPLATE \"{{ .Prompt }}\"",
            "# PARAMETER temperature 0.7",
        ]) + "\n",
        encoding="utf-8",
    )
    (exports_dir / "README_OLLAMA_FINETUNE.txt").write_text(
        "Fine-tuning completed on local checkpoint.\n"
        "Bundle contains fine-tuned checkpoint and tokenizer.\n"
        "GGUF conversion is required before final Ollama import.\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "job_path": str(job_path),
        "base_model_path": str(base_model_path),
        "output_model_name": model_name,
        "output_model_path": str(model_out),
        "dataset_path": job["dataset_path"],
        "method": job.get("method"),
        "template": job.get("template"),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "artifacts": {
            "log_path": str(log_path),
            "exports_dir": str(exports_dir),
            "checkpoint_bundle_path": str(bundle_model_path),
            "tokenizer_bundle_path": str(bundle_tokenizer_path),
            "modelfile_template_path": str(modelfile_path),
        },
        "pipeline": {
            "train_stage": "completed",
            "package_stage": "completed",
            "ollama_handoff_stage": "blocked_waiting_for_gguf_conversion",
        },
        "runtime_diagnostics_path": str(runtime_diag_path),
        "resume_state_path": str(resume_state_path),
        "backend": job.get("backend", "ollama-local"),
    }
    (out_dir / "ollama_finetune_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    handoff_status = {
        "status": "blocked",
        "reason": "gguf_conversion_required",
        "message": "Fine-tuned checkpoint is ready, but Ollama handoff requires GGUF conversion.",
        "generated_at_utc": utc_now(),
        "expected_inputs": {
            "checkpoint": str(bundle_model_path),
            "tokenizer": str(bundle_tokenizer_path),
        },
    }
    (exports_dir / "ollama_handoff_status.json").write_text(json.dumps(handoff_status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
