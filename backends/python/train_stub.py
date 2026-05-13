import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        b, t = idx.shape
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
    out = Path(job.get("OutputDirectory", "runs/default"))
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    dataset_path = Path(job.get("DatasetPath", ""))
    if dataset_path.exists():
        text = dataset_path.read_text(encoding="utf-8")
    else:
        text = "hello world\n" * 1000

    mcfg = job.get("Model", {})
    tcfg = job.get("Training", {})

    stoi, itos = build_char_tokenizer(text)
    ids = encode(text, stoi)
    if len(ids) < 2048:
        ids = ids * (2048 // max(1, len(ids)) + 1)

    split = int(len(ids) * float(tcfg.get("TrainSplit", 0.9)))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)

    cfg = GPTConfig(
        vocab_size=max(2, len(stoi)),
        block_size=int(mcfg.get("BlockSize", 128)),
        n_layer=int(mcfg.get("Layers", 4)),
        n_head=int(mcfg.get("Heads", 4)),
        n_embd=int(mcfg.get("EmbeddingSize", 128)),
        dropout=float(mcfg.get("Dropout", 0.0)),
    )

    force_cpu = bool(tcfg.get("ForceCpu", False))
    if force_cpu:
        device = "cpu"
        device_label = "CPU (forced by user)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_label = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        try:
            import torch_directml

            device = torch_directml.device()
            device_label = "DirectML"
        except Exception:
            device = "cpu"
            device_label = "CPU fallback"

    model = GPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("LearningRate", 3e-4)))

    max_steps = int(tcfg.get("MaxSteps", 300))
    eval_every = max(1, int(tcfg.get("EvalEvery", 20)))
    batch_size = int(tcfg.get("BatchSize", 16))

    with log_path.open("w", encoding="utf-8") as f:
        start = time.time()
        for step in range(max_steps + 1):
            x, y = get_batch(train_ids, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch(val_ids if len(val_ids) > cfg.block_size + 2 else train_ids, cfg.block_size, batch_size, device)
                    _, vloss = model(vx, vy)
                model.train()

                elapsed = max(1e-6, time.time() - start)
                tokens_done = (step + 1) * batch_size * cfg.block_size
                row = {
                    "step": step,
                    "train_loss": round(float(loss.item()), 6),
                    "val_loss": round(float(vloss.item()), 6),
                    "tokens_per_second": round(tokens_done / elapsed, 2),
                    "message": f"training on {device_label}"
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)

    model_path = out / "model.pt"
    tokenizer_path = out / "tokenizer.json"
    manifest_path = out / "checkpoint_manifest.json"

    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, model_path)
    tokenizer_path.write_text(json.dumps({"type": "char", "stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, indent=2), encoding="utf-8")

    manifest = {
        "formatVersion": "0.1",
        "modelWeightsPath": str(model_path),
        "tokenizerPath": str(tokenizer_path),
        "step": max_steps,
        "trainLoss": float(loss.item()),
        "valLoss": float(vloss.item()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
