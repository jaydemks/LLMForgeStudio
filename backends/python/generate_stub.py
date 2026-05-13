import argparse
import json
from pathlib import Path

import torch

from train_stub import GPT, GPTConfig


def sample_top_k(logits, top_k, temperature):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = v[..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    if checkpoint.name.endswith(".json"):
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        model_path = Path(manifest["modelWeightsPath"])
        tokenizer_path = Path(manifest["tokenizerPath"])
    else:
        raise RuntimeError("Expected checkpoint manifest json path")

    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    stoi = tok["stoi"]
    itos = {int(k): v for k, v in tok["itos"].items()}

    payload = torch.load(model_path, map_location="cpu")
    cfg = GPTConfig(**payload["config"])
    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except Exception:
            device = "cpu"

    model = GPT(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(args.max_new_tokens):
        idx_cond = idx[:, -cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        next_id = sample_top_k(logits, args.top_k, args.temperature)
        idx = torch.cat([idx, next_id], dim=1)

    out = "".join(itos.get(i, "?") for i in idx[0].tolist())
    print(json.dumps({"text": out}))


if __name__ == "__main__":
    main()
