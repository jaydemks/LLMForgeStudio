import argparse
import json
import re
from pathlib import Path

import torch

from train_stub import GPT, GPTConfig


def sample_top_k_top_p(logits, top_k, top_p, temperature):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = v[..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


USER_TURN_RE = re.compile(
    r"(?im)^\s*(mi dai|dammi|puoi|potresti|scrivimi|spiegami|fammi|come si|what|can you|please)\b"
)


def normalize_completion(text: str, prompt: str) -> str:
    if not text:
        return text

    prompt_clean = (prompt or "").strip().lower()
    lines = text.splitlines()
    kept = []
    for line in lines:
        probe = line.strip()
        if not probe:
            kept.append(line)
            continue

        probe_l = probe.lower()
        # Drop user-echo lines instead of truncating the whole answer.
        if prompt_clean and (probe_l == prompt_clean or probe_l in prompt_clean or prompt_clean in probe_l):
            continue
        if USER_TURN_RE.match(probe):
            continue
        kept.append(line)

    out = "\n".join(kept).strip()
    if out:
        return out
    return text.strip()


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or generated_ids.numel() == 0:
        return logits

    unique_ids = torch.unique(generated_ids)
    for token_id in unique_ids:
        tid = int(token_id.item())
        if tid < 0 or tid >= logits.shape[-1]:
            continue
        value = logits[0, tid]
        logits[0, tid] = value / penalty if value > 0 else value * penalty
    return logits


def block_repeated_ngrams(logits: torch.Tensor, all_ids: list[int], ngram_size: int) -> torch.Tensor:
    if ngram_size <= 1 or len(all_ids) < ngram_size - 1:
        return logits

    prefix = tuple(all_ids[-(ngram_size - 1):])
    banned = set()
    for i in range(0, len(all_ids) - ngram_size + 1):
        chunk = tuple(all_ids[i:i + ngram_size])
        if chunk[:-1] == prefix:
            banned.add(chunk[-1])

    if not banned:
        return logits

    for token_id in banned:
        if 0 <= token_id < logits.shape[-1]:
            logits[0, token_id] = float("-inf")
    return logits


def encode_with_tokenizer(prompt_text: str, stoi: dict[str, int], tokenizer_type: str) -> list[int]:
    if tokenizer_type == "char":
        ids = [stoi[ch] for ch in prompt_text if ch in stoi]
        return ids if ids else [0]

    # UI tokenizer-state with ByteLevelBPE vocabulary uses hex-byte symbols and
    # merged symbols separated by "|", e.g. "43|65|72". Handle this path explicitly.
    if _looks_like_bytelevel_vocab(stoi):
        return _encode_bytelevel_prompt(prompt_text, stoi)

    vocab_tokens = [t for t in stoi.keys() if t]
    if not vocab_tokens:
        return [0]
    vocab_set = set(vocab_tokens)
    max_len = max(len(t) for t in vocab_tokens)

    i = 0
    ids: list[int] = []
    while i < len(prompt_text):
        matched = None
        upper = min(max_len, len(prompt_text) - i)
        for k in range(upper, 0, -1):
            piece = prompt_text[i : i + k]
            if piece in vocab_set:
                matched = piece
                break
        if matched is not None:
            ids.append(stoi[matched])
            i += len(matched)
            continue

        ch = prompt_text[i]
        if ch in stoi:
            ids.append(stoi[ch])
        else:
            ids.append(0)
        i += 1

    return ids if ids else [0]


_HEX_RE = re.compile(r"^[0-9A-Fa-f]{2}$")


def _looks_like_bytelevel_vocab(stoi: dict[str, int]) -> bool:
    if not stoi:
        return False
    sample = list(stoi.keys())[:200]
    if not sample:
        return False

    hits = 0
    for tok in sample:
        parts = tok.split("|")
        if parts and all(_HEX_RE.match(p or "") for p in parts):
            hits += 1
    return (hits / max(1, len(sample))) >= 0.35


def _encode_bytelevel_prompt(prompt_text: str, stoi: dict[str, int]) -> list[int]:
    byte_tokens = [f"{b:02X}" for b in prompt_text.encode("utf-8", errors="replace")]
    if not byte_tokens:
        return [0]

    max_parts = 1
    for tok in stoi.keys():
        c = tok.count("|") + 1
        if c > max_parts:
            max_parts = c

    ids: list[int] = []
    i = 0
    n = len(byte_tokens)
    while i < n:
        matched = None
        upper = min(max_parts, n - i)
        for span in range(upper, 0, -1):
            candidate = "|".join(byte_tokens[i : i + span])
            if candidate in stoi:
                matched = candidate
                break
        if matched is not None:
            ids.append(int(stoi[matched]))
            i += matched.count("|") + 1
            continue

        base = byte_tokens[i]
        if base in stoi:
            ids.append(int(stoi[base]))
        else:
            ids.append(0)
        i += 1

    return ids if ids else [0]


def decode_ids_with_tokenizer(ids: list[int], itos: dict[int, str], tokenizer_type: str) -> str:
    if tokenizer_type == "char":
        return "".join(itos.get(i, "?") for i in ids)

    if _looks_like_bytelevel_itos(itos):
        return _decode_bytelevel_ids(ids, itos)

    return "".join(itos.get(i, "?") for i in ids)


def _looks_like_bytelevel_itos(itos: dict[int, str]) -> bool:
    if not itos:
        return False
    sample = list(itos.values())[:200]
    if not sample:
        return False

    hits = 0
    for tok in sample:
        parts = str(tok).split("|")
        if parts and all(_HEX_RE.match(p or "") for p in parts):
            hits += 1
    return (hits / max(1, len(sample))) >= 0.35


def _decode_bytelevel_ids(ids: list[int], itos: dict[int, str]) -> str:
    out = bytearray()
    for tid in ids:
        tok = itos.get(int(tid), "")
        if not tok:
            continue
        for part in str(tok).split("|"):
            if _HEX_RE.match(part or ""):
                out.append(int(part, 16))
    if not out:
        return ""
    return out.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    if checkpoint.name.endswith(".json"):
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        model_path = Path(manifest["modelWeightsPath"])
        tokenizer_path = Path(manifest["tokenizerPath"])
    elif checkpoint.suffix.lower() == ".pt":
        # Fallback compatibility: allow direct model checkpoint path.
        model_path = checkpoint
        tokenizer_path = checkpoint.with_name("tokenizer.json")
    else:
        raise RuntimeError("Expected checkpoint manifest json path or model .pt path")

    if not model_path.exists():
        raise RuntimeError(f"Model checkpoint not found: {model_path}")
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer file not found: {tokenizer_path}")

    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    stoi = tok["stoi"]
    tokenizer_type = str(tok.get("type", "char"))
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

    prompt_text = args.prompt
    lowered = (prompt_text or "").lower()
    has_chat_markers = any(x in lowered for x in ["<|assistant|>", "<|user|>", "assistant:", "user:", "risposta:", "domanda:"])
    if not has_chat_markers:
        prompt_text = (
            "Istruzione: rispondi solo alla domanda dell'utente in modo diretto, "
            "senza generare nuove domande o prompt.\n"
            f"Domanda: {prompt_text}\n"
            "Risposta:"
        )

    ids = encode_with_tokenizer(prompt_text, stoi, tokenizer_type)
    prompt_len = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    stop_markers = [
        "<|user|>",
        "\nRichiesta #",
        "\nPrompt:",
        "\nUser:",
        "\n### Instruction",
    ]

    for _ in range(args.max_new_tokens):
        idx_cond = idx[:, -cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        logits = apply_repetition_penalty(logits, idx[0], float(args.repetition_penalty))
        logits = block_repeated_ngrams(logits, idx[0].tolist(), int(args.no_repeat_ngram_size))
        next_id = sample_top_k_top_p(logits, args.top_k, float(args.top_p), args.temperature)
        idx = torch.cat([idx, next_id], dim=1)

        # Early-stop once assistant output starts turning into a new prompt/turn.
        generated = decode_ids_with_tokenizer(idx[0].tolist()[prompt_len:], itos, tokenizer_type)
        if any(marker in generated for marker in stop_markers):
            break

    full = decode_ids_with_tokenizer(idx[0].tolist(), itos, tokenizer_type)
    completion = decode_ids_with_tokenizer(idx[0].tolist()[prompt_len:], itos, tokenizer_type)

    # Prefer assistant-only text when tag is present.
    marker = "<|assistant|>"
    if marker in full:
        completion = full.split(marker)[-1]

    # Clamp to first assistant answer and drop prompt-like continuation tails.
    for m in stop_markers:
        pos = completion.find(m)
        if pos >= 0:
            completion = completion[:pos]
    completion = normalize_completion(completion.strip(), args.prompt)

    print(json.dumps({"text": completion, "full_text": full}))


if __name__ == "__main__":
    main()
