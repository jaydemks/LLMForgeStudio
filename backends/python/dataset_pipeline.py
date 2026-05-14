import re
import unicodedata


def clean_text(text: str, cfg: dict) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if bool(cfg.get("NormalizeUnicode", True)):
        text = unicodedata.normalize("NFKC", text)

    if bool(cfg.get("CollapseWhitespace", False)):
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    if bool(cfg.get("EnableDeduplication", False)):
        if bool(cfg.get("RemoveDuplicateLines", False)):
            seen = set()
            lines = []
            for line in text.split("\n"):
                key = line.strip()
                if key and key in seen:
                    continue
                seen.add(key)
                lines.append(line)
            text = "\n".join(lines)

        if bool(cfg.get("RemoveDuplicateParagraphs", False)):
            seen = set()
            chunks = []
            for paragraph in text.split("\n\n"):
                key = re.sub(r"\s+", " ", paragraph.strip())
                if key and key in seen:
                    continue
                seen.add(key)
                chunks.append(paragraph)
            text = "\n\n".join(chunks)

    return text


def maybe_curriculum_view(train_ids, step: int, max_steps: int, cfg: dict):
    if not bool(cfg.get("CurriculumLearning", False)):
        return train_ids, 1.0

    ratio = float(cfg.get("CurriculumWarmupRatio", 0.2))
    ratio = min(0.9, max(0.05, ratio))
    warmup_steps = max(1, int(max_steps * ratio))
    if step >= warmup_steps:
        return train_ids, 1.0

    frac = max(0.25, (step + 1) / warmup_steps)
    keep = max(256, int(len(train_ids) * frac))
    return train_ids[:keep], frac
