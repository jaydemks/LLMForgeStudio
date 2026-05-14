import json
import random
from pathlib import Path


DEFAULT_FALLBACK = "hello world\n" * 1000


def _read_manifest(dataset_path: Path):
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    shards = payload.get("ShardItems") or payload.get("shardItems")
    if isinstance(shards, list) and shards and isinstance(shards[0], dict):
        shard_paths = [item.get("RelativePath") or item.get("relativePath") for item in shards]
    else:
        shard_paths = payload.get("Shards") or payload.get("shards") or []
    shard_paths = [s for s in shard_paths if isinstance(s, str) and s.strip()]
    return shard_paths


def _load_resume_state(state_path: Path):
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_resume_state(state_path: Path, manifest_path: Path, last_index: int):
    state = {
        "manifest_path": str(manifest_path),
        "last_completed_shard_index": int(last_index),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _deterministic_shuffle(items, seed: int):
    rng = random.Random(seed)
    copied = list(items)
    rng.shuffle(copied)
    return copied


def _partition_for_rank(shards, rank: int, world_size: int):
    if world_size <= 1:
        return shards
    return [s for i, s in enumerate(shards) if (i % world_size) == rank]


def load_dataset_text(
    dataset_path: Path,
    output_dir: Path,
    resume_enabled: bool = True,
    shuffle_enabled: bool = True,
    shuffle_seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> str:
    if not dataset_path.exists():
        return DEFAULT_FALLBACK

    if not dataset_path.name.lower().endswith("dataset_manifest.json"):
        return dataset_path.read_text(encoding="utf-8")

    shard_refs = _read_manifest(dataset_path)
    if not shard_refs:
        return DEFAULT_FALLBACK

    if shuffle_enabled:
        shard_refs = _deterministic_shuffle(shard_refs, shuffle_seed)

    shard_refs = _partition_for_rank(shard_refs, rank=rank, world_size=world_size)
    if not shard_refs:
        return DEFAULT_FALLBACK

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_suffix = f"_r{rank}" if world_size > 1 else ""
    cache_path = output_dir / f"dataset_stream_cache{rank_suffix}.txt"
    state_path = output_dir / f"dataset_resume_state{rank_suffix}.json"

    start_idx = 0
    if resume_enabled and cache_path.exists() and state_path.exists():
        state = _load_resume_state(state_path)
        if state and state.get("manifest_path") == str(dataset_path):
            start_idx = int(state.get("last_completed_shard_index", -1)) + 1

    if start_idx <= 0:
        cache_path.write_text("", encoding="utf-8")

    base_dir = dataset_path.parent
    with cache_path.open("a", encoding="utf-8") as out:
        for idx, shard_ref in enumerate(shard_refs):
            if idx < start_idx:
                continue

            shard_path = Path(shard_ref)
            if not shard_path.is_absolute():
                shard_path = base_dir / shard_path
            if not shard_path.exists():
                continue

            text = shard_path.read_text(encoding="utf-8")
            if cache_path.stat().st_size > 0:
                out.write("\n\n")
            out.write(text)
            _save_resume_state(state_path, dataset_path, idx)

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    return DEFAULT_FALLBACK
