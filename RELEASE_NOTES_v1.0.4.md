# LLM Forge Studio - Release Notes v1.0.4 (Critical Corrections)
Date: 2026-05-17

## Summary
Critical correction release focused on tokenizer/training consistency, telemetry accuracy, and reliability safeguards in the guided from-scratch pipeline.

## Critical Fixes
- Tokenization/training strict consistency enforcement:
  - backend now requires valid tokenizer state when tokenizer-driven training is expected
  - removed silent fallback behavior that could mask tokenizer mismatches
  - explicit fail-fast path with actionable error (`TOKENIZER_STATE_REQUIRED`)
- Tokenizer progress telemetry correctness:
  - `Processed`/`Rate` units now reflect selected tokenizer mode
  - unit mapping:
    - `Character` / `Hierarchical` -> `chars`
    - `ByteLevelBPE` -> `bytes`
    - `Word` / `Unigram` / `WordPiece` -> `words`
    - `SimpleBPE` / `Hybrid` -> `symbols`
- Generation compatibility hardening for non-char vocabularies:
  - prompt encoding path improved for tokenizer vocab workflows
  - reduced mismatch risk between UI tokenization and generation backend behavior
- Profile-driven generation defaults:
  - sampling defaults (`temperature`, `top_k`, `seed`, `max_new_tokens`) are now auto-applied with training profile presets
  - defaults are also tokenizer-aware to keep chat output more stable out-of-the-box
- ONNX export robustness:
  - default profile presets no longer auto-enable ONNX export
  - when ONNX is manually enabled but dependency is missing, export is marked as `skipped_missing_dependency` with clear note artifact instead of opaque failure
  - backend dependency baseline now includes `onnx` in `backends/python/requirements.txt` for open-source reproducibility

## New Features
- Training Advisor (post-training intelligent guidance):
  - after each training run, UI can highlight key controls in red when loss signals indicate likely tuning issues
  - advisor suggests concrete `↑/↓` adjustments and target values
  - suggestions auto-hide per field as soon as the user edits that control
  - covers both base training controls and selected `Advanced Training` controls

## Stability & Guardrails
- Training quality gate threshold tuning to reduce false positives on valid chat corpora.
- Better deterministic behavior for project reload when tokenizer/training state exists.
- Additional reliability hardening around long-running training runtime events and state transitions.
- Recommendation tuning for medium datasets:
  - profile auto-recommendation now avoids pushing `Serious` too early on mid-scale corpora
  - `Balanced` is preferred for typical ~10k chat sample runs to reduce overfitting risk
  - exception added: when dataset is detected as structured chat (prompt/response or messages JSONL), medium-scale recommendations now prefer `Serious`
- Serious profile default retune for mid-scale corpora:
  - reduced baseline step floor (`1800` instead of `3000`)
  - lower learning rate and earlier/more frequent evaluation checkpoints
  - stronger text dedup/cleanup defaults (`paragraph dedup + whitespace collapse`)
  - launch safeguard now aligns with new floor to avoid forced overtraining
- Validation sample refresh:
  - `samples/validation/it_chat_conversation_4k` published as `v4_diverse_chat_first`
  - reduced fixed boilerplate patterns and increased response-shape diversity
  - optimized for faster local chat-model validation with lower template-collapse risk
- Dataset folder import hardening:
  - markdown/doc files (`.md`, including README) are now excluded from dataset/training ingestion paths
  - metadata/config json files (for example `metadata.json`, `dataset_info.json`, `dataset_infos.json`, `card.json`, `manifest.json`) are now excluded from trainable-source scans
  - prevents accidental corpus contamination from documentation files when using `Import Folder`
- Post-training tuning advisor in UI:
  - after training ends, key fields (`Batch size`, `Max steps`, `Learning rate`, `Eval every`) can be highlighted in red when loss signals suggest adjustments
  - concise `↑/↓` recommendations are shown next to the field and auto-hide as soon as the user edits that field
  - advisor coverage extended into `Advanced Training` for impactful toggles (`Grad clipping`, `Dedup dataset`, `Paragraph dedup`, `Collapse whitespace`, `Curriculum learning`) with the same auto-hide-on-edit behavior

## Notes
- This version is a correction pass on core pipeline behavior, not a UI polish release.
- Future roadmap/draft items have been moved forward to `v1.0.5` draft track.
