# LLM Forge Studio - Release Notes v1.0.1
Date: 2026-05-15

## Summary
Introduced the first complete Ollama-oriented fine-tuning workspace, separated from the from-scratch training path.

## Added
- Dedicated `Fine-Tuning (Ollama)` tab with its own UI and runtime flow.
- Auto base-model resolution from current run (`model.pt`) to avoid manual base-path typing.
- Required `New model name` field for generated fine-tuned outputs.
- Sequential guided pipeline with stage gating:
  1. `Prepare Run`
  2. `Start Fine-Tune`
  3. `Convert to GGUF`
  4. `Finalize Ollama Export`
- Pipeline progress UI (step text + progress bar).
- Runtime controls for fine-tuning (`Start` / `Cancel`).

## Changed
- Fine-tuning artifacts are now isolated under dedicated run folders:
  - `<run>/fine_tuning_ollama/`
  - `<run>/fine_tuning_ollama/exports/ollama_finetune/`

## Technical
- Added backend runner: `backends/python/ollama_finetune_runner.py`.
- Generated artifacts include:
  - `ollama_finetune_job.json`
  - `ollama_finetune_log.jsonl`
  - `ollama_finetune_manifest.json`
  - checkpoint/export support files in `exports/ollama_finetune`
- Deprecated legacy stub backend (`ollama_finetune_stub.py`) with explicit fail-fast behavior.
- Added deterministic stage-gating state logic in ViewModel.

## Known Limits
- GGUF conversion is not yet fully native in-app.
- `Convert to GGUF` requires converter availability/configuration (e.g. `LLMFORGE_GGUF_CONVERTER`).
- Final Ollama-ready handoff depends on successful GGUF conversion.

## Upgrade Notes
- Existing projects remain compatible.
- Fine-tuning no longer writes into Ollama internal storage directly.
