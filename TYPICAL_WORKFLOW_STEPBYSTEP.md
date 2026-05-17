# LLM Forge Studio From Scratch End-to-End Quickstart Guide
Version: `v1.0.0`
Updated: `2026-05-16`

This document describes a practical end-to-end workflow to:
- build a model from scratch,
- refine it with a focused fine-tuning stage,
- finalize an Ollama-ready bundle,
- and validate the model through real local chat queries.

## Goal
Run one full real pipeline:
1. Gather dataset from provider links and/or your own local files/folders (legally safe sources)
2. Build tokenizer and train from scratch
3. Fine-tune with cleaner chat/instruction material
4. Convert/finalize for Ollama
5. Query the model in Ollama and evaluate output quality

## 1) Gather Dataset (Providers and/or Local Files)
You can use:
- only provider links,
- only your own local files/folders,
- or a combination of both.

Use 2-3 different sources only if you want to test merge/composition.

Suggested provider entry points:
- Hugging Face (MIT): https://huggingface.co/datasets?license=license%3Amit
- Hugging Face (Apache-2.0): https://huggingface.co/datasets?license=license%3Aapache-2.0
- Hugging Face (CC0): https://huggingface.co/datasets?license=license%3Acc0-1.0
- Kaggle datasets (license-filter entrypoint): https://www.kaggle.com/datasets?license=gpl
- Zenodo search: https://zenodo.org/search
- EU open datasets: https://data.europa.eu/en/download-datasets

In app (`Gather Dataset`):
1. Add source A (provider URL or local folder)
2. Check/accept license gate only if permissive and valid
3. Fetch source
4. Repeat for source B/C

## 2) Convert + Merge + Validate
1. If parquet sources are present, run `Convert parquet`
2. Configure merge:
   - source toggles ON/OFF
   - contribution weights (example: `1.0`, `0.7`, `0.5`)
   - dedup policy (`line` or `paragraph`)
3. Run `Merge`
4. Run `Validate`
5. Run `Apply recommendation`

Expected outputs include provenance/analytics/readiness artifacts in the gather workspace.

## 3) Tokenization
1. Go to `Tokenization`
2. Keep recommended tokenizer mode/preset (or select Byte-level BPE)
3. Train tokenizer
4. Confirm `tokenizer.json` is produced

## 4) Model From Scratch Training
1. Go to `Model` and keep a realistic small/medium config for first full run
2. Go to `Training`:
   - choose stable profile (`Balanced` or recommendation)
   - enable evaluation suite (`quick-5` or `standard-10`)
   - keep export path enabled according to your test plan
3. Start training and wait for completion

Expected minimum artifacts:
- `model.pt`
- `checkpoint_manifest.json`
- `train_log.jsonl`

## 5) Fine-Tuning Material (Middle Phase)
For fine-tuning, use cleaner and more focused conversation/instruction data.

Recommended sources for this phase:
- instruction/chat-style datasets from permissive-licensed providers
- curated internal JSONL conversation data

Preferred formats:
- `prompt/response` JSONL
- `messages` JSONL

Goal:
Improve chat behavior quality for final Ollama usage.

## 6) Fine-Tuning (Ollama Tab)
In `Fine-Tuning (Ollama)`:
1. `Step 1: Prepare`
2. `Step 2: Start Fine-Tune`
3. `Step 3: Convert to GGUF`
4. `Step 4: Finalize Export`

Required final state:
- bundle status `ready`

Expected final artifacts in `fine_tuning_ollama/exports/ollama_finetune/`:
- `model.gguf`
- `Modelfile`
- `ollama_handoff_status.json` (`status: ready`)
- `ollama_bundle_validation.json`

## 7) Import to Ollama + Real Chat Validation
1. Take the finalized bundle
2. Create/import the model in local Ollama
3. Run a representative set of prompts for your target use case (for example 10-20 prompts)
4. Check:
   - coherence
   - instruction-following
   - style consistency
   - stability across turns

## 8) PASS / FAIL Criteria
PASS when:
1. The pipeline completes without blocking failures
2. Final Ollama handoff status is `ready`
3. The model answers in Ollama with acceptable conversational quality for your target

FAIL when:
1. conversion/finalization cannot produce `ready`
2. artifacts are missing/inconsistent
3. chat behavior is unusable for intended purpose

## Legal and Compliance Notes
- Use only datasets with clear permissive licenses
- Respect per-dataset usage conditions
- If the app license gate blocks a source, replace the source (do not bypass)

## Reference Links
- Hugging Face MIT filter: https://huggingface.co/datasets?license=license%3Amit
- Zenodo licensing docs: https://help.zenodo.org/docs/deposit/describe-records/licenses/
- Zenodo permission/licensing FAQ: https://support.zenodo.org/help/en-gb/2-content/21-can-i-get-permission-to-use-a-specific-record
- data.europa.eu dataset hub: https://data.europa.eu/en/download-datasets
- Kaggle datasets entrypoint: https://www.kaggle.com/datasets?license=gpl
- OpenML dataset metadata/license field: https://docs.openml.org/reference/datasets/dataset/
