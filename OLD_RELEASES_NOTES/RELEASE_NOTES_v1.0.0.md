# LLM Forge Studio v1.0.0

## Summary
LLM Forge Studio reaches `v1.0.0` as a serious, guided training application in .NET/Avalonia, evolving from early local-LLM experimentation to a structured platform for data pipelines, training runtime controls, alignment flows, quantization/export, and benchmark-driven release gating.

This release keeps the user-friendly step-by-step flow:
`Dataset -> Tokenization -> Model -> Training -> Generation`

## Validation Status
- Local/manual validation completed: `15/17` tests (`PASS`)
- Pending community/hardware validation: `2/17`
  - `Test 2` (`VAL-04`, `VAL-05`): real distributed + multi-GPU runtime validation
  - `Test 4` (`VAL-07`): real cluster runtime validation (multi-machine)

Reference docs:
- `RELEASE_VALIDATION_CHECKLIST.md`
- `ROADMAP.md`

## Major Capabilities Delivered

### 1) Data Pipeline and Large-Corpus Foundations
- Advanced cleaning hooks (unicode normalization, whitespace collapse)
- Dedup hooks (line/paragraph)
- Curriculum-learning hook
- Sharded dataset manifest pipeline + integrity checks
- Resume-aware streaming artifacts (`dataset_stream_cache.txt`, `dataset_resume_state.json`)
- Deterministic shard shuffle/seed foundation

### 2) Training Runtime Serious Upgrades
- Optimizer choices: `AdamW`, `Lion`, `Adafactor`
- Scheduler choices: `none`, `cosine`, `linear` (+ warmup)
- Mixed precision controls (`fp16`/`bf16`)
- Gradient clipping controls
- Checkpointing controls
- Distributed preflight protections in UI

### 3) Cluster Orchestration Foundations
- Cluster profile manager and descriptors
- Run state + heartbeat artifacts
- SharedFS coordinator/worker mode (queue/claim/result/heartbeat) for multi-machine testing
- Central artifact registry (`artifact_registry.json`)

### 4) Alignment Pipeline (SFT / DPO / RLHF)
- Fine-tuning orchestration (`SFT -> DPO -> RLHF`)
- SFT and DPO formatting foundations
- RLHF import path (`jsonl-human-feedback` / external)
- RLHF inline collection path (UI prompt/chosen/rejected)
- Reward/safety metadata hooks in manifest

### 5) Quantization and Export
- PTQ INT8 + INT4 profile runs with report artifacts
- Quantization reports: `quantization_report.json` + `quantization_report.md`
- QAT foundation artifacts: `qat_report.json` + manifest metadata
- ONNX export path with fallback artifact (`model.onnx.export_error.txt` when dependency missing)
- GGUF placeholder export artifact (`model.gguf.placeholder.json`)

### 6) Evaluation Suite and Release Gates
- Eval packs: `quick-5`, `standard-10`, `full-20`
- Full-20 benchmark harness path
- Eval artifacts: summary, scorecard, trend, regression
- Release candidate scorecard (`release_candidate_scorecard.md`) with pass/fail verdict and weakest tasks

### 7) UX and Guided Flow Enhancements
- Header shortcut: `Guide` button moved to top-right actions (with Settings/Load/Save)
- Guided defaults for tokenizer/training profiles
- Advanced Training panel with progressive disclosure
- Save/load project persistence for advanced fields
- Guided wizard blocks for multi-GPU, cluster, RLHF, eval
- Improved preflight/readiness feedback
- Visual emphasis for wizard/dynamic guidance text via dedicated background panels
- End-to-end workflow validated in repeated UI runs

### 8) Tokenization Evolution
- Byte-level BPE option
- Unigram option
- WordPiece option
- Live ideal-values guidance/readouts

## Operational Notes
- Quantization validation currently uses the CPU path for PTQ execution.
- ONNX export requires ONNX dependency in backend environment; fallback artifact is expected when unavailable.
- Cluster/multi-GPU final verification should be completed by community testers with suitable hardware and topology.

## Recommended Community Validation Focus
1. Multi-GPU DDP/FSDP runtime behavior under real workloads.
2. SharedFS cluster coordinator/worker behavior across multiple machines.
3. Stability of retries/heartbeat under fault injection.

## Upgrade Context
This release supersedes `v0.1.x` with a significantly broader training/control surface and release-gate-oriented evaluation artifacts, while preserving user-friendly operation for local testing.
