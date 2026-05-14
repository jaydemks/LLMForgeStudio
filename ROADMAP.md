# LLM Forge Studio Roadmap (Serious Edition)

## Objective
Evolve from an educational local-LLM lab to a structured, production-style training platform while preserving the guided workflow (`Dataset -> Tokenization -> Model -> Training -> Generation`) in .NET/Avalonia.

## Release Notes Order (Draft)
This order is intended for GitHub release notes readability:
1. Data pipeline and large-corpus foundations
2. Training runtime serious upgrades (optimizer/scheduler/mp/distributed)
3. Cluster orchestration and artifacts
4. Alignment pipeline (SFT/DPO/RLHF, reward/safety)
5. Quantization + export targets (PTQ/QAT/ONNX/GGUF)
6. Eval suite and release gates (20-task harness, regression, RC scorecard)
7. UX guided flow and wizards
8. Tokenization evolution

## Validation Tracking
- Manual validation checklist: [RELEASE_VALIDATION_CHECKLIST.md](RELEASE_VALIDATION_CHECKLIST.md)
- Use validation IDs `VAL-01..VAL-20` to align roadmap and manual test outcomes.

## Status Legend
- `Completed Verified`: implemented and validated with build/tests in this repo.
- `Completed To Be Tested`: implemented, but requires manual/runtime validation.
- `Planned`: not implemented yet.

## Milestones

### M1 - Data & Pipeline Foundation
- `Completed Verified` Advanced cleaning hooks (unicode normalization, whitespace collapse). `VAL-02`
- `Completed Verified` Dedup hooks (line-level and paragraph-level). `VAL-02`
- `Completed Verified` Curriculum-learning hook (progressive corpus fraction). `VAL-03`
- `Completed Verified` Streaming dataset reader runtime (manifest shards -> stream cache file). `VAL-01`
- `Completed Verified` Sharded data format foundation + manifest (`dataset_manifest.json` + shard files).
- `Completed Verified` Dataset ingestion foundation for large corpora (folder -> shard manifest pipeline).
- `Completed Verified` Shard integrity verification (SHA-256 checksum validation).
- `Completed Verified` Resume state foundation (`dataset_resume_state.json` creation).
- `Completed Verified` Resume-aware runtime shard processing (`dataset_stream_cache.txt` + state updates). `VAL-01`
- `Completed Verified` Distributed dataloader foundation with deterministic shard shuffling and seed control. `VAL-01`

### M2 - Training Runtime Upgrades
- `Completed To Be Tested` Optimizers: AdamW, Lion, Adafactor (fallback-safe).
- `Completed To Be Tested` Schedulers: none, cosine, linear (+ warmup).
- `Completed To Be Tested` Mixed precision: fp16/bf16 autocast support.
- `Completed To Be Tested` Gradient clipping controls.
- `Completed To Be Tested` Periodic checkpointing.
- `Completed To Be Tested` Optional post-training dynamic quantization (CPU).
- `Completed To Be Tested` Basic distributed bootstrap (env-driven). `VAL-04`
- `Completed Verified` Distributed preflight gate in UI/start flow (blocks invalid WORLD_SIZE).
- `Completed To Be Tested` Multi-GPU training foundation (DDP/FSDP strategy field, gradient accumulation, auto device map, distributed preflight checks). `VAL-05`
- `Completed Verified` Orchestrated distributed pipeline stages foundation (data/preprocess/train/eval stage state + stage artifacts). `VAL-06`

### M3 - Cluster Orchestration
- `Completed Verified` Cluster profile manager (single-node, multi-node templates + JSON persistence).
- `Completed Verified` Cluster job descriptor artifact (`cluster_job_descriptor.json`) wired into training run folder.
- `Completed To Be Tested` Job orchestrator adapter layer (local scheduler first, cloud/HPC adapters later). `VAL-07`
- `Completed To Be Tested` Fault-tolerant run state foundation (`cluster_runner.py` + `cluster_run_state.json` + `cluster_heartbeat.json` + retry policy). `VAL-07`
- `Completed To Be Tested` SharedFS cluster mode (coordinator/worker queue with atomic claim + result/heartbeat), usable across multiple machines sharing one folder. `VAL-07`
- `Completed Verified` Central artifact registry foundation (`artifact_registry.json` indexing checkpoints/cluster/eval artifacts).

### M4 - Alignment & Post-Training
- `Completed To Be Tested` Alignment mode field in spec/manifest (foundation).
- `Completed Verified` Fine-tuning stage orchestration (`SFT -> DPO -> RLHF`) with guided setup and stage artifact (`fine_tuning_stages.json`). `VAL-08`
- `Completed Verified` SFT formatting foundation (JSONL prompt/response and messages -> chat training text conversion).
- `Completed Verified` RLHF pipeline inline collection path (UI collector + collected JSONL run artifact). `VAL-10`
- `Completed Verified` RLHF pipeline import path (JSONL human feedback import + run artifacts). `VAL-09`
- `Completed Verified` DPO formatting foundation (JSONL prompt/chosen/rejected -> preference training text conversion).
- `Completed Verified` Reward/safety foundation (reward-model toggle, safety policy mode, RLHF feedback import metadata + manifest annotations). `VAL-11`

### M5 - Quantization & Inference Optimization
- `Completed Verified` Post-training dynamic quantization (baseline). `VAL-12`
- `Completed Verified` PTQ calibration foundation (profile + calibration samples metadata in manifest). `VAL-12`
- `Completed Verified` QAT path foundation for selected model sizes (QAT intent fields + `qat_report.json` artifact/manifest metadata). `VAL-13`
- `Completed Verified` Export targets foundation (`ONNX`, `GGUF` placeholder metadata) with manifest export status. `VAL-14`
- `Completed Verified` Quantization profiles (`INT8`, `INT4`) foundation with quality/latency reports (`quantization_report.json` + `quantization_report.md`). `VAL-12`

### M6 - Evaluation Suite (20 Benchmarks Target)
- `Completed Verified` Core eval metrics in logs: perplexity + generalization gap. `VAL-15`
- `Completed Verified` Benchmark harness with 20 tasks (reasoning, coding, factuality, safety, robustness) via pack-driven runner. `VAL-15`
- `Completed Verified` Eval packs runner (lite): `quick(5)`, `standard(10)`, `full(20)` with `eval_summary.json` + `eval_scorecard.md`.
- `Completed To Be Tested` Data-driven benchmark packs via JSON config (`eval_benchmarks.json`).
- `Completed Verified` Eval trend export (`eval_trend.json`) for score trajectory tracking. `VAL-16`
- `Completed To Be Tested` Release gate fields in manifest (`evalReleaseGatePassed`, `evalReleaseGateThreshold`).
- `Completed Verified` Eval gate UI readout in Training dashboard (suite/avg/band/pass-threshold).
- `Completed Verified` Regression tracking across checkpoints and runs (`eval_regression.json`). `VAL-16`
- `Completed Verified` Scorecards + pass/fail gates for release candidates (`release_candidate_scorecard.md`). `VAL-17`

### M7 - UX Integration (Step-by-Step .NET)
- `Completed Verified` Guided Defaults Engine centralized per valori ideali (tokenizer + training profile).
- `Completed Verified` Advanced Training panel in UI with progressive disclosure.
- `Completed Verified` Advanced config persistence (save/load project roundtrip test).
- `Completed Verified` Preset family expansion: Tiny / Balanced / Serious / Research / Cluster (+ Custom fallback).
- `Completed Verified` Validation rules (initial): distributed env hint, CPU quantization hint, mixed-precision-on-CPU hint.
- `Completed Verified` In-app roadmap checklist in Guide section.
- `Completed Verified` Pre-training wizard text + preflight status panel in Training section.
- `Completed Verified` Guided wizards for multi-GPU setup, cluster profile, RLHF import, eval pack selection. `VAL-18`
- `Completed Verified` End-to-end guided workflow runtime (`Dataset -> Tokenization -> Model -> Training -> Generation`) validated in manual UI runs. `VAL-19`

### M8 - Tokenization Evolution
- `Completed Verified` Byte-level BPE tokenizer option.
- `Completed Verified` Unigram tokenizer option (lite implementation).
- `Completed Verified` WordPiece tokenizer option (lite implementation).
- `Completed Verified` Guided tokenizer profiles with recommended value badges per mode (incl. ideal-values live readout in Tokenization UI).

## Verification Notes (2026-05-14)
- `Completed Verified`: `dotnet build LLMForgeStudio.sln` passed (0 errors).
- `Completed Verified`: `dotnet test LLMForgeStudio.sln` passed (23/23 tests).
- Runtime behaviors in Python backend are marked `Completed To Be Tested` until end-to-end manual training runs confirm them on real hardware (`VAL-01..VAL-20`).
