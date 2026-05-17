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
9. Dataset gathering & smart ingestion
10. Ollama integration track (bundle export + dedicated fine-tuning workspace)
11. Frontier hardening track
12. End-to-end production closure track
13. Multi-target export ecosystem track (post-Ollama)

## Validation Tracking
- Manual validation checklist: [RELEASE_VALIDATION_CHECKLIST.md](RELEASE_VALIDATION_CHECKLIST.md)
- Use validation IDs `VAL-01..VAL-37`, `M11-01..M11-07`, `M12-01..M12-08`, and `M13-01..M13-07` to align roadmap and manual test outcomes.

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
- `Completed To Be Tested` Cluster live sync observability panel foundation (coordinator/worker realtime heartbeat readout, node list, shared-root queue counters, and remote GPU telemetry when reported by nodes). `VAL-37`
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
- `Completed Verified` Export targets foundation (`ONNX`, `GGUF` runtime conversion status artifact) with manifest export status. `VAL-14`
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

### M9 - Ollama Integration & Dedicated Fine-Tuning UI
- `Completed Verified` Manual handoff Ollama export bundle in run directory (`exports/ollama`) with `model.gguf` + `Modelfile` packaging rules. `VAL-21`
- `Completed Verified` Dedicated `Export for Ollama` UI action with readiness gating (enabled only when GGUF artifact is really available). `VAL-22`
- `Completed Verified` Dedicated `Fine-Tuning (Ollama)` tab separated from from-scratch workflow (new layout + dedicated controls). `VAL-23`
- `Completed Verified` Fine-tuning runtime integration for local Ollama-oriented adapters (separated backend path + dedicated artifacts). `VAL-24`
- `Completed Verified` Fine-tuning export packaging for Ollama handoff (`exports/ollama_finetune`) with run metadata and reproducibility notes. `VAL-24`
- `Completed Verified` Step-by-step gated pipeline UX for fine-tune flow (`Prepare -> Start -> Convert -> Finalize`) with sequential button enablement and pipeline progress bar.

### M10 - Dataset Gathering & Smart Ingestion (Immediate Priority)
- `Completed Verified` New dedicated UI section `Gather Dataset` for end-to-end dataset acquisition and preparation. `VAL-25`
- `Completed To Be Tested` Integrated remote dataset retrieval workflows (URL/file/folder source staging to local workspace). `VAL-25`
- `Completed Verified` Mandatory visual+functional license gate before source fetch (explicit user acknowledgement required).
- `Completed To Be Tested` Hugging Face dataset license resolver + permissive-license allowlist gate (blocks restricted/unknown licenses before fetch). `VAL-25`
- `Completed To Be Tested` Parquet ingestion/conversion pipeline to LLM Forge compatible formats (`jsonl`) via integrated backend script. `VAL-26`
- `Completed Verified` Unified dataset workspace manager to combine:
  - downloaded sources
  - user local files/folders
  - generated/converted outputs
- `Completed Verified` Automated dataset validation checks with immediate training-readiness feedback:
  - size/coverage warnings
  - format/schema consistency checks
  - language/domain hints
  - duplicate/noise quality checks `VAL-27`
- `Completed Verified` Smart recommendations engine connected to existing LLM Forge configs:
  - recommended tokenizer mode/preset based on gathered dataset
  - recommended training profile/preset based on dataset scale/quality
  - apply-to-config action before training start `VAL-28`
- `Completed Verified` Dataset-to-pipeline handoff actions to move directly from gathered dataset into: `VAL-29`
  1. from-scratch training flow
  2. fine-tuning flow
- `Completed Verified` Guided sequential Gather UX with numbered steps (`1..7`), stage-gated button enablement, and async progress feedback to keep UI responsive during long operations. `VAL-25`
- `Completed Verified` Multi-provider dataset connectors architecture foundation (provider contract + provider detection + per-provider fetch/compliance hooks + unsupported-provider graceful block). `VAL-30`
- `Completed To Be Tested` Provider set expansion beyond Hugging Face (HTTP/local/HF enabled in current phase), with per-provider compliance hooks. `VAL-30`
- `Completed To Be Tested` Multi-source dataset composition (staged sources + merge into unified dataset artifact):
  - single dataset from one provider
  - one dataset built from multiple providers
  - one dataset built from multiple datasets across one or more providers `VAL-31`
- `Completed Verified` Advanced dataset merge/orchestration workspace with source-level toggles, weighting, dedup policies, and provenance tracking (`dataset_merged_provenance.json`). `VAL-36`
- `Completed Verified` Per-source and merged-dataset legal policy checks foundation:
  - license compatibility matrix across mixed sources
  - automatic block on restricted/unknown licenses
  - explicit user acknowledgement + traceable compliance snapshot `VAL-32`
- `Completed Verified` Automatic schema harmonization for mixed-source ingestion foundation (JSONL/JSON/CSV normalization, chat/instruction extraction, unified merge text output). `VAL-33`
- `Completed Verified` Advanced quality analytics dashboard foundation for gathered/merged datasets:
  - coverage, duplicates, language mix, format drift, outlier/noise indicators
  - readiness score and action recommendations `VAL-34`
- `Completed Verified` Tokenizer/training bootstrap from gathered datasets with one-click apply + direct jump to Tokenization section. `VAL-35`

### M11 - Frontier Hardening (Deferred Until M10 Completes)
- `Completed To Be Tested` Native GGUF conversion integration foundation in-app (internal converter runtime orchestration script + converter status artifact + compatibility metadata path).  
  Note: still requires configured converter executable in current phase. `M11-01`
- `Completed To Be Tested` Ollama-ready finalization automation path (auto convert+finalize after successful fine-tuning when packaging is enabled and converter succeeds). `M11-02`
- `Completed To Be Tested` Robust failure taxonomy + retry UX foundation for conversion/export stages (`errorCode`-based status + retry attempts + actionable status/log messages). `M11-03`
- `Completed To Be Tested` Fine-tuning engine hardening foundation for larger checkpoints (adaptive batch fallback on OOM + resume state file + runtime diagnostics snapshot). `M11-04`
- `Completed To Be Tested` End-to-end integration tests foundation for `from-scratch -> export -> fine-tune -> convert -> finalize`:
  - scenario-pack suite runner (`backends/python/e2e_release_gate.py --scenario-pack ...`)
  - aggregate CI-friendly artifacts (`e2e_suite_summary.json` + `e2e_suite_summary.md`) `M11-05`
- `Completed To Be Tested` Reproducible runtime packaging foundation (runtime environment snapshots/artifacts per run). `M11-06`
- `Completed To Be Tested` Multi-backend fine-tuning architecture foundation (backend target selector + backend field propagated in job/manifest, Ollama-first). `M11-07`

### M12 - End-to-End Production Closure (Final Functional Gap)
- `Completed To Be Tested` Native GGUF conversion internalized foundation:
  - converter resolution order (`env -> bundled -> fallback copy-existing-gguf`)
  - converter runtime compatibility metadata and deterministic status artifact path
  - semantic version probe (`--version`) + compatibility matrix gate (`converter_compatibility_matrix.json`)
  - reduced dependency on manual env-only configuration (bundled path supported)
  - **Remaining to reach 100% closure (`M12-01` done criteria):**
    1. ship converter binary/script set in release artifacts for supported OS targets
    2. validate runtime behavior on real converters/checkpoints across supported OS targets
    3. guarantee conversion availability in default install path (no manual copy step) `M12-01`
- `Completed To Be Tested` Ollama finalization deterministic readiness foundation:
  - explicit bundle validation artifact before final handoff state
  - `ready` state emitted only when required export files are complete/non-empty
  - blocked status with actionable message on validation failure `M12-02`
- `Completed To Be Tested` Hardware/device classification hardening:
  - filter virtual display adapters from detected GPU list
  - classify training-capable accelerators vs non-training adapters
  - hardware-aware recommendation guardrails for CPU/GPU/VRAM tiers `M12-03`
- `Completed To Be Tested` Legal/compliance hard gate completion for dataset composition:
  - strict per-source + merged-source compatibility matrix
  - non-overridable block for incompatible/restricted combinations
  - compliance report artifact for release/legal audit `M12-04`
- `Completed To Be Tested` Full automated end-to-end integration suite:
  - `dataset -> tokenization -> model -> training -> eval -> export -> fine-tune -> convert -> finalize`
  - reproducible CI scenario packs with pass/fail gate artifacts `M12-05`
- `Completed To Be Tested` Runtime packaging closure:
  - lockfile-driven Python/toolchain reproducibility
  - deterministic runtime environment snapshot validation
  - preflight mismatch detection with actionable remediation `M12-06`
- `Completed To Be Tested` Long-run resilience closure:
  - robust resume/recovery semantics for training, fine-tuning, conversion, and finalization
  - crash-safe stage checkpoints and safe retry continuation `M12-07`
- `Planned` UI/UX production polish closure (post-functional):
  - finalize modern glass antracite visual system
  - full section-level readability pass (no floating status text)
  - accessibility/contrast consistency checks across light/dark modes `M12-08`

### M13 - Multi-Target Export Ecosystem (Post-Ollama)
- `Planned` Export adapter architecture (`IExportAdapter`-style contract) with target capability checks and deterministic error taxonomy. `M13-01`
- `Planned` Native Hugging Face export target (checkpoint packaging + tokenizer/config validation + manifest wiring). `M13-02`
- `Planned` ONNX Runtime production target (graph export profile presets + runtime validation bundle). `M13-03`
- `Planned` TensorRT target foundation (GPU capability guardrails + conversion preflight + unsupported-device hard blocks). `M13-04`
- `Planned` Unified export matrix UI (target selector, readiness state, per-target requirements, and step-by-step guidance). `M13-05`
- `Planned` Per-target compatibility registry (tool versions, supported model families, supported input formats). `M13-06`
- `Planned` End-to-end export certification suite across targets (artifact validator + smoke inference + reproducibility snapshot). `M13-07`

## Sequential Delivery Mode (User-Gated)
- Work is delivered in small sequential blocks.
- Each block is followed by manual user validation before the next block.
- Roadmap/checklist status is updated only after user confirmation on real runtime behavior.

## Verification Notes (2026-05-16)
- `Completed Verified`: `dotnet build LLMForgeStudio.sln` passed (0 errors).
- `Completed Verified`: Python runner checks passed for new fine-tuning backend scripts.
- `Completed Verified`: Gather Dataset UI now exposes ordered numbered actions with gating (`1..7`) and async progress bar updates.
- `Completed Verified`: Gather advanced merge UI/build path added (source toggles, weights, dedup policy, provenance artifact, compliance block) and compiles cleanly.
- `Completed Verified`: M11 foundation code compiles (`dotnet build ... -o /tmp/llmforge-m11-build`, 0 errors) and Python syntax checks pass for updated runtime scripts.
- `Completed Verified`: E2E gate suite runner now supports scenario packs with aggregate summary artifacts (`e2e_suite_summary.json/.md`) and per-scenario report folders (`M11-05` foundation).
- Runtime behaviors beyond local checkpoints (especially GGUF converter toolchain runtime on real checkpoints and long-run fine-tune stress scenarios) remain to be validated in M11 manual tests.
