# Release Validation Checklist (UI Step-by-Step)

This checklist is designed for manual **UI-only** validatedon, in operational order.
Each test maps `VAL-xx` IDs in `ROADMAP.md`.

## Validation Progress
- Total tests: `39`
- Completed `PASS`: `21/39` (`Test 1`, `Test 3`, `Test 5`, `Test 6`, `Test 7`, `Test 8`, `Test 9`, `Test 10`, `Test 11`, `Test 12`, `Test 13`, `Test 14`, `Test 15`, `Test 16`, `Test 17`, `Test 22`, `Test 29`, `Test 30`, `Test 31`, `Test 32`, `Test 33`)
- Pending (`TODO`): `18/39`
- Last update: `2026-05-16`

Per-test result values:
- `PASS`
- `FAIL`
- `BLOCKED`

---

## Preparation (one-time)

1. Start the app.
2. Go to `Hardware` and confirm that Python backend is configured (`Backend ready`).
3. Set `Run directory` in a new folder dedicated to this test (e.g. `runs/validatedon_ui`).
4. Keep the run folder open (you will use it to check i file artifact).

---

## Test 1 - Dataset Sharding + Streaming + Resume
**Covers:** `VAL-01`, `VAL-02`, `VAL-03`
Suggested dataset: `samples/validatedon/test01_folder_flat/` (or `samples/validatedon/test01_folder_nested/`)

### UI Steps
1. Go to `Dataset`.
2. Click `Import Folder` with a multi-file dataset (txt/md/csv).
3. In `Training` -> `Advanced Training` enable:
   - dedup line/paragraph
   - normalize unicode
   - collapse whitespace
   - curriculum learning
4. Start `Start Backend Training`.
5. Stop training after a few steps (`Cancel Training`).
6. Restart training with the same run directory.

### Expected verifiedon (files)
- These files exist: `dataset_stream_cache.txt`, `dataset_resume_state.json`.
- On restart, files are updated (timestamp/contenuto change).
- In `train_log.jsonl` appears `curriculum_fraction` and progresses over time.

Result: `PASS` (verified from `ui_debug_log.jsonl` + artifact run)

---

## Test 2 - Distributed bootstrap + Multi-GPU foundation
**Covers:** `VAL-04`, `VAL-05`

### UI Steps
1. In `Training` -> `Advanced Training`:
   - `Distributed mode` ON
   - `Multi-GPU strategy` = `ddp` (then repeat with `fsdp`)
   - set `Grad accum steps` > 1
2. select `Cluster profile` consistent (es. `workstation-multigpu` o `cluster-standard`).
3. Check the `Training Preflight`: should not block if env is valid.
4. Start training.

### Expected verifiedon (UI + files)
- In UI non appears blocco preflight invalid.
- In manifest (`checkpoint_manifest.json`) appear:
  - `distributedTraining: true`
  - `multiGpuStrategy`
  - `gradientAccumulationSteps`

Result: `TODO (implemented, manual UI/runtime test required)`

---

## Test 3 - Pipeline stages orchestrate
**Covers:** `VAL-06`

### UI Steps
1. In `Training` -> `Advanced Training`:
   - `Orchestrate pipeline stages` ON
   - leave ON: `Stage data/preprocess/train/eval`
2. Start training.

### Expected verifiedon (files)
- Esiste `pipeline_stage_state.json`.
- Stage files exist such as `pipeline_stage_data.json`, `pipeline_stage_train.json`, ecc.

Result: `PASS` (verified on `VAL_06_SET_TEST_3`)

---

## Test 4 - Cluster run-state / retry / heartbeat
**Covers:** `VAL-07`

### UI Steps
1. In `Training` use a cluster profile (o set retry > 0 via profile).
2. Start training e let it run.

### Expected verifiedon (files)
- Esiste `cluster_run_state.json` con stato consistent (`running/completed/...`).
- Esiste `cluster_heartbeat.json` updated during run.
- In case of simulated error, verify retry transitions (`retrying`).

### Multi-machine variant (SharedFS, optional but recommended)
1. Set `Cluster orchestrator = sharedfs`.
2. On all machines use the same shared folder (`LLMFORGE_CLUSTER_SHARED_ROOT`).
3. Start one node with `LLMFORGE_CLUSTER_ROLE=coordinator` (from UI/training start).
4. Start uno o plus nodes con `LLMFORGE_CLUSTER_ROLE=worker` (stesso software/backend).
5. Verify in shared-folder queue `pending/claimed/result/heartbeats`.
6. Verify in run dir: `cluster_queue_ticket.json`, `cluster_run_state.json`, `cluster_heartbeat.json`.

Result: `TODO (implemented, manual conversion test required)`

---

## Test 5 - SFT e DPO formatting
**Covers:** `VAL-08`
Suggested datasets:
- SFT: `samples/validatedon/test05_sft/sft_prompt_response.jsonl` (alternativa: `sft_messages.jsonl`)
- DPO: `samples/validatedon/test05_dpo/dpo_preferences.jsonl`

### UI Steps
1. In `Dataset`, load JSONL SFT (`prompt/response` o `messages`).
2. In `Training` enable fine-tuning orchestration e `SFT` ON.
3. Start short training.
4. Repeat with dataset DPO (`prompt/chosen/rejected`) e `DPO` ON.

### Expected verifiedon (files)
- `dataset_consolidated.txt` non vuoto e consistent con conversione attesa.
- `fine_tuning_stages.json` con stage correctly completed.

Result: `PASS` (DPO + SFT verified con `fine_tuning_stages.json` e `dataset_consolidated.txt`)

---

## Test 6 - RLHF import external
**Covers:** `VAL-09`
Suggested dataset: `samples/validatedon/test06_rlhf_import/rlhf_feedback_import.jsonl`

### UI Steps
1. In `Training`:
   - `RLHF` ON
   - `RLHF feedback source` = `external-import` (o `jsonl-human-feedback`)
   - set `RLHF feedback path` a JSONL valid
2. Start training.

### Expected verifiedon (files)
- Esiste `rlhf_feedback_import.json`.
- `imported_records` > 0.

Result: `PASS` (verified `rlhf_feedback_import.json` con `imported_records > 0` + stage RLHF completed)

---

## Test 7 - RLHF inline collection from UI
**Covers:** `VAL-10`

### UI Steps
1. In `Training`:
   - `RLHF` ON
   - `RLHF feedback source` = `inline`
2. Fill fields:
   - `RLHF prompt`
   - `Preferred answer`
   - `Rejected answer`
3. Click `Add RLHF feedback` (plus record).
4. Start training.

### Expected verifiedon (UI + files)
- `Collected RLHF records` increases in UI.
- Esiste `rlhf_feedback_collected.jsonl` in run directory.
- Se no record: preflight blocks (correct behavior).

Result: `PASS` (verified `rlhf_feedback_collected.jsonl` + stage RLHF completed)

---

## Test 8 - Reward/Safety hooks
**Covers:** `VAL-11`

### UI Steps
1. In `Training`:
   - toggle `Reward modeling` ON
   - change `Safety policy` (standard/strict/research)
2. Start training.

### Expected verifiedon (manifest)
- In `checkpoint_manifest.json` present campi reward/safety coerenti.

Result: `PASS` (manifest con `rewardModelingEnabled` e `safetyPolicyMode` coerenti)

---

## Test 9 - Quantization INT8/INT4 + report
**Covers:** `VAL-12`

### UI Steps
1. In `Training`:
   - `Post quantization` ON
   - `Quant profile` = `ptq-int8` (poi ripeti `ptq-int4`)
   - `Quant calib samples` populated
   - if needed, set `Force CPU` ON
2. Start training.

### Expected verifiedon (files)
- Esiste modello quantizzato.
- Esistono `quantization_report.json` e `quantization_report.md`.
- Manifest contains metadata quantizzazione + path report.

Result: `PASS` (verified profiles `ptq-int8` e `ptq-int4`, quantization reports e manifest metadata)

---

## Test 10 - QAT foundation
**Covers:** `VAL-13`

### UI Steps
1. In `Training` enable `Enable QAT path`.
2. Set `QAT fine-tune steps`.
3. Start training.

### Expected verifiedon (files)
- Esiste `qat_report.json`.
- Manifest contains `qatReportPath`, `qatFineTuneSteps`, `qatStatus`.

Result: `PASS` (verified `qat_report.json` + `qatReportPath/qatFineTuneSteps/qatStatus` nel manifest)

---

## Test 11 - Export ONNX / GGUF
**Covers:** `VAL-14`

### UI Steps
1. In `Training` enable `Export ONNX` e/o `Export GGUF`.
2. Start training.

### Expected verifiedon (files)
- ONNX: `model.onnx` or `model.onnx.export_error.txt` (fallback gestito).
- GGUF: `model.gguf` (if converter path available) or `gguf_conversion_status.json` with deterministic blocked/failed reason.
- Manifest: `exportTargets` con status/path coerenti.

Result: `TODO (GGUF export path moved from placeholder artifact to runtime conversion status flow; re-test required)`

---

## Test 12 - Eval harness full-20
**Covers:** `VAL-15`
Suggested dataset: `samples/validatedon/test12_eval/eval_corpus_long.txt`

### UI Steps
1. In `Training` scegli `Eval suite = full-20`.
2. Start training.

### Expected verifiedon (files)
- `eval_summary.json` con `num_benchmarks = 20`.
- `eval_scorecard.md` populated.

Result: `PASS` (verified `eval_summary.json` con `num_benchmarks=20` + `eval_scorecard.md`)

---

## Test 13 - Eval regression cross-run
**Covers:** `VAL-16`

### UI Steps
1. Esegui almeno 2 run nella stessa run directory (o con history preservata).
2. Mantieni stesso eval suite.

### Expected verifiedon (files)
- `eval_regression.json` con `history` (>=2 entry).
- `last_delta_vs_previous` populated.

Result: `PASS` (verified `eval_regression.json` con `history >= 2` e `last_delta_vs_previous`)

---

## Test 14 - Release candidate scorecard gate
**Covers:** `VAL-17`

### UI Steps
1. Esegui training con eval attivo.

### Expected verifiedon (files)
- Esiste `release_candidate_scorecard.md`.
- Contiene verdict `PASS` o `FAIL` + weak tasks.

Result: `PASS` (verified `release_candidate_scorecard.md` con verdict + weakest tasks)

---

## Test 15 - Guided wizards in Training
**Covers:** `VAL-18`

### UI Steps
1. In `Training`, controlla riquadro wizard dettagliato.
2. Verify state progression across the 4 steps:
   - Multi-GPU
   - Cluster profile
   - RLHF import
   - Eval pack
3. Cambia opzioni e verifica aggiornamento immediato del wizard.

### Expected verifiedon
- The guide text reflects the real configuration without inconsistencies.

Result: `PASS` (wizard/guidance consistent with real configuration during executed tests)

---

## Test 16 - End-to-End complete
**Covers:** `VAL-19`

### UI Steps
1. Workflow complete:
   - Dataset -> Tokenization -> Model -> Training -> Generation
2. Start generation da checkpoint final.

### Expected verifiedon
- The full pipeline works without unexpected blocking issues.
- Output generation ottenuto.

Result: `PASS` (workflow end-to-end completed plus volte: Dataset -> Tokenization -> Model -> Training -> Generation)

---

## Test 17 - Artifact registry final
**Covers:** `VAL-20`

### UI Steps
1. Dopo run complete, open output folder.
2. Verify `artifact_registry.json`.

### Expected verifiedon
- Registry includes references to:
  - checkpoint
  - cluster state/heartbeat
  - eval summary/scorecard/trend/regression/release scorecard
  - quantization/QAT/export artifacts if enabled

Result: `PASS` (verified `artifact_registry.json` with references cluster/eval/checkpoint e artifact disponibili)

---

## Test 18 - Ollama Export Button Gating
**Covers:** `VAL-22`

### UI Steps
1. In `Training`, keep `Export GGUF` OFF and finish a short run.
2. Verify `Export for Ollama` button stays disabled.
3. Enable `Export GGUF` and run training without real GGUF artifact.
4. Verify button remains disabled and hint explains missing GGUF.
5. Provide run with real GGUF artifact and verify button becomes enabled.

### Expected validation (UI)
- Button enablement follows prerequisites exactly.
- Hint text is coherent with current state.

Result: `TODO (implemented, manual quality-check validation required)`

---

## Test 19 - Manual Ollama Bundle Export
**Covers:** `VAL-21`

### UI Steps
1. Use a run where GGUF is available.
2. Click `Export for Ollama`.
3. Open run output folder.

### Expected validation (files)
- `exports/ollama/model.gguf`
- `exports/ollama/Modelfile`
- `exports/ollama/README_OLLAMA.txt`
- `exports/ollama/ollama_export_status.json` with status `ready`

Result: `TODO (implemented, manual recommendation accuracy test required)`

---

## Test 20 - Manual Handoff Integrity (No Internal Ollama Writes)
**Covers:** `VAL-21`

### UI Steps
1. Run `Export for Ollama`.
2. Inspect generated files under run directory.
3. Check that app did not write to local Ollama internal storage paths.

### Expected validation
- Export is confined to run folder (`exports/ollama`).
- No automatic write to Ollama `blobs/manifests`.

Result: `TODO (implemented, manual handoff flow validation required)`

---

## Test 21 - Fine-Tuning (Ollama) Dedicated Workspace
**Covers:** `VAL-23`, `VAL-24`

### UI Steps
1. Open the dedicated `Fine-Tuning (Ollama)` tab (once implemented).
2. Verify separate controls from from-scratch workflow.
3. Execute one end-to-end fine-tune run in the new workspace.

### Expected validation
- No workflow overlap/confusion with existing from-scratch tab.
- Dedicated run artifacts and logs are produced.
- Export folder for fine-tune path is generated (`exports/ollama_finetune` when implemented).

Result: `TODO`

---

## Test 22 - Gather Dataset UI and Source Acquisition
**Covers:** `VAL-25`

### UI Steps
1. Open the new `Gather Dataset` section (when implemented).
2. Add at least one remote source (dataset URL/repository style input).
3. Trigger automated fetch into local staging workspace.

### Expected validation
- Dataset source is downloaded/staged without manual shell steps.
- UI reports source status, destination path, and fetch outcome.
- Fetch is blocked until license verification and user acknowledgement are completed.
- For Hugging Face dataset URLs, restricted or unknown licenses are blocked before fetch.
- Actions are presented in explicit sequence (`1..7`) with gated enablement by step.

Result: `PASS (base Gather flow and UX sequencing validated in-app; no runtime errors reported during guided execution)`

---

## Test 23 - Parquet Auto-Conversion Pipeline
**Covers:** `VAL-26`

### UI Steps
1. In `Gather Dataset`, select a source containing parquet files.
2. Run conversion to LLM Forge supported formats.

### Expected validation (files)
- Converted outputs are generated in compatible formats (`txt/jsonl/json/csv`).
- Conversion report metadata is produced (input files, output files, row counts/errors).

Result: `TODO`

---

## Test 24 - Dataset Quality and Readiness Checks
**Covers:** `VAL-27`

### UI Steps
1. Run dataset validation checks from `Gather Dataset`.
2. Test both a good-quality and low-quality/noisy dataset sample.

### Expected validation
- Immediate UI feedback on:
  - size/coverage
  - duplicates/noise
  - schema/format consistency
  - language/domain hints
- Clear PASS/WARN/BLOCK style readiness outcome.

Result: `TODO`

---

## Test 25 - Smart Recommendations to Training Config
**Covers:** `VAL-28`

### UI Steps
1. Complete dataset gathering/validation.
2. Apply suggested training setup from recommendation panel.

### Expected validation
- Recommended tokenizer preset and training profile are shown.
- Applying recommendations updates real config fields in app.
- Suggestions are traceable/explainable in UI.

Result: `TODO`

---

## Test 26 - Dataset-to-Pipeline Direct Handoff
**Covers:** `VAL-29`

### UI Steps
1. From `Gather Dataset`, execute direct handoff:
   - to from-scratch training flow
   - to fine-tuning flow
2. Verify destination sections receive dataset/config context automatically.

### Expected validation
- No manual copy/paste of dataset paths required.
- Handoff creates a reproducible link between gathered dataset workspace and selected pipeline.

Result: `TODO`

---

## Test 27 - Multi-Provider Source Connectors
**Covers:** `VAL-30`

### UI Steps
1. In `Gather Dataset`, add dataset sources from at least two different providers.
2. Run source checks and staging for each provider.

### Expected validation
- Each provider source is tracked independently in the workspace.
- Connector status and error details are shown per provider.
- Unsupported providers fail gracefully with actionable messages.

Result: `TODO (implemented foundation: provider contract/detection + per-provider compliance hook + unsupported-provider block; pending full multi-provider runtime validation)`

---

## Test 28 - Multi-Source Dataset Composition
**Covers:** `VAL-31`

### UI Steps
1. Select multiple datasets from one or more providers.
2. Build a single merged dataset artifact.
3. Apply source toggles/weights if available.

### Expected validation
- One merged dataset output is generated from selected sources.
- Source provenance (which files/sources contributed) is preserved.
- Merge controls affect resulting dataset composition.

Result: `TODO (implemented baseline merge, manual multi-source runtime validation required)`

---

## Test 29 - Cross-Source Legal Compatibility Gate
**Covers:** `VAL-32`

### UI Steps
1. Attempt to merge sources with compatible permissive licenses.
2. Attempt to merge with one restricted/unknown licensed source.

### Expected validation
- Compatible merge is allowed after acknowledgement.
- Restricted/unknown source blocks the merge/fetch path.
- Compliance snapshot is stored with merge metadata.

Result: `PASS (manual runtime validated: external source fetch is blocked until Check License, MIT allows proceed, non-permissive licenses block flow as expected)`

---

## Test 30 - Schema Harmonization for Mixed Sources
**Covers:** `VAL-33`

### UI Steps
1. Import mixed schemas (e.g., chat JSONL + plain text + csv).
2. Run harmonization to unified training-ready format.

### Expected validation
- Unified output is generated without manual column surgery.
- Mapping rules and dropped/normalized fields are reported.
- Resulting dataset can be consumed by existing training/fine-tuning pipeline.

Result: `PASS (manual/runtime validated on mixed-source folder input; merged output generated with schema harmonization enabled and consumed by Gather flow)`

---

## Test 31 - Advanced Dataset Analytics & Readiness Score
**Covers:** `VAL-34`

### UI Steps
1. Open quality analytics for gathered/merged dataset.
2. Verify duplicate/language/noise/coverage indicators.
3. Compare analytics between clean and noisy sample sets.

### Expected validation
- Analytics metrics update coherently with dataset quality differences.
- Readiness score and action recommendations are shown clearly.

Result: `PASS (validated: dataset_analytics.json generated with readiness, duplicateRatio, nonAsciiRatio, avgLineLength and recommendations)`

---

## Test 32 - One-Click Tokenizer/Training Bootstrap from Gather
**Covers:** `VAL-35`

### UI Steps
1. From `Gather Dataset`, run one-click bootstrap.
2. Verify tokenizer/training recommendations are applied.
3. Use direct jump to Tokenization and inspect updated settings.

### Expected validation
- Suggested tokenizer/training config is applied in real fields.
- Jump to Tokenization section is immediate and coherent.
- Pipeline is ready to proceed without manual re-entry.

Result: `PASS (validated: Apply Recommendations updates config and jumps directly to Tokenization section)`

---

## Test 33 - Advanced Merge/Orchestration Workspace
**Covers:** `VAL-36`

### UI Steps
1. In `Gather Dataset`, stage at least 3 sources.
2. Enable/disable source-level toggles and assign different source weights.
3. Select dedup policy and run merge.
4. Inspect merge provenance output.

### Expected validation
- Merge includes only enabled sources.
- Weighting impacts final contribution ratios.
- Selected dedup policy affects duplicate rate in output.
- Provenance report lists source-level contribution and merge settings.

Result: `PASS (validated manually: 3 sources staged, weights 1/2/3 with source #3 disabled; merge output includes only enabled sources and provenance file reflects weights/toggles correctly)`

---

## Test 34 - M11 Native GGUF Runtime Orchestration
**Covers:** `M11-01`
**Retest note (2026-05-16):** conversion runtime updated (converter resolution order + fallback path). Re-run required.

### UI Steps
1. Complete a fine-tune run in `Fine-Tuning (Ollama)`.
2. Trigger GGUF conversion step.
3. Inspect export folder.

### Expected validation
- `exports/ollama_finetune/gguf_converter_status.json` exists.
- Status contains deterministic state (`completed`/`blocked`/`failed`) and `errorCode`.
- No hard crash when converter is missing or fails.

Result: `TODO`

---

## Test 35 - M11 Auto Finalization Path
**Covers:** `M11-02`
**Retest note (2026-05-16):** finalization flow updated with deterministic validation status. Re-run required.

### UI Steps
1. Enable `Pack output for Ollama handoff`.
2. Run fine-tune with valid converter configured.
3. Verify post-run conversion/finalization.

### Expected validation
- `model.gguf` generated.
- `Modelfile` + `ollama_handoff_status.json` generated automatically.
- Final status is `ready`.

Result: `TODO`

---

## Test 36 - M11 Failure Taxonomy + Retry
**Covers:** `M11-03`
**Retest note (2026-05-16):** converter failure path/messages updated. Re-run required.

### UI Steps
1. Run conversion with invalid/missing converter path.
2. Run conversion with a failing converter wrapper.

### Expected validation
- Failure status includes deterministic `errorCode`.
- Retry attempts are visible in status/log artifacts.
- UI status text is actionable.

Result: `TODO`

---

## Test 37 - M11 Fine-Tune Hardening
**Covers:** `M11-04`

### UI Steps
1. Run fine-tuning with intentionally aggressive batch settings.
2. Stop/restart run to check resume metadata behavior.

### Expected validation
- Runtime writes `ollama_finetune_resume_state.json`.
- Runtime writes `runtime_diagnostics_snapshot.json`.
- Adaptive fallback avoids immediate crash on OOM-prone settings where possible.

Result: `TODO`

---

## Test 38 - M11 Reproducible Runtime Snapshot
**Covers:** `M11-06`
**Retest note (2026-05-16):** runtime snapshot now includes converter resolution metadata. Re-run required.

### UI Steps
1. Run conversion and finalization flow.
2. Inspect exported diagnostics files.

### Expected validation
- `runtime_environment_snapshot.json` exists in export folder.
- Environment + toolchain snapshot fields are populated.

Result: `TODO`

---

## Test 39 - M11 Multi-Backend Foundation
**Covers:** `M11-07`

### UI Steps
1. In fine-tuning UI, select backend target.
2. Prepare and start run.
3. Inspect generated job + manifest.

### Expected validation
- Backend selector is visible and persisted in job payload.
- Manifest includes backend field.
- Current stable backend (`ollama-local`) remains fully functional.

Result: `TODO`

---

## Test 39B - M11 End-to-End Scenario-Pack Suite
**Covers:** `M11-05`
**Retest note (2026-05-16):** suite runner now supports `--scenario-pack` with aggregate output artifacts.

### CLI Steps
1. Prepare one or more completed runs.
2. Create/use scenario pack JSON (example: `samples/validation/e2e_scenarios/quick_suite.json`).
3. Run:
   - `python3 backends/python/e2e_release_gate.py --scenario-pack <pack.json> --output-dir <out-dir>`
4. Inspect artifacts:
   - `e2e_suite_summary.json`
   - `e2e_suite_summary.md`
   - per-scenario subfolder reports.

### Expected validation
- Suite exits deterministically (`0` pass, `2` fail).
- Aggregate summary includes total/pass/fail scenario counters.
- Every scenario has its own check report trail.

Result: `TODO`

---

## Test 40 - M12 Native GGUF Fully Internalized
**Covers:** `M12-01`
**Retest note (2026-05-16):** implemented as foundation with version gate; validate env/bundled/fallback converter paths plus `--version` compatibility blocking.

### UI Steps
1. Run export/fine-tune conversion flow without external converter environment variables.
2. Trigger GGUF conversion from in-app pipeline.
3. Inspect conversion artifacts and logs.
4. Repeat with:
   - valid `LLMFORGE_GGUF_CONVERTER`
   - missing env + bundled converter present
   - missing env + missing bundled converter + existing GGUF in input tree
5. (When available) test an intentionally incompatible converter version.

### Expected validation
- Conversion works with internal managed runtime path only.
- No manual `LLMFORGE_GGUF_CONVERTER` dependency is required in standard path.
- Compatibility/version metadata is recorded in conversion artifacts.
- Deterministic source selection is visible (`env` / `bundled` / `fallback-copy-existing-gguf`).
- Incompatible converter versions are blocked with explicit remediation guidance.
- Missing/invalid `--version` response is blocked with deterministic `GGUF_CONVERTER_VERSION_CHECK_FAILED`.
- Unsupported input formats are blocked with deterministic `GGUF_INPUT_FORMAT_UNSUPPORTED`.

Result: `TODO`

---

## Test 41 - M12 Deterministic Ollama Finalization
**Covers:** `M12-02`
**Retest note (2026-05-16):** implemented as foundation; validate `ollama_bundle_validation.json` + status gating.

### UI Steps
1. Complete fine-tune flow through finalization.
2. Inspect final export folder and handoff status.

### Expected validation
- Finalization reaches deterministic `ready` status.
- Handoff package is complete and self-consistent.
- UI state reflects completion without ambiguous intermediate state.

Result: `TODO`

---

## Test 42 - M12 Hardware/Adapter Classification
**Covers:** `M12-03`
**Retest note (2026-05-16):** GPU detection now filters virtual adapters (Meta/Virtual Desktop/etc.). Re-run required.

### UI Steps
1. Open Hardware section on a machine with physical GPU + virtual adapters.
2. Inspect detected accelerator list and recommendation outputs.

### Expected validation
- Virtual monitor/display adapters are filtered or clearly marked non-training.
- Physical training-capable accelerators remain visible.
- Recommendations respect real device capabilities and VRAM tiers.

Result: `TODO`

---

## Test 43 - M12 Compliance Hard Gate Completion
**Covers:** `M12-04`
**Retest note (2026-05-16):** strict mixed-license matrix + compliance report artifact implemented; re-run required.

### UI Steps
1. Stage multiple dataset sources with mixed licenses.
2. Attempt merge and downstream handoff.

### Expected validation
- Incompatible/restricted combinations are hard-blocked.
- Compatible combinations proceed.
- Compliance report artifact is generated for audit trail.

Result: `TODO`

---

## Test 44 - M12 Full Automated E2E Suite
**Covers:** `M12-05`
**Retest note (2026-05-16):** scenario-pack suite mode added (`--scenario-pack`) with aggregate artifacts (`e2e_suite_summary.json/.md`) plus per-scenario reports.

### UI Steps
1. Execute full pipeline scenario pack:
   - dataset
   - tokenization
   - model
   - training
   - eval
   - export
   - fine-tune
   - convert
   - finalize
2. Inspect generated pass/fail artifacts.

### Expected validation
- Scenario pack completes with deterministic pass/fail output.
- CI-friendly artifacts are generated and reproducible.
- Per-scenario subreports are generated and rolled up in aggregate summary.

Result: `TODO`

---

## Test 45 - M12 Reproducible Runtime Packaging Closure
**Covers:** `M12-06`
**Retest note (2026-05-16):** runtime lock profile + preflight mismatch block artifact implemented. Re-run required.

### UI Steps
1. Run pipeline on baseline environment.
2. Re-run on environment with intentional tool/version mismatch.
3. Inspect runtime lock/snapshot and preflight diagnostics.

### Expected validation
- Runtime/toolchain lock data is generated.
- Mismatch is detected preflight with actionable remediation message.
- Snapshot and lock artifacts are consistent with run metadata.

Result: `TODO`

---

## Test 46 - M12 Long-Run Resume/Recovery Closure
**Covers:** `M12-07`
**Retest note (2026-05-16):** stage checkpoint state + idempotent resume for convert/finalize implemented. Re-run required.

### UI Steps
1. Start long-running training/fine-tune/conversion.
2. Interrupt process (controlled stop/crash simulation).
3. Resume from saved state.

### Expected validation
- Stage checkpoints exist and are crash-safe.
- Resume continues without full restart.
- Final artifacts remain consistent after recovery path.

Result: `TODO`

---

## Test 47 - M12 UI/UX Polish + Accessibility Consistency
**Covers:** `M12-08`
**Retest note (2026-05-16):** global UI styling and section panelization changed; full visual regression pass required.

### UI Steps
1. Review all sections in dark and light theme.
2. Verify readability/contrast on cards, controls, status text, and logs.
3. Validate no floating informational text outside intended containers.

### Expected validation
- Visual system is consistent across sections and themes.
- Contrast/readability is acceptable for prolonged use.
- Layout remains usable and clear at small and large window sizes.

Result: `TODO`

---

## Test 48 - Cluster Live Sync Realtime Panel
**Covers:** `VAL-37`

### UI Steps
1. Configure cluster profile (`cluster-sharedfs` or equivalent) on primary node.
2. Start coordinator run, then start at least one worker node.
3. Open Training section on primary node and inspect `Cluster Live Sync (Realtime)` panel.
4. Verify:
   - role text
   - coordinator/shared-root link text
   - cluster summary counters (`pending/claimed/result`)
   - connected/operational node heartbeat entries
   - remote GPU telemetry entries when worker reports GPUs
5. Open worker node and verify worker heartbeat appears on primary within refresh interval.

### Expected validation
- Primary node shows live cluster heartbeat state and queue counters.
- Worker nodes become visible as operational entries while running.
- Remote GPU telemetry is displayed when reported by workers.
- No UI freeze while panel refreshes.

Result: `TODO`

---

## Test Closure
When all tests are completed:
1. Update `ROADMAP.md` by changing `Completed To Be Tested` items to `Completed Verified` where applicable.
2. Freeze the candidate release notes version.

## Sequential Execution Rule (Current Phase)
1. Implement one roadmap block at a time.
2. Run build/local checks by developer.
3. User performs manual runtime validation for that block.
4. Only then proceed to the next block.

---

## Community Playbook (Multi-GPU / Cluster)

This section is for community testers with different hardware.
Objective: clearly separate local tests, cluster tests, and combined tests.

### A) Multi-GPU only test (single machine, multiple GPUs)
**Covers:** `VAL-04`, `VAL-05` (full multi-GPU verification)

1. Open the app on a machine with at least 2 GPUs.
2. In `Training` -> `Advanced Training`:
   - `Distributed mode` ON
   - `Multi-GPU strategy` = `ddp` (also repeat with `fsdp`)
   - `Grad accum steps` > 1
   - `Cluster orchestrator` = `local`
3. Start training.
4. Verify files:
   - `checkpoint_manifest.json` contains `distributedTraining: true`
   - `multiGpuStrategy` consistent (`ddp` / `fsdp`)
   - `gradientAccumulationSteps` populated
5. Suggested result:
   - `PASS (complete)` if training finishes without unexpected single-device fallback.

### B) Cluster-only test (multiple machines, even 1 GPU per node or CPU)
**Covers:** `VAL-07`

1. Prepare a shared folder accessible from all nodes.
2. Set env su tutti i nodes:
   - `LLMFORGE_CLUSTER_SHARED_ROOT=<shared_folder>`
3. Coordinator node:
   - `LLMFORGE_CLUSTER_ROLE=coordinator`
   - in UI: `Cluster orchestrator = sharedfs`
   - start training.
4. Worker nodes:
   - `LLMFORGE_CLUSTER_ROLE=worker`
   - run backend worker (`cluster_runner.py`) sullo stesso software.
5. Verify in the shared folder:
   - `queue/pending`, `queue/claimed`, `queue/result`, `queue/heartbeats`
6. Verify in run:
   - `cluster_queue_ticket.json`
   - `cluster_run_state.json`
   - `cluster_heartbeat.json`
7. Suggested result:
   - `PASS (cluster complete)` if claim/result/heartbeat are consistent and the job completes.

### C) Combined Multi-GPU + Cluster test (multiple machines, multi-GPU nodes)
**Covers:** `VAL-04`, `VAL-05`, `VAL-07` (most realistic scenario)

1. Usa setup del test `B` (sharedfs coordinator/worker).
2. Additionally, in training configuration:
   - `Distributed mode` ON
   - `Multi-GPU strategy` = `ddp` (poi `fsdp`)
   - `Grad accum steps` > 1
3. Start run coordinata.
4. Combined verification:
   - artifact cluster (`cluster_*`, queue sharedfs) present/coerenti
   - manifest con campi multi-gpu/distributed corretti
   - training finishes without distributed systemic errors
5. Suggested result:
   - `PASS (distributed end-to-end)` if cluster + multi-GPU are both consistent.

### D) Community Result Rules
- If hardware does not support real multi-GPU: mark `PASS partial` (preflight/manifest only).
- If multi-machine environment is missing: mark `BLOCKED` for real cluster section.
- Always attach:
  - `ui_debug_log.jsonl`
  - `cluster_run_state.json`
  - `cluster_heartbeat.json`
  - `checkpoint_manifest.json`
