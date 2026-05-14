# Release Validation Checklist (UI Step-by-Step)

This checklist is designed for manual **UI-only** validatedon, in operational order.
Each test maps `VAL-xx` IDs in `ROADMAP.md`.

## Validation Progress
- Total tests: `17`
- Completed `PASS`: `15/17` (`Test 1`, `Test 3`, `Test 5`, `Test 6`, `Test 7`, `Test 8`, `Test 9`, `Test 10`, `Test 11`, `Test 12`, `Test 13`, `Test 14`, `Test 15`, `Test 16`, `Test 17`)
- Pending (`TODO`): `2/17`
- Last update: `2026-05-14`

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

Result: `TODO`

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

Result: `TODO`

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
- GGUF: `model.gguf.placeholder.json`.
- Manifest: `exportTargets` con status/path coerenti.

Result: `PASS` (validated ONNX fallback artifact e GGUF placeholder in separate runs; `exportTargets` consistent per run)

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

## Test Closure
When all tests are completed:
1. Update `ROADMAP.md` by changing `Completed To Be Tested` items to `Completed Verified` where applicable.
2. Freeze the candidate release notes version.

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
