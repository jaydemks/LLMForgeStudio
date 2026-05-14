# Backend Protocol

The .NET UI writes a JSON job spec and starts Python with:

```bash
python cluster_runner.py --job path/to/job_spec.json
```

`cluster_runner.py` wraps `train_stub.py` and adds:
- retry policy (`ClusterMaxRetries`)
- heartbeat file (`cluster_heartbeat.json`)
- run-state file (`cluster_run_state.json`)
- optional pipeline stage orchestration state (`pipeline_stage_state.json` + `pipeline_stage_<stage>.json`)
- optional shared-filesystem coordinator/worker mode (`ClusterOrchestrator=sharedfs`)

On successful (or artifact-valid) completion, the app writes:
- `artifact_registry.json` with indexed references to cluster/eval/checkpoint artifacts.

## Expected Train Backend Behavior

1. Read `job_spec.json`.
   - Optional cluster metadata fields: `ClusterProfileName`, `ClusterJobDescriptorPath`.
2. Load and optionally clean/deduplicate dataset text.
   - `DatasetPath` can be a plain text file or a `dataset_manifest.json` referencing shard files.
   - Manifest pipeline may also produce `dataset_resume_state.json` for future resume-aware flows.
   - Runtime streaming path can produce `dataset_stream_cache.txt` and update `dataset_resume_state.json` while processing shards.
   - When `AlignmentMode=sft`, the app pre-formats conversational JSONL into chat-style training text before backend launch.
   - When `AlignmentMode=dpo`, the app pre-formats preference JSONL (`prompt/chosen/rejected`) into paired training text before backend launch.
   - Optional fine-tuning orchestration can pre-apply staged transforms (`SFT -> DPO -> RLHF`) and emits `fine_tuning_stages.json` in run folder.
   - RLHF stage supports app-collected inline feedback (`rlhf_feedback_collected.jsonl`) and imported JSONL feedback.
3. Build tokenizer.
4. Build GPT model.
5. Train with selected optimizer/scheduler/precision.
6. Write logs as JSONL:

```json
{"step":0,"train_loss":4.2,"val_loss":4.3,"tokens_per_second":0,"train_perplexity":66.7,"val_perplexity":73.7,"generalization_gap":0.1,"curriculum_fraction":0.25,"message":"started"}
```

7. Save checkpoint manifest:

```json
{
  "formatVersion":"0.2",
  "modelWeightsPath":"model.pt",
  "tokenizerPath":"tokenizer.json",
  "step":1000,
  "trainLoss":1.8,
  "valLoss":2.0,
  "optimizer":"adamw",
  "scheduler":"cosine",
  "alignmentMode":"none",
  "evalSuite":"basic"
}
```

8. Run eval pack and write:

- `eval_summary.json` (machine-readable benchmark summary)
- `eval_scorecard.md` (human-readable scorecard)
- `eval_trend.json` (score trend from training log checkpoints/steps)

Manifest includes optional fields:

- `evalSummaryPath`
- `evalScorecardPath`
- `evalTrendPath`
- `evalRegressionPath`
- `evalReleaseScorecardPath`
- `evalAverageScore`
- `evalBand`
- `evalReleaseGatePassed`
- `evalReleaseGateThreshold`
- `quantization` (optional metadata object: profile, calibration_samples, dtype)
- `quantizationReportPath` and `quantizationReportMarkdownPath` (quality/latency report artifacts)
- `qatReportPath`, `qatFineTuneSteps`, `qatStatus` (QAT foundation artifact metadata)
- `exportTargets` (ONNX/GGUF export status + artifact path)

## Training Options (Current)

Options are read from `Training` section in `job_spec.json`.

- `Optimizer`: `adamw` | `lion` | `adafactor`
- `Scheduler`: `none` | `cosine` | `linear`
- `WarmupSteps`: integer
- `MixedPrecision`: bool
- `Precision`: `fp16` | `bf16`
- `EnableGradientClipping`: bool
- `GradientClipNorm`: float
- `CheckpointEvery`: integer
- `EnablePostTrainingQuantization`: bool (CPU)
- `QuantizationProfile`: `dynamic-int8` | `ptq-int8` | `ptq-int4` (foundation)
- `QuantizationCalibrationSamples`: integer
- `EnableQatPath`, `QatFineTuneSteps` (QAT foundation controls)
- `EnableDeduplication`, `RemoveDuplicateLines`, `RemoveDuplicateParagraphs`
- `NormalizeUnicode`, `CollapseWhitespace`
- `CurriculumLearning`, `CurriculumWarmupRatio`
- `ResumeDatasetFromState`
- `DeterministicShardShuffle`, `DataShuffleSeed`
- `DistributedTraining` (env-driven bootstrap)
- `MultiGpuStrategy`: `none` | `ddp` | `fsdp` (foundation fields)
- `GradientAccumulationSteps`: integer >= 1
- `AutoDeviceMap`: bool (foundation metadata/preflight)
- `OrchestratePipelineStages`: bool
- `PipelineRunDataStage`, `PipelineRunPreprocessStage`, `PipelineRunTrainStage`, `PipelineRunEvalStage`
- `ClusterOrchestrator`: `local` | `sharedfs`
- `AlignmentMode` (manifest annotation)
- `FineTuningOrchestration`, `FineTuneStageSft`, `FineTuneStageDpo`, `FineTuneStageRlhf`, `RlhfFeedbackSource`
- `RlhfFeedbackPath` (optional JSONL import path used by app-side orchestration)
- `RewardModelingEnabled`, `SafetyPolicyMode` (foundation metadata fields)
- `ExportOnnx`, `ExportGguf` (export target toggles)
- `EvalSuite` (manifest annotation)

## SharedFS Cluster Mode (Coordinator + Worker)

When `Training.ClusterOrchestrator = "sharedfs"`, `cluster_runner.py` uses a shared directory queue for multi-machine coordination.

Environment variables:

- `LLMFORGE_CLUSTER_SHARED_ROOT`: absolute path to a folder visible by all nodes.
  - default: `<run_dir>/_cluster_shared`
- `LLMFORGE_CLUSTER_ROLE`: `coordinator` | `worker` | `auto` (default `auto`)
- `LLMFORGE_CLUSTER_NODE_ID`: optional node id label for claimed jobs.

Queue layout under `LLMFORGE_CLUSTER_SHARED_ROOT/queue`:

- `pending/*.json`: queued tickets
- `claimed/*.json`: atomically claimed tickets (`<ticket>__<node>.json`)
- `result/*.json`: completion/failure result per ticket
- `heartbeats/*.json`: coordinator heartbeat/status

Run-folder artifacts in this mode:

- `cluster_queue_ticket.json`
- `cluster_heartbeat.json`
- `cluster_run_state.json`

`auto` mode behavior:

1. creates ticket as coordinator,
2. if a worker claims it, waits for result,
3. if nobody claims within a grace window, falls back to local execution.

## Expected Generation Backend

```bash
python generate_stub.py --checkpoint runs/default/checkpoint.json --prompt "The morning" --temperature 0.8 --top-k 40 --seed 42
```

Output JSON:

```json
{"text":"generated text here"}
```
