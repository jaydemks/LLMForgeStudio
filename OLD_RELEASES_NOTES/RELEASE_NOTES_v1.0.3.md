# LLM Forge Studio - Release Notes v1.0.3
Date: 2026-05-16

## Summary
Major stabilization release: deep runtime hardening for Gather/Tokenizer/Training, project-state persistence improvements, and end-to-end reliability fixes.

## Added
- Runtime logging overlay in top-right header (floating console with clear/close).
- Notification center in top-right header (`🔔`) with overlay history for important process events (success/warning/error) and timestamps.
- Cluster live observability foundation:
  - coordinator/worker heartbeat visibility
  - queue counters (`pending/claimed/result`)
  - remote GPU telemetry surface (when reported by workers)
- Gather staged-source row-level `Remove` action.
- Gather busy-operation cancellation and live transfer telemetry (size, elapsed, MB/s).
- Gather handoff loading overlay (EN/IT) with fade transitions.
- Training watchdog/heartbeat hardening:
  - live no-progress heartbeat during long finalization phases
  - explicit status showing idle duration + last logged step while backend is still running
  - hard watchdog timeout on prolonged no-progress windows with deterministic process termination (prevents indefinite stuck runs)
  - cancel now also interrupts monitor token path, not only process kill

## Changed
- Fine-Tuning section is now explicitly labeled `Experimental` in UI/title to reflect current validation status.
- Top-right header actions are now icon-only and reordered for faster workflow:
  - `🔔 Notifications` -> `🪵 Logs` -> `📂 Load Project` -> `💾 Save Project` -> `📘 Guide` -> `⚙ Settings`
- Model/Generation clarity improved with explicit token-window semantics:
  - `Context Window` is now shown in Model section as input-token window (mapped to `Block Size`)
  - `Max Output Tokens` is now shown in Generation section (mapped to `Max New Tokens`)
  - dynamic EN/IT hints now show current effective values directly in UI
- Default training stability retune for from-scratch demo runs:
  - ByteLevelBPE recommended model now uses a safer default capacity (`block=192`, `layers=6`, `heads=6`, `emb=384`)
  - Serious profile now defaults to a lower-noise optimizer setup (`AdamW`, lower LR, higher warmup, grad accumulation=2, eval cadence tightened)
  - goal: reduce gibberish/template collapse risk on small/medium chat corpora while keeping the existing guided workflow unchanged
- Tokenizer flow is now fully async/non-blocking with inline progress, ETA, live throughput stats, and cancel action.
- `Build x/y Preview` is explicitly optional and no longer blocks training readiness.
- Generation seed now supports natural/random mode:
  - `Seed = -1` auto-randomizes each run
  - fixed numeric seed remains available for reproducibility
- Generation defaults tuned for user-friendly stability on fresh projects:
  - `temperature=0.45`
  - `top_k=30`
  - `max_new_tokens=80`
- Tokenizer finalization UX now explicitly enters `Finalizing...` phase when ETA is no longer reliable.
- Tokenizer decoded preview is bounded/truncated for UI safety on large datasets.
- Dataset section now uses large-dataset safe preview mode (full data remains on disk, preview is truncated).
- Dataset loader now normalizes structured files before tokenization/training:
  - `.jsonl/.json/.csv` are parsed into training text (chat pairs/messages/text)
  - metadata-only JSON payloads are ignored instead of polluting corpus quality
- Dataset path display hardened for long paths (read-only horizontal scrolling + stable layout).
- KPI/stats refresh in Dataset mode improved (chars/lines/words/unique chars refresh reliably).
- Guided defaults retuned for quality:
  - stronger BPE vocab floors
  - adaptive BPE min-frequency on smaller corpora
  - `Serious` profile baseline depth increased (`>=3000` max steps)
  - automatic DPO disabled in from-scratch Serious baseline flow

## Fixed
- Tokenization/training pipeline alignment fix (critical):
  - training backend now consumes UI tokenization state (`tokenization_state.json`) when available, instead of always forcing char-level fallback
  - checkpoints now preserve tokenizer type metadata (`ui-tokenizer-state` vs `char`) in `tokenizer.json`
  - generation backend now supports non-char tokenizer vocab encoding via greedy longest-match fallback, improving compatibility with BPE-like token vocabularies
  - strict-mode enforcement: backend training now fails fast when tokenizer state is missing/invalid (`TOKENIZER_STATE_REQUIRED`) instead of silently falling back to char-level
- Validation sample dataset quality fix (later replaced by `samples/validation/it_chat_conversation_4k` in v1.0.4):
  - regenerated as `v2_clean_chat_first` for stable from-scratch chat tests
  - removed artificial prompt markers (`id`, `sample`, `n.` patterns)
  - removed embedded English source phrase prompts from the demo corpus
  - kept 10k rows with full prompt/pair uniqueness for cleaner baseline convergence
- Chat behavior vs raw completion fix:
  - generation now auto-wraps plain prompts into chat context (`<|user|> ... <|assistant|>`) and returns assistant-focused output instead of raw full-sequence completion
  - training launch now auto-enables SFT alignment when structured `prompt/response` or `messages` JSONL is detected and alignment was `none`
  - prevents JSON-style continuation outputs on chat-first datasets and improves baseline chatbot behavior before dedicated fine-tuning
  - generation decoder hardening:
    - added `top-p` sampling, repetition penalty, and no-repeat ngram guard
    - added response normalization to remove prompt-echo artifacts
    - strengthened default plain-text prompt shaping (`Istruzione + Domanda + Risposta`) to reduce template-collapse and off-topic continuation
- Training backend stall fix (`~1400/1500` no-error freeze):
  - removed `stdout/stderr` pipe buffering in `cluster_runner.py` training subprocess path
  - prevents child-process deadlock when output buffers fill during long runs
- Cluster heartbeat/state write hardening (Windows file-lock resilience):
  - JSON heartbeat/state writes now use atomic temp-file replace with retry/backoff
  - prevents intermittent training aborts caused by transient concurrent file access on `cluster_heartbeat.json` / `cluster_run_state.json`
- Runtime compatibility cleanup:
  - replaced deprecated `torch.cuda.amp.autocast(...)` usage with `torch.amp.autocast("cuda", ...)`
- Project load/profile sync fix:
  - `SelectedTrainingProfile` is now persisted in project payload
  - on `Load Project`, saved profile is reapplied deterministically (not only displayed), so Advanced Training values and runtime config stay aligned
  - legacy projects without saved profile now infer best matching profile from loaded training config (`Serious/Balanced/Tiny/Research/Cluster`) instead of always defaulting to `Custom`
  - profile/value consistency guard on load:
    - if displayed preset and loaded training fields do not match (e.g. `Serious` with `MaxSteps=1000`), UI now auto-normalizes to `Custom` to avoid misleading state
- Generation checkpoint-path compatibility fix:
  - generation backend now accepts both `checkpoint_manifest.json` and direct `model.pt` paths
  - when `model.pt` is used, tokenizer is auto-resolved from sibling `tokenizer.json`
  - UI now auto-resolves generation checkpoint (`selected path -> run manifest -> run model`) and blocks with explicit message only when no valid checkpoint exists
- `DATASET_EMPTY` false negatives reduced via resolver fallback chain:
  - `external source -> dataset path -> gather merged path`
- Alignment-stage empty-output hardening:
  - if SFT/DPO formatting yields empty output, stage is skipped with fallback to previous dataset content
  - warnings persisted in `fine_tuning_stages.json`
- Training backend safety on short datasets:
  - automatic `block_size` clamp
  - safer split/window bounds
  - deterministic backend error taxonomy (`LLMFORGE_ERROR`)
  - actionable UI hints for error codes
- Updated AMP scaler call to modern API (`torch.amp.GradScaler("cuda", ...)`).
- Gather provider fetch filtering hardened to avoid dataset-card/metadata pollution.
- Metadata-only JSON extraction now filtered from merged training corpus.
- Parquet flow gating hardened:
  - explicit parquet detection hints
  - merge/validate gated until conversion when needed
  - converted sources correctly replace/disable parquet-origin rows
- License gate regression fixed: changing source resets license approval state.
- Save/Load persistence hardened:
  - gather staged-source state restored
  - tokenizer workflow/artifacts restored (`tokenization_state.json`)
- Training quality gate hardening:
  - blocked training on clearly low-quality/template-collapsed corpora (low uniqueness, excessive repeated template phrases, too-few effective lines)
  - emits explicit user-facing block reason and notification event
  - false-positive reduction pass: line-uniqueness hard-block threshold lowered to critical-only range so valid chat corpora are warned instead of blocked

## Conversion & Export Hardening (Foundation)
- Deterministic GGUF converter resolution order (`env -> bundled -> fallback copy-existing-gguf`).
- Converter compatibility matrix/version probe enforcement.
- Deterministic converter status/error codes + environment snapshot.
- Ollama finalization now validates bundle files before `ready` status.

## Verification Update
- `2026-05-16`: `dotnet build LLMForgeStudio.sln /p:UseAppHost=false` passed (`0 errors`, `0 warnings`).
- `2026-05-17`: same build command passed after latest persistence/fallback fixes.

## Known Limits
- Full native GGUF packaging/distribution closure across all runtime combinations is not complete yet.
- Large-scale multi-machine and long-run recovery matrix still requires broader runtime validation.
- Final accessibility/polish pass is still pending.
