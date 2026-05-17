# LLM Forge Studio - Release Notes v1.0.2
Date: 2026-05-16

## Summary
Completed the first full `Gather Dataset` foundation: acquisition, compliance gating, merge/normalize pipeline, analytics, and direct handoff into training.

## Added
- End-to-end `Gather Dataset` guided flow with ordered steps.
- Advanced merge workspace controls:
  - per-source enable/disable
  - per-source influence weight
  - dedup policy (`none`, `line`, `paragraph`)
- Merge provenance artifact:
  - `<run>/gather_dataset/merged/dataset_merged_provenance.json`
- Compliance gate foundation for merge:
  - blocks merge on restricted/unknown licenses
  - explicit compliance status in UI
- Provider-aware source tracking (provider/license/path metadata).
- Schema harmonization foundation for `.jsonl/.json/.csv/.txt/.md` and common chat/instruction layouts.
- Analytics/readiness foundation with export:
  - `<run>/gather_dataset/dataset_analytics.json`
- One-click recommendation bootstrap (`Apply Recommendations`) with direct jump to `Tokenization`.

## Changed
- Provider detection/handling expanded:
  - Hugging Face, GitHub, GitLab, Zenodo, OpenML, UCI, Data.gov, generic HTTP, local file/folder
- GitHub repository license resolver support added.
- Permissive-license allowlist extended (including `ISC`, `MPL-2.0`).
- Recommendation engine became hardware-aware (Force CPU / no-GPU downgrades, safer defaults).

## Fixed
- Gather source-row layout bug where influence weight could be hidden by long paths.
- Added `Clear Dataset Staging` action to reset staged workflow state.
- Merge gating corrected to allow valid single-source merge.
- Performance panel `DISK` metric changed from occupancy to real-time I/O activity.

## UI Refresh (v1.0.2 pass)
- Initial anthracite/glass refresh for shell/cards/controls.
- Hardware section split into clearer visual sub-panels.

## Validation
- `dotnet build LLMForgeStudio.sln` passed (`0 errors`).
- Gather validation assets/checklist expanded (`test23`..`test33`).

## Known Limits
- Mixed-license compatibility matrix is foundation-level and needs broader runtime validation.
- Multi-provider coverage is expanded but still not exhaustive.
- Analytics dashboard is artifact/text-first (not full chart UI yet).
