# LLM Forge Studio v0.1.1

## Highlights

- Improved end-to-end UX for tokenization, training, and generation.
- Added automatic checkpoint handoff from training to generation.
- Added live training status with clearer lifecycle states.
- Added training quality score (0-100), color band, and tuning hints.
- Added optional `Force CPU training` toggle from the Hardware section.
- Added generated output panel with chatbot-like "Thinking..." flow and streaming text.
- Fixed locale-related generation bug (`0,8` vs `0.8`) for Python arguments.

## UI/UX Improvements

- Training Readiness panel always visible in sidebar.
- Modernized glass-style highlight cards for key runtime info.
- Better training telemetry visibility (`last step`, `train`, `val`, `tok/s avg`, `elapsed`).
- Reduced confusing controls and redundant actions.
- Improved layout containment and reduced oversized spacing.

## Localization

- Fixed mixed-language UI sections.
- Header subtitle and Guide section now follow current language setting consistently.

## Backend and Reliability Fixes

- Robust log polling while backend writes (`FileShare.ReadWrite` behavior).
- Better handling of non-zero exit code runs when valid artifacts are produced.
- Clearer generation/training status messages for success/failure paths.

## Notes

- This release focuses on product polish, reliability, and clearer operator feedback.
- Future expansion is intentionally community-driven.
