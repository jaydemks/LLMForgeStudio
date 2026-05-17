# LLM Forge Studio - Release Notes v1.0.4 (Draft / Future)
Date: Planned

## Summary
Roadmap commitment release for the post-Ollama expansion phase.

> Draft note:
> This file is intentionally forward-looking and is **not** the current shipped release.
> Current active release line is `v1.0.3`.

## Added (Roadmap / Forward-Looking)
- New milestone announced in `ROADMAP.md`:
  - `M13 - Multi-Target Export Ecosystem (Post-Ollama)`

Planned M13 blocks:
1. Export adapter contract with deterministic capability/error checks.
2. Hugging Face export target.
3. ONNX Runtime production export target.
4. TensorRT target foundation with strict hardware preflight.
5. Unified export matrix UI for target readiness and guidance.
6. Per-target compatibility registry (versions/formats/model-family support).
7. Cross-target export certification suite (artifact checks + smoke inference + reproducibility snapshots).

## Clarification
- Current production closure remains focused on Ollama.
- Multi-target conversion/export is roadmap-committed but intentionally deferred.

## Deferred Technical Polish Track
1. Unified timeout/retry/cancel policy across long-running backend operations.
2. Unified error taxonomy across UI + backend scripts.
3. Deterministic cleanup policy for temporary artifacts on cancel/fail.
4. Strict schema validation across generated JSON artifacts/manifests.
5. Cross-platform edge-case hardening (path/permission/process behavior).
6. Accessibility and keyboard-flow closure.

## Minor UX Note
- Gather `Source (URL or Path)` guidance is now explicit for provider URLs and local paths.

## SEO / Discoverability Baseline
- README positioning/headline was tuned for search intent.
- `GITHUB_SEO_PROFILE.md` added with metadata guidance.

## Notes
- This release note is primarily a roadmap communication update, not a claim that M13 exporters are already implemented.
