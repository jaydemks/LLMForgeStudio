# LLMForgeStudio - Manual QC Protocol

This document defines a repeatable end-to-end manual quality check for:

- Dataset import
- Tokenization
- Model/training pipeline
- Checkpoint integrity
- Save/load persistence
- Text generation

It includes two full paths:

1. Single file import flow
2. Folder import flow

Use this protocol before public releases.

## 1. Test Data

Use the built-in repository samples:

- `samples/qc_tiny_poetry.txt`
- `samples/qc_medium_tech.txt`
- `samples/qc_noisy_text.txt`
- `samples/qc_import_folder/part1_intro.txt`
- `samples/qc_import_folder/part2_training.txt`
- `samples/qc_import_folder/part3_generation.txt`

## 2. Execution Rules

- Run tests through the application UI only.
- Record outcome for each test case (`PASS` / `FAIL`).
- If a test fails, capture exact UI error text and the action that triggered it.
- Do not skip tokenizers: test all available options.

## 3. Path A - Single File Import QC

### A1. Dataset Import - Tiny Corpus

Steps:

1. Open app.
2. Click `Import Dataset`.
3. Select `samples/qc_tiny_poetry.txt`.

Expected:

- Import completes without errors.
- Dataset preview is populated.
- Dataset stats update from previous state.

### A2. Dataset Import - Noisy Corpus

Steps:

1. Click `Import Dataset`.
2. Select `samples/qc_noisy_text.txt`.

Expected:

- Import handles noisy characters/symbols.
- No UI freeze/crash.
- Preview/stats remain coherent.

### A3. Tokenizer Matrix (Required)

Dataset under test: `samples/qc_medium_tech.txt`

Steps:

1. Import `samples/qc_medium_tech.txt`.
2. Run tokenization with each tokenizer:
- Character
- Word
- Simple BPE
- Hybrid Fallback

Expected:

- All tokenizers produce non-empty output.
- No tokenizer throws runtime errors.
- Token count changes by strategy:
- Character usually highest token count.
- Word usually lower than character.
- BPE typically between character and word (implementation-dependent).
- Hybrid remains robust on mixed text.

### A4. Training Sanity Run

Steps:

1. Configure training in UI with a short run.
2. Set a clean run directory (new folder for this run).
3. Start training.

Expected:

- Training starts and emits live progress logs.
- Loss values are numeric (not NaN/Inf).
- Run reaches completion without crash.

### A5. Cancel Behavior

Steps:

1. Start another training run.
2. Click `Cancel` while training is active.

Expected:

- Cancellation is accepted.
- Process stops cleanly.
- UI remains responsive.

### A6. Checkpoint Validation

Steps:

1. Open run output from UI.
2. Confirm checkpoint artifacts were produced.

Expected:

- Checkpoint manifest exists.
- Model weights file exists.
- No corrupted/empty checkpoint artifact.

### A7. Save/Load Project

Steps:

1. Save project to `.llmforge.json`.
2. Close and reopen the app.
3. Load saved project.

Expected:

- Paths and core settings are restored.
- Dataset and training config remain consistent.
- No load-time exceptions.

### A8. Generation Final Check

Use checkpoint from the successful run.

Prompts:

- `A tokenizer splits`
- `Training quality depends`
- `The sun is`

Expected:

- Generation output is non-empty for all prompts.
- Output style reflects training corpus domain.
- No generation-time backend/UI errors.

## 4. Path B - Folder Import QC

### B1. Folder Import

Steps:

1. Open app.
2. Click `Import Folder`.
3. Select `samples/qc_import_folder`.

Expected:

- All files in folder are ingested.
- Combined preview contains content from all three parts.
- Stats reflect larger merged corpus.

### B2. Tokenizer Matrix on Merged Corpus

Repeat tokenizer checks from A3 on folder-imported dataset.

Expected:

- Same stability requirements as A3.
- No regression versus single-file flow.

### B3. Training and Cancel on Merged Corpus

Repeat A4 and A5 on folder-imported dataset.

Expected:

- Training works on merged dataset.
- Cancel works identically.

### B4. Checkpoint and Generation on Merged Corpus

Repeat A6 and A8 using checkpoint from folder-based training.

Prompt set:

- `Machine learning experiments`
- `Training quality depends`
- `Generated text should`

Expected:

- Non-empty outputs.
- Domain-consistent continuations.
- No runtime errors.

## 5. GO / NO-GO Criteria

Release is `GO` only if all conditions below are true:

1. Single-file import path passes all tests.
2. Folder import path passes all tests.
3. All tokenizers are functional in both paths.
4. Training and cancel behavior are stable.
5. Checkpoint creation is valid.
6. Save/load persistence works.
7. Generation is non-empty and stable.

Release is `NO-GO` if any critical condition fails.

## 6. QC Recording Template

Copy this table section in your test notes and fill it during execution.

| Test ID | Area | Input | Expected | Actual | Result | Notes |
|---|---|---|---|---|---|---|
| A1 | Single File Import | qc_tiny_poetry.txt | Import+preview+stats OK |  |  |  |
| A2 | Single File Import | qc_noisy_text.txt | No crash, coherent output |  |  |  |
| A3-C | Tokenizer | Character | Non-empty tokens, stable |  |  |  |
| A3-W | Tokenizer | Word | Non-empty tokens, stable |  |  |  |
| A3-B | Tokenizer | Simple BPE | Non-empty tokens, stable |  |  |  |
| A3-H | Tokenizer | Hybrid Fallback | Non-empty tokens, stable |  |  |  |
| A4 | Training | medium_tech single file | Run completes, numeric loss |  |  |  |
| A5 | Training Cancel | medium_tech single file | Clean cancel, responsive UI |  |  |  |
| A6 | Checkpoint | single file run | Manifest + weights exist |  |  |  |
| A7 | Save/Load | .llmforge.json | Config restored correctly |  |  |  |
| A8 | Generation | 3 prompts | Non-empty, coherent output |  |  |  |
| B1 | Folder Import | qc_import_folder | Merged content imported |  |  |  |
| B2-C | Tokenizer | Character | Stable on merged corpus |  |  |  |
| B2-W | Tokenizer | Word | Stable on merged corpus |  |  |  |
| B2-B | Tokenizer | Simple BPE | Stable on merged corpus |  |  |  |
| B2-H | Tokenizer | Hybrid Fallback | Stable on merged corpus |  |  |  |
| B3 | Training + Cancel | merged corpus | Stable run + cancel |  |  |  |
| B4 | Generation | 3 folder prompts | Non-empty, coherent output |  |  |  |

