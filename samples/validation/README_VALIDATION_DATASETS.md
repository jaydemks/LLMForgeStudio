# Validation Datasets Pack

Questa cartella contiene dataset pronti per i test in `RELEASE_VALIDATION_CHECKLIST.md`.

## Mapping rapido
- Test 1 (VAL-01/02/03):
  - `test01_folder_flat/`
  - `test01_folder_nested/` (contiene sottocartelle)
- Test 5 (VAL-08):
  - SFT: `test05_sft/sft_prompt_response.jsonl`, `test05_sft/sft_messages.jsonl`
  - DPO: `test05_dpo/dpo_preferences.jsonl`
- Test 6 (VAL-09):
  - RLHF import: `test06_rlhf_import/rlhf_feedback_import.jsonl`
- Test 12+ (VAL-15/16/17):
  - Eval corpus esteso: `test12_eval/eval_corpus_long.txt`
- Test generici Import Folder / multi-file:
  - `test_general_mixed/` (.txt/.md/.csv)

## Note pratiche
- I JSONL sono uno-per-riga, UTF-8.
- I file sono piccoli/medi per test veloci locali.
- Per test robusti, puoi duplicare i file o aggiungere tuo testo reale.
