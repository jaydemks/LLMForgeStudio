# Validation Datasets Pack

Questa cartella contiene dataset pronti per i test in `RELEASE_VALIDATION_CHECKLIST.md`.

## Mapping rapido
- Dataset 10k consigliato per conversazione (training reale):
  - `it_chat_conversation_10k/train.jsonl`
- Dataset 10k per test hardening/quality gate (intenzionalmente debole):
  - `it_template_collapse_hardening_10k/train.jsonl`
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
- Test 23 (VAL-26) - Parquet conversion:
  - `test23_parquet_placeholder/` (metti qui i `.parquet` reali per la conversione)
- Test 28 (VAL-31) - Multi-source composition:
  - `test28_multisource/a_local_chat/`
  - `test28_multisource/b_local_faq/`
- Test 29 (VAL-32) - License compatibility gate:
  - `test29_license_mix/allowed/`
  - `test29_license_mix/blocked/`
- Test 33 (VAL-36) - Advanced merge orchestration:
  - `test33_advanced_merge/src_alpha/`
  - `test33_advanced_merge/src_beta/`
  - `test33_advanced_merge/src_gamma/`
- Test 24 (VAL-27) - Quality/readiness checks:
  - `test24_quality_checks/good/`
  - `test24_quality_checks/noisy/`
- Test 25 (VAL-28) - Recommendation apply:
  - `test25_recommendations_medium/reco_input.txt`
- Test 26 (VAL-29) - Handoff:
  - `test26_handoff/handoff_source.txt`
- Test 27 (VAL-30) - Multi-provider (practical local/http mix):
  - `test27_multiprovider/local_like/`
  - `test27_multiprovider/http_like/`
- Test 30 (VAL-33) - Schema harmonization:
  - `test30_schema_mix/schema_chat.jsonl`
  - `test30_schema_mix/schema_prompt_response.jsonl`
  - `test30_schema_mix/schema_plain.txt`
- Test 31 (VAL-34) - Analytics baseline:
  - `test31_analytics/analytics_input.txt`
- Test 32 (VAL-35) - Bootstrap/apply:
  - `test32_bootstrap/bootstrap_input.txt`

## Note pratiche
- I JSONL sono uno-per-riga, UTF-8.
- I file sono piccoli/medi per test veloci locali.
- Per test robusti, puoi duplicare i file o aggiungere tuo testo reale.
