# GGUF Converter Tools (M12-01)

This folder is the default bundled location for GGUF converter runtime binaries/wrappers.

## Expected files
- `gguf_converter.exe` (Windows) or `gguf_converter.sh` (Linux)
- optional wrappers: `gguf_converter.bat`, `gguf_converter.cmd`
- `converter_compatibility_matrix.json` (required)

## Runtime contract
The converter must support:
1. `--version` -> returns semantic version (e.g. `0.1.0`)
2. positional conversion call:
   - `<converter> <input_dir> <output_gguf_path>`

## Compatibility gate
`backends/python/gguf_converter_runtime.py` blocks conversion when:
- converter version cannot be read
- converter version is not listed in `supportedConverterVersions`
- detected input format is not in `supportedInputFormats`

This guarantees deterministic behavior for release builds.

## Current supported input formats
- `hf-transformers` (folder with HF config/tokenizer/weights)
- `existing-gguf` (reuse/copy existing GGUF artifact)

`llmforge-pt-checkpoint` is intentionally blocked for direct Ollama conversion in this profile.
