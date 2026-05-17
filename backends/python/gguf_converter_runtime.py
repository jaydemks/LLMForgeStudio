import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import re


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def repo_root_from_script() -> Path:
    # .../<repo>/backends/python/gguf_converter_runtime.py
    return Path(__file__).resolve().parents[2]


def load_compat_matrix() -> dict:
    root = repo_root_from_script()
    matrix_path = root / "backends" / "python" / "tools" / "converter_compatibility_matrix.json"
    if not matrix_path.exists():
        return {"available": False, "path": str(matrix_path)}
    try:
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        data["available"] = True
        data["path"] = str(matrix_path)
        return data
    except Exception as ex:
        return {"available": False, "path": str(matrix_path), "parseError": str(ex)}


def resolve_converter_path() -> tuple[str, str]:
    env_value = os.environ.get("LLMFORGE_GGUF_CONVERTER", "").strip()
    if env_value:
        p = Path(env_value)
        if p.exists():
            return str(p), "env"
        return "", "env-invalid"

    root = repo_root_from_script()
    candidates = [
        root / "backends" / "python" / "tools" / "gguf_converter.exe",
        root / "backends" / "python" / "tools" / "gguf_converter.bat",
        root / "backends" / "python" / "tools" / "gguf_converter.cmd",
        root / "backends" / "python" / "tools" / "gguf_converter.sh",
    ]
    for c in candidates:
        if c.exists():
            return str(c), "bundled"
    return "", "none"


def parse_semver(s: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", s or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_converter_version(conv_path: Path) -> tuple[bool, str]:
    # Contract: converter wrapper should return semantic version on --version.
    try:
        proc = subprocess.run([str(conv_path), "--version"], capture_output=True, text=True, timeout=15)
        text = (proc.stdout or proc.stderr or "").strip()
        if proc.returncode != 0 or not text:
            return False, text or f"exit={proc.returncode}"
        return True, text
    except Exception as ex:
        return False, str(ex)


def is_converter_version_compatible(version_text: str, matrix: dict) -> tuple[bool, str]:
    if not matrix.get("available", False):
        return False, "compatibility matrix missing"
    supported = matrix.get("supportedConverterVersions", [])
    if not isinstance(supported, list) or not supported:
        return False, "supportedConverterVersions not declared"
    conv = parse_semver(version_text)
    if conv is None:
        return False, f"unable to parse converter version: {version_text}"
    conv_s = f"{conv[0]}.{conv[1]}.{conv[2]}"
    if conv_s in supported:
        return True, conv_s
    return False, f"converter version {conv_s} not in supported set {supported}"


def try_copy_existing_gguf(input_dir: Path, output_path: Path) -> bool:
    ggufs = sorted(input_dir.rglob("*.gguf"))
    if not ggufs:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ggufs[0], output_path)
    return output_path.exists()


def detect_input_format(input_dir: Path) -> str:
    # Existing GGUF in tree is always supported via fallback-copy flow.
    if any(input_dir.rglob("*.gguf")):
        return "existing-gguf"
    # HF/llama.cpp conversion flows typically require config + tokenizer + model shards.
    has_config = (input_dir / "config.json").exists()
    has_tokenizer = (input_dir / "tokenizer.json").exists() or (input_dir / "tokenizer.model").exists()
    has_weights = any(input_dir.glob("*.safetensors")) or any(input_dir.glob("pytorch_model*.bin"))
    if has_config and has_tokenizer and has_weights:
        return "hf-transformers"
    # LLM Forge local checkpoint bundle (.pt + tokenizer) is not directly Ollama-convertible.
    has_pt = any(input_dir.glob("*.pt"))
    if has_pt:
        return "llmforge-pt-checkpoint"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    status_path = Path(args.status)
    attempts = max(1, int(args.attempts))

    if not input_dir.exists():
        write_status(status_path, {
            "status": "failed",
            "errorCode": "GGUF_INPUT_DIR_MISSING",
            "message": f"Input dir not found: {input_dir}",
            "generatedAtUtc": utc_now()
        })
        return 2

    converter, source = resolve_converter_path()
    matrix = load_compat_matrix()
    input_format = detect_input_format(input_dir)
    supported_formats = matrix.get("supportedInputFormats", [])
    if not isinstance(supported_formats, list):
        supported_formats = []
    if output_path.exists():
        write_status(status_path, {
            "status": "completed",
            "errorCode": "",
            "message": "GGUF already present. Conversion skipped.",
            "generatedAtUtc": utc_now(),
            "outputPath": str(output_path),
            "converterSource": source,
            "detectedInputFormat": input_format,
            "compatibilityMatrixPath": matrix.get("path", ""),
        })
        return 0

    if not converter:
        # Fallback path: if an existing GGUF is available in input tree, use it.
        if try_copy_existing_gguf(input_dir, output_path):
            write_status(status_path, {
                "status": "completed",
                "errorCode": "",
                "message": "Existing GGUF artifact found and copied to output.",
                "generatedAtUtc": utc_now(),
                "outputPath": str(output_path),
                "converterSource": "fallback-copy-existing",
                "detectedInputFormat": input_format,
                "compatibilityMatrixPath": matrix.get("path", ""),
            })
            return 0

        write_status(status_path, {
            "status": "blocked",
            "errorCode": "GGUF_CONVERTER_NOT_CONFIGURED",
            "message": "No converter configured/found. Configure LLMFORGE_GGUF_CONVERTER or provide bundled converter in backends/python/tools.",
            "generatedAtUtc": utc_now(),
            "compatibilityMatrix": {
                "supportedModes": [
                    "env-converter",
                    "bundled-converter",
                    "fallback-copy-existing-gguf"
                ],
                "notes": "Built-in orchestration active. No compatible converter/bundled binary available."
            },
            "detectedInputFormat": input_format,
            "compatibilityMatrixPath": matrix.get("path", "")
        })
        return 3

    if supported_formats and input_format not in supported_formats:
        write_status(status_path, {
            "status": "blocked",
            "errorCode": "GGUF_INPUT_FORMAT_UNSUPPORTED",
            "message": "Input format is not compatible with this Ollama GGUF converter profile.",
            "generatedAtUtc": utc_now(),
            "detectedInputFormat": input_format,
            "supportedInputFormats": supported_formats,
            "remediation": "Provide an existing GGUF artifact or an HF-compatible model folder for conversion."
        })
        return 8

    conv_path = Path(converter)
    if not conv_path.exists():
        write_status(status_path, {
            "status": "failed",
            "errorCode": "GGUF_CONVERTER_PATH_INVALID",
            "message": f"Converter path not found: {converter}",
            "generatedAtUtc": utc_now()
        })
        return 4

    # Deterministic version gate before conversion attempts.
    ver_ok, ver_text = get_converter_version(conv_path)
    if not ver_ok:
        write_status(status_path, {
            "status": "failed",
            "errorCode": "GGUF_CONVERTER_VERSION_CHECK_FAILED",
            "message": "Converter version check failed (`--version`).",
            "generatedAtUtc": utc_now(),
            "converterPath": str(conv_path),
            "converterSource": source,
            "versionProbeOutput": ver_text[:2000],
            "compatibilityMatrixPath": matrix.get("path", "")
        })
        return 6

    compat_ok, compat_msg = is_converter_version_compatible(ver_text, matrix)
    if not compat_ok:
        write_status(status_path, {
            "status": "blocked",
            "errorCode": "GGUF_CONVERTER_VERSION_INCOMPATIBLE",
            "message": "Converter version is not compatible with supported matrix.",
            "generatedAtUtc": utc_now(),
            "converterPath": str(conv_path),
            "converterSource": source,
            "converterVersionText": ver_text,
            "compatibility": compat_msg,
            "compatibilityMatrixPath": matrix.get("path", ""),
            "remediation": "Use a supported converter version from converter_compatibility_matrix.json or update matrix after validation."
        })
        return 7

    last_err = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            [str(conv_path), str(input_dir), str(output_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and output_path.exists():
            write_status(status_path, {
                "status": "completed",
                "errorCode": "",
                "message": "GGUF conversion completed.",
                "generatedAtUtc": utc_now(),
                "attempt": attempt,
                "outputPath": str(output_path),
                "converterPath": str(conv_path),
                "converterSource": source
            })
            return 0
        last_err = (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip()

    write_status(status_path, {
        "status": "failed",
        "errorCode": "GGUF_CONVERSION_FAILED",
        "message": "GGUF conversion failed after retries.",
        "generatedAtUtc": utc_now(),
        "attempts": attempts,
        "lastError": last_err[:4000]
    })
    return 5


if __name__ == "__main__":
    sys.exit(main())
