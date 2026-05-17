import argparse
import json
from pathlib import Path


def convert_one(src: Path, out_dir: Path) -> dict:
    try:
        import pandas as pd
    except Exception as ex:
        raise RuntimeError("pandas is required for parquet conversion") from ex

    df = pd.read_parquet(src)
    out_path = out_dir / (src.stem + ".jsonl")
    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {}
            for k, v in row.to_dict().items():
                if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                    obj[k] = v
                else:
                    obj[k] = str(v)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows += 1
    return {"input": str(src), "output": str(out_path), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [in_path] if in_path.is_file() else sorted(in_path.rglob("*.parquet"))
    if not files:
        raise SystemExit("No parquet files found.")

    results = []
    for src in files:
        results.append(convert_one(src, out_dir))

    report = out_dir / "parquet_conversion_report.json"
    report.write_text(json.dumps({"files": results}, indent=2), encoding="utf-8")
    print(str(report), flush=True)


if __name__ == "__main__":
    main()
