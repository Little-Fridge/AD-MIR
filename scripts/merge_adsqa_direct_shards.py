#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", required=True)
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    rows = []
    for shard in sorted(parent.glob("shard_*")):
        pred_path = shard / "predictions.jsonl"
        if not pred_path.exists():
            continue
        for row in read_jsonl(pred_path):
            row["source_shard"] = shard.name
            rows.append(row)

        src_official = shard / "official_format_results"
        if src_official.exists():
            for qdir in src_official.iterdir():
                pred = qdir / "pred.json"
                if not pred.exists():
                    continue
                dst_dir = parent / "official_format_results" / qdir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pred, dst_dir / "pred.json")

    rows.sort(key=lambda r: (r.get("dataset_index", 10**12), r.get("question_id", "")))
    with open(parent / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_prediction_rows": len(rows),
        "unique_question_ids": len({r.get("question_id") for r in rows if r.get("question_id")}),
        "error_rows": sum(1 for r in rows if r.get("error")),
        "official_pred_dirs": len(list((parent / "official_format_results").glob("*/pred.json")))
        if (parent / "official_format_results").exists()
        else 0,
    }
    with open(parent / "merge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
