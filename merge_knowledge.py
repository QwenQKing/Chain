from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT    = Path(__file__).parent
SRC_DIR = ROOT / "datasets" / "knowledge"
OUT_DIR = ROOT / "datasets" / "knowledge_all"
OUT_FILE = OUT_DIR / "all_knowledge.jsonl"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="强制重新合并,即使输出文件已存在且源未变")
    args = ap.parse_args()

    if not SRC_DIR.exists():
        print(f"✗ 源目录不存在: {SRC_DIR}")
        sys.exit(1)

    jsonl_files = sorted(SRC_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"✗ 未找到 jsonl 文件: {SRC_DIR}/*.jsonl")
        sys.exit(1)

    if not args.force and OUT_FILE.exists():
        out_mtime = OUT_FILE.stat().st_mtime
        newer_sources = [f.name for f in jsonl_files if f.stat().st_mtime > out_mtime]
        if not newer_sources:
            print(f"⏭  [skip] {OUT_FILE} 已存在且源未变,跳过合并")
            print("     需要强制重建请加 --force")
            return
        print(f"⚠  源文件更新: {newer_sources},重新合并")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for old in OUT_DIR.glob("*.jsonl"):
        old.unlink()
        print(f"  [clean] 删除旧文件: {old.name}")

    total_in = 0
    total_out = 0
    dupes = 0
    per_dataset: Counter = Counter()
    seen_ids: set = set()

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for jf in jsonl_files:
            dataset_name = jf.stem
            with open(jf, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    total_in += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rid = rec.get("id")
                    if rid and rid in seen_ids:
                        dupes += 1
                        continue
                    if rid:
                        seen_ids.add(rid)

                    if "source" not in rec:
                        rec["source"] = dataset_name

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_out += 1
                    per_dataset[dataset_name] += 1

    print("=" * 60)
    print(f"Merged {total_in} records → {total_out} unique records")
    print(f"Duplicates removed: {dupes}")
    print("=" * 60)
    print("Per-dataset counts:")
    for name, cnt in sorted(per_dataset.items()):
        print(f"  {name:30s} {cnt}")
    print()
    print(f"Output: {OUT_FILE}")

if __name__ == "__main__":
    main()
