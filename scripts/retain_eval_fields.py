from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


KEEP_FIELDS = ("id", "question", "cutoff", "binary_axis", "ground_truth")
LABEL_FIELDS = ("ground_truth", "answer", "gold_event")
LEGACY_AXIS = {
    "schema_version": "chain-binary-axis-v1",
    "positive": {
        "id": "yes",
        "label": "Yes",
        "semantics": "event occurs",
        "aliases": ["yes", "true", "1"],
    },
    "negative": {
        "id": "no",
        "label": "No",
        "semantics": "event does not occur",
        "aliases": ["no", "false", "0"],
    },
    "source": "explicit",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def minimal_row(
    row: dict[str, Any],
    path: Path,
    index: int,
    use_legacy_axis: bool,
    fallback_cutoff: str,
) -> dict[str, Any]:
    case = f"{path.name}:{index}"
    case_id = row.get("id")
    question = row.get("question")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{case} requires a non-empty string id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{case} requires a non-empty string question")

    cutoff = row.get("cutoff")
    if cutoff in (None, ""):
        cutoff = row.get("date")
    if cutoff in (None, ""):
        cutoff = fallback_cutoff
    if not isinstance(cutoff, str) or not cutoff.strip():
        raise ValueError(f"{case} requires cutoff or a non-empty date")

    axis = row.get("binary_axis")
    if axis is None and use_legacy_axis:
        axis = LEGACY_AXIS
    if not isinstance(axis, dict):
        raise ValueError(f"{case} requires binary_axis; rerun with --legacy-yes-no-axis for legacy yes/no files")

    label_name = next((name for name in LABEL_FIELDS if name in row), None)
    if label_name is None:
        raise ValueError(f"{case} requires one of {', '.join(LABEL_FIELDS)}")

    return {
        "id": case_id,
        "question": question,
        "cutoff": cutoff,
        "binary_axis": axis,
        "ground_truth": row[label_name],
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/eval-minimal"))
    parser.add_argument("--legacy-yes-no-axis", action="store_true")
    parser.add_argument("--fallback-cutoff", default="")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    output_dir = args.input_dir if args.in_place else args.output_dir
    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no JSONL files found in {args.input_dir}")
    for source in files:
        rows = load_rows(source)
        reduced = [
            minimal_row(
                row,
                source,
                index,
                args.legacy_yes_no_axis,
                args.fallback_cutoff,
            )
            for index, row in enumerate(rows, 1)
        ]
        target = output_dir / source.name
        write_rows(target, reduced)
        print(f"{source} -> {target}: {len(reduced)} rows, fields={KEEP_FIELDS}")


if __name__ == "__main__":
    main()
