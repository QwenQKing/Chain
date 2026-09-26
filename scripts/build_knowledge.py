from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(".")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KNO_DIR = ROOT / "datasets" / "knowledge"
TXT_DIR = ROOT / "knowledge_files"
BUILD_RUN_RECEIPT_SCHEMA = "chain-build-run-receipt-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_path(graph_dir: Path) -> Path:


    return graph_dir.parent / f"{graph_dir.name}.build_run_receipt.json"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _empty_usage() -> dict[str, int]:
    return {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _provider_attempt_summary(telemetry: Mapping[str, Any]) -> dict[str, Any]:


    operational = telemetry.get("operational")
    if not isinstance(operational, Mapping):
        raise ValueError("build telemetry has no operational provider traces")
    by_role: dict[str, dict[str, Any]] = {}

    def consume(role: str, trace: Any) -> None:
        if not isinstance(trace, Mapping) or not isinstance(trace.get("attempts"), list):
            return
        row = by_role.setdefault(
            role,
            {
                "api_calls": 0,
                "attempts_with_reported_usage": 0,
                "attempts_without_reported_usage": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "models": set(),
            },
        )
        for attempt in trace["attempts"]:
            if not isinstance(attempt, Mapping):
                continue
            row["api_calls"] += 1
            model = attempt.get("model") or trace.get("model")
            if isinstance(model, str) and model:
                row["models"].add(model)
            usage = attempt.get("usage")
            if not isinstance(usage, Mapping):
                row["attempts_without_reported_usage"] += 1
                continue
            values: dict[str, int] = {}
            valid = True
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                raw = usage.get(key)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    valid = False
                    break
                values[key] = int(raw)
            if not valid:
                row["attempts_without_reported_usage"] += 1
                continue
            row["attempts_with_reported_usage"] += 1
            for key, value in values.items():
                row[key] += value

    extraction_traces = operational.get("extraction_traces", [])
    if isinstance(extraction_traces, list):
        for chunk_trace in extraction_traces:
            if not isinstance(chunk_trace, Mapping):
                continue
            outer_attempts = chunk_trace.get("attempts", [])
            if not isinstance(outer_attempts, list):
                continue
            for attempt in outer_attempts:
                if isinstance(attempt, Mapping):
                    consume("extraction", attempt.get("provider_trace"))
    embedding_traces = operational.get("embedding_traces", [])
    if isinstance(embedding_traces, list):
        for wrapper in embedding_traces:
            if not isinstance(wrapper, Mapping):
                continue
            role = wrapper.get("role") if isinstance(wrapper.get("role"), str) else "embedding"
            consume(role, wrapper.get("trace"))

    totals = {**_empty_usage(), "attempts_with_reported_usage": 0, "attempts_without_reported_usage": 0}
    serializable_roles: dict[str, Any] = {}
    for role, row in sorted(by_role.items()):
        item = {**row, "models": sorted(row["models"])}
        serializable_roles[role] = item
        for key in totals:
            totals[key] += int(item[key])
    return {"by_role": serializable_roles, "totals": totals}


def _success_run_receipt(
    *, graph_dir: Path, result: Mapping[str, Any], started_at_utc: str,
    finished_at_utc: str, elapsed_seconds: float,
) -> dict[str, Any]:
    telemetry_path = graph_dir / "build_telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    if not isinstance(telemetry, Mapping):
        raise ValueError("build_telemetry.json must contain an object")
    operational = telemetry.get("operational")
    declared = operational.get("provider_usage") if isinstance(operational, Mapping) else None
    if not isinstance(declared, Mapping):
        raise ValueError("build telemetry has no provider_usage object")
    fields = ("api_calls", "prompt_tokens", "completion_tokens", "total_tokens")
    declared_usage = {key: int(declared.get(key, 0)) for key in fields}
    attempts = _provider_attempt_summary(telemetry)
    recomputed = {key: int(attempts["totals"][key]) for key in fields}
    reused = bool(result.get("reused", False))
    return {
        "schema_version": BUILD_RUN_RECEIPT_SCHEMA,
        "status": "success",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "graph_dir": str(graph_dir),
        "graph_bundle_sha256": str(result["graph_bundle_sha256"]),
        "build_telemetry_sha256": _sha256_file(telemetry_path),
        "reused_existing_bundle": reused,
        "bundle_counts": {key: int(result[key]) for key in ("chunks", "entities", "hyperedges", "causal_edges", "edges")},
        "bundle_provider_usage": declared_usage,
        "current_run_provider_usage": _empty_usage() if reused else declared_usage,
        "provider_attempts": attempts,
        "usage_closure": {"status": "closed" if recomputed == declared_usage else "mismatch", "declared": declared_usage, "recomputed": recomputed},
        "estimated_cost": {
            "status": "not_configured", "currency": None, "estimated_total": None,
            "reason": "No verified pricing snapshot was supplied; exact token usage is retained without inventing a billing rate.",
        },
    }


def _failure_run_receipt(
    *, graph_dir: Path, started_at_utc: str, finished_at_utc: str,
    elapsed_seconds: float, error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": BUILD_RUN_RECEIPT_SCHEMA,
        "status": "failed",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "graph_dir": str(graph_dir),
        "error": {"type": type(error).__name__, "message": str(error)},
        "provider_usage_status": "unavailable_before_atomic_publication",
        "estimated_cost": {"status": "unavailable", "currency": None, "estimated_total": None, "reason": "The failed staged bundle was removed atomically."},
    }


def _enable_build_progress() -> None:


    build_logger = logging.getLogger("chain.graph.kg_builder")
    if not any(
        getattr(handler, "_chain_build_progress", False)
        for handler in build_logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [KG] %(message)s", datefmt="%H:%M:%S")
        )
        handler._chain_build_progress = True
        build_logger.addHandler(handler)
    build_logger.setLevel(logging.INFO)
    build_logger.propagate = False


def _compat_graph_dir(explicit: Optional[str]) -> Path:
    return Path(
        explicit
        or os.environ.get("HYPERGRAPH_DIR", "")
        or str(ROOT / "datasets" / "KG" / "chain")
    )


def check_chain_kg(kg_dir: Path) -> tuple[bool, str]:


    try:
        from chain.graph.completion import load_graph_bundle_portable

        result = load_graph_bundle_portable(kg_dir)
    except Exception as exc:
        return False, str(exc)
    stats = result["stats"]
    return True, (
        f"chunks={stats['chunks']} entities={stats['entities']} "
        f"hyperedges={stats['hyperedges']} causal_edges={stats['causal_edges']}"
    )


def _write_legacy_transport(source_dir: Path, output_dir: Path) -> dict[str, int]:


    from chain.graph.kg_builder import load_source_records
    from chain.graph.validation import atomic_publish_directory, make_staging_directory

    records, _ = load_source_records(source_dir, allow_compat_adapters=False)
    staging = make_staging_directory(output_dir)
    try:
        by_source: dict[str, list[str]] = {}
        for record in records:
            payload = record["construction_payload"]
            block = []
            if payload["timestamp_raw"]:
                block.append(f"[DATE] {payload['timestamp_raw']}")
            block.extend(
                [
                    f"[SOURCE] {payload['source']}",
                    "[BACKGROUND]",
                    payload["content"],
                ]
            )
            by_source.setdefault(payload["source"], []).append("\n".join(block))
        for source, blocks in sorted(by_source.items()):
            safe_name = source.replace("/", "_").replace("\\", "_")
            (staging / f"{safe_name}.txt").write_text(
                "\n\n---\n\n".join(blocks), encoding="utf-8"
            )
        atomic_publish_directory(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"records": len(records), "files": len(by_source)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate a CHAIN graph bundle")
    parser.add_argument(
        "--knowledge-dir",
        default=str(KNO_DIR),
        help="canonical JSONL knowledge directory (legacy flag retained)",
    )
    parser.add_argument("--no-ingest", action="store_true", help="compat-only text transport")
    parser.add_argument("--force", action="store_true", help="rebuild a valid target")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="read-only structural completion validation (no builder/config/environment identity checks)",
    )
    parser.add_argument("--graph-dir", help="explicit graph target (required for paper)")
    parser.add_argument("--scratch-dir", help="explicit isolated operational scratch path")
    parser.add_argument(
        "--config",
        help=(
            "versioned JSON configuration file for a new build; a complete "
            "existing graph is reused without reading it"
        ),
    )
    return parser


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_only and (args.no_ingest or args.force or args.scratch_dir):
        parser.error("--check-only conflicts with --no-ingest/--force/--scratch-dir")
    if args.no_ingest and args.graph_dir:
        parser.error("--graph-dir is meaningless with --no-ingest")
    return args


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    extractor: Any = None,
    embedder: Any = None,
) -> int:
    args = _parse_args(argv)


    if args.check_only:
        from chain.graph.completion import load_graph_bundle_portable

        graph_dir = Path(args.graph_dir) if args.graph_dir else _compat_graph_dir(None)
        result = load_graph_bundle_portable(graph_dir)
        print(json.dumps(result["stats"], ensure_ascii=False, sort_keys=True))
        return 0

    knowledge_dir = Path(args.knowledge_dir)
    if args.no_ingest:
        if not knowledge_dir.exists():
            raise FileNotFoundError(
                f"knowledge directory does not exist: {knowledge_dir}"
            )
        output_dir = Path(args.scratch_dir) if args.scratch_dir else TXT_DIR
        result = _write_legacy_transport(knowledge_dir, output_dir)
        print(json.dumps({**result, "output": str(output_dir)}, ensure_ascii=False, sort_keys=True))
        return 0

    from chain.graph.kg_builder import build_graph_bundle

    graph_dir = Path(args.graph_dir) if args.graph_dir else _compat_graph_dir(None)
    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    try:
        if graph_dir.exists() and not args.force:




            result = build_graph_bundle(knowledge_dir, graph_dir)
        else:
            if not knowledge_dir.exists():
                raise FileNotFoundError(
                    f"knowledge directory does not exist: {knowledge_dir}"
                )
            from chain.config import resolve_config

            resolved = resolve_config("compat", args.config)
            result = build_graph_bundle(
                knowledge_dir,
                graph_dir,
                resolved_config=resolved,
                profile="compat",
                extractor=extractor,
                embedder=embedder,
                force=args.force,
                allow_compat_adapters=False,
                require_extraction_trace=extractor is None,
                scratch_dir=args.scratch_dir,
            )
    except Exception as exc:
        _atomic_write_json(
            _receipt_path(graph_dir),
            _failure_run_receipt(
                graph_dir=graph_dir,
                started_at_utc=started_at_utc,
                finished_at_utc=_utc_now(),
                elapsed_seconds=time.monotonic() - started_monotonic,
                error=exc,
            ),
        )
        raise
    receipt_path = _receipt_path(graph_dir)
    _atomic_write_json(
        receipt_path,
        _success_run_receipt(
            graph_dir=graph_dir,
            result=result,
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now(),
            elapsed_seconds=time.monotonic() - started_monotonic,
        ),
    )
    result = {**result, "build_run_receipt": str(receipt_path)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    _enable_build_progress()
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
