from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from chain.config import ResolvedConfig
from chain.graph.hypergraph import MMDTHypergraph
from chain.graph.kg_builder import (
    InputRecordError,
    load_source_records,
    normalize_source_descriptor,
)


class CollectionError(RuntimeError):
    pass


def collect_and_build(
    question: str,
    hypergraph: MMDTHypergraph,
    time_cutoff: str = "",
    extra_queries: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    *,
    config: Optional[ResolvedConfig] = None,
    source_descriptor: str = "",
) -> Dict[str, Any]:
    if not isinstance(config, ResolvedConfig):
        raise CollectionError("online ingest requires one ResolvedConfig")
    resolved = config
    if not isinstance(question, str) or not question.strip():
        raise CollectionError("question must be non-empty")
    if not isinstance(time_cutoff, str) or not time_cutoff.strip():
        raise CollectionError("online ingest requires an explicit prediction cutoff")
    try:
        descriptor = normalize_source_descriptor(source_descriptor)
    except InputRecordError as exc:
        raise CollectionError(
            "source_descriptor must be a non-absolute stable identifier"
        ) from exc
    if extra_queries is not None and (
        not isinstance(extra_queries, list)
        or not all(isinstance(item, str) for item in extra_queries)
    ):
        raise CollectionError("extra_queries must be a string list")
    source = Path(data_dir or resolved.knowledge_files_dir)
    if not source.exists() or not (source.is_file() or source.is_dir()):
        raise CollectionError(f"knowledge source does not exist: {source}")
    allow_compat = resolved.profile != "paper"
    try:
        records, source_audit = load_source_records(
            source,
            allow_compat_adapters=allow_compat,
        )
    except Exception as exc:
        raise CollectionError(f"source preflight failed for {source}: {exc}") from exc
    if not records:
        raise CollectionError(f"knowledge source contains no records: {source}")
    replacement_builder = getattr(hypergraph, "build_replacement_bundle", None)
    if not callable(replacement_builder):
        raise CollectionError(
            "online ingest requires the staging/atomic build_replacement_bundle interface"
        )
    try:
        built = replacement_builder(
            str(source),
            source_descriptor=descriptor,
            config=resolved,
        )
    except Exception as exc:
        raise CollectionError(f"staged ingest failed for {source}: {exc}") from exc
    if not isinstance(built, dict):
        raise CollectionError("staged ingest returned a non-object result")
    receipt = built.get("validation") or built.get("graph_validation") or built
    if not isinstance(receipt, dict):
        raise CollectionError("staged ingest returned no validation receipt")
    bundle_hash = str(receipt.get("graph_bundle_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_hash):
        raise CollectionError("staged ingest returned no canonical graph_bundle_sha256")
    stats = built.get("stats") or built.get("ingest_stats") or {}
    if not isinstance(stats, dict):
        raise CollectionError("staged ingest stats must be an object")
    files_processed = stats.get("files_processed", len(source_audit.get("inputs", [])))
    if isinstance(files_processed, bool) or not isinstance(files_processed, int):
        raise CollectionError("ingest result lacks integer files_processed")
    if files_processed <= 0:
        raise CollectionError("ingest processed no files")
    safe_stats = {
        key: value
        for key, value in stats.items()
        if key in {"files_processed", "records_processed", "chunks_processed", "entities_processed"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    safe_stats["files_processed"] = files_processed
    return {
        "queries": list(extra_queries or []),
        "search_results": files_processed,
        "ingest_stats": safe_stats,
        "cutoff": time_cutoff,
        "source_descriptor": descriptor,
        "graph_bundle_sha256": bundle_hash,
        "profile_id": resolved.profile_id,
        "scientific_config_sha256": resolved.scientific_config_sha256,
    }
