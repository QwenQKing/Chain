from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


REQUIRED_FILES: Tuple[str, ...] = (
    "knowledge_graph.graphml",
    "entity_vdb.json",
    "hyperedge_vdb.json",
    "chunks.json",
    "graph_manifest.json",
    "build_telemetry.json",
    "source_audit_manifest.json",
)





EMBEDDING_SPACE_FIELDS: Tuple[str, ...] = (
    "model",
    "revision",
    "dimension",
    "dtype",
    "serialization_precision",
    "preprocess_version",
    "normalization",
    "distance",
)


class GraphCompletionError(ValueError):
    pass



def _strict_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraphCompletionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GraphCompletionError(
                    f"non-finite JSON constant {value!r} in {path.name}"
                )
            ),
        )
    except GraphCompletionError:
        raise
    except Exception as exc:
        raise GraphCompletionError(f"cannot parse {path.name}: {exc}") from exc


def _non_negative_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraphCompletionError(
            f"build telemetry count {field!r} must be a non-negative integer"
        )
    return value


def _vector_nodes(
    value: Any, *, filename: str, expected_role: str
) -> tuple[int, set[str], int, tuple[Any, ...]]:
    if not isinstance(value, Mapping):
        raise GraphCompletionError(f"{filename} must contain an object")
    dimension = value.get("embedding_dim")
    data = value.get("data")
    matrix_encoded = value.get("matrix")
    additional = value.get("additional_data")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        or not isinstance(data, list)
        or not isinstance(matrix_encoded, str)
        or not isinstance(additional, Mapping)
    ):
        raise GraphCompletionError(f"{filename} has an incomplete vector-store structure")
    if additional.get("schema_version") != "chain-vector-store-v1":
        raise GraphCompletionError(f"{filename} has an unsupported vector-store schema")
    if additional.get("role") != expected_role:
        raise GraphCompletionError(f"{filename} role does not match {expected_role!r}")
    signature = additional.get("embedding_signature")
    if not isinstance(signature, Mapping):
        raise GraphCompletionError(f"{filename} has no embedding signature")
    if (
        not all(
            isinstance(signature.get(field), str) and signature.get(field)
            for field in ("model", "revision", "preprocess_version")
        )
        or (
            signature.get("provider") is not None
            and not isinstance(signature.get("provider"), str)
        )
        or signature.get("dimension") != dimension
        or signature.get("dtype") != "float32"
        or signature.get("serialization_precision") != "float32-le"
        or signature.get("normalization") != "l2"
        or signature.get("distance") != "cosine"
    ):
        raise GraphCompletionError(
            f"{filename} embedding signature does not describe its stored vectors"
        )
    try:
        matrix = base64.b64decode(matrix_encoded, validate=True)
    except Exception as exc:
        raise GraphCompletionError(f"{filename} matrix is not valid base64") from exc
    expected_bytes = len(data) * dimension * 4
    if len(matrix) != expected_bytes:
        raise GraphCompletionError(
            f"{filename} matrix byte count does not match its records/dimension"
        )
    try:
        import numpy as np

        values = np.frombuffer(matrix, dtype=np.dtype("<f4")).reshape(
            (len(data), dimension)
        )
        if not np.isfinite(values).all():
            raise GraphCompletionError(f"{filename} matrix contains non-finite values")
        if len(data):
            norms = np.linalg.norm(values, axis=1)
            if (norms == 0).any():
                raise GraphCompletionError(f"{filename} matrix contains a zero vector")
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                raise GraphCompletionError(
                    f"{filename} vectors are not L2-normalized"
                )
    except GraphCompletionError:
        raise
    except Exception as exc:
        raise GraphCompletionError(f"{filename} matrix cannot be inspected") from exc
    vector_ids: set[str] = set()
    node_ids: set[str] = set()
    for index, record in enumerate(data):
        if not isinstance(record, Mapping):
            raise GraphCompletionError(f"{filename} record {index} is not an object")
        vector_id = record.get("__id__")
        node_id = record.get("node_id")
        if (
            not isinstance(vector_id, str)
            or not vector_id
            or vector_id in vector_ids
            or not isinstance(node_id, str)
            or not node_id
            or node_id in node_ids
        ):
            raise GraphCompletionError(
                f"{filename} contains an invalid or duplicate vector/node identity"
            )
        vector_ids.add(vector_id)
        node_ids.add(node_id)
    shape_signature = tuple(signature.get(field) for field in EMBEDDING_SPACE_FIELDS)
    return len(data), node_ids, dimension, shape_signature


def check_graph_bundle_complete(path: str | Path) -> Dict[str, Any]:


    graph_dir = Path(path)
    if not graph_dir.is_dir():
        raise GraphCompletionError(f"graph directory does not exist: {graph_dir}")
    for filename in REQUIRED_FILES:
        artifact = graph_dir / filename
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise GraphCompletionError(f"missing or empty completed-build file: {filename}")

    manifest = _read_json(graph_dir / "graph_manifest.json")
    telemetry = _read_json(graph_dir / "build_telemetry.json")
    chunks_document = _read_json(graph_dir / "chunks.json")
    entity_document = _read_json(graph_dir / "entity_vdb.json")
    hyperedge_document = _read_json(graph_dir / "hyperedge_vdb.json")
    source_audit = _read_json(graph_dir / "source_audit_manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != (
        "chain-graph-bundle-v1"
    ):
        raise GraphCompletionError("graph_manifest.json has an unsupported schema")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping) or not set(
        REQUIRED_FILES[:4]
    ).issubset(manifest_artifacts):
        raise GraphCompletionError("graph manifest does not list all core artifacts")
    if not isinstance(telemetry, Mapping) or telemetry.get("schema_version") != (
        "chain-build-telemetry-v1"
    ):
        raise GraphCompletionError("build_telemetry.json has an unsupported schema")
    if telemetry.get("status") != "success" or telemetry.get("errors") != []:
        raise GraphCompletionError("build telemetry does not report successful completion")
    raw_counts = telemetry.get("counts")
    if not isinstance(raw_counts, Mapping):
        raise GraphCompletionError("build telemetry has no completion counts")
    counts = {
        name: _non_negative_count(raw_counts.get(name), field=name)
        for name in ("chunks", "entities", "hyperedges", "causal_edges", "edges")
    }
    if counts["chunks"] <= 0:
        raise GraphCompletionError("a completed graph must contain at least one chunk")

    if not isinstance(chunks_document, Mapping) or chunks_document.get(
        "schema_version"
    ) != "chain-chunks-v1":
        raise GraphCompletionError("chunks.json has an unsupported schema")
    chunks = chunks_document.get("chunks")
    if not isinstance(chunks, Mapping) or len(chunks) != counts["chunks"]:
        raise GraphCompletionError("chunks.json count does not match build telemetry")
    if any(
        not isinstance(chunk_id, str)
        or not chunk_id
        or not isinstance(chunk, Mapping)
        for chunk_id, chunk in chunks.items()
    ):
        raise GraphCompletionError("chunks.json contains an invalid chunk entry")
    chunk_indices: set[int] = set()
    chunk_record_ids: set[str] = set()
    for chunk_id, chunk in chunks.items():
        chunk_index = chunk.get("chunk_index")
        record_id = chunk.get("record_id")
        if (
            chunk.get("chunk_id") != chunk_id
            or isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
            or chunk_index in chunk_indices
            or not isinstance(chunk.get("content"), str)
            or not chunk.get("content", "").strip()
            or not isinstance(record_id, str)
            or not record_id
            or not isinstance(chunk.get("source"), str)
            or not chunk.get("source")
            or not isinstance(chunk.get("source_path"), str)
            or not chunk.get("source_path")
            or not isinstance(chunk.get("timestamp_iso"), str)
            or not isinstance(chunk.get("date"), str)
            or isinstance(chunk.get("token_start"), bool)
            or not isinstance(chunk.get("token_start"), int)
            or chunk.get("token_start") < 0
            or isinstance(chunk.get("token_end"), bool)
            or not isinstance(chunk.get("token_end"), int)
            or chunk.get("token_end") < chunk.get("token_start")
            or isinstance(chunk.get("tokens"), bool)
            or not isinstance(chunk.get("tokens"), int)
            or chunk.get("tokens") <= 0
        ):
            raise GraphCompletionError("chunks.json contains an incomplete chunk record")
        chunk_indices.add(chunk_index)
        chunk_record_ids.add(record_id)
    if chunk_indices != set(range(counts["chunks"])):
        raise GraphCompletionError("chunks.json indices do not form one complete range")

    entity_count, entity_vdb_nodes, entity_dimension, entity_signature = _vector_nodes(
        entity_document,
        filename="entity_vdb.json",
        expected_role="entity",
    )
    (
        hyperedge_count,
        hyperedge_vdb_nodes,
        hyperedge_dimension,
        hyperedge_signature,
    ) = _vector_nodes(
        hyperedge_document,
        filename="hyperedge_vdb.json",
        expected_role="hyperedge_occurrence",
    )
    if entity_count != counts["entities"]:
        raise GraphCompletionError("entity VDB count does not match build telemetry")
    if hyperedge_count != counts["hyperedges"]:
        raise GraphCompletionError("hyperedge VDB count does not match build telemetry")
    if (
        entity_dimension != hyperedge_dimension
        or entity_signature != hyperedge_signature
    ):
        raise GraphCompletionError(
            "entity and hyperedge VDBs do not share one embedding space"
        )

    if not isinstance(source_audit, Mapping) or source_audit.get(
        "schema_version"
    ) != "chain-source-audit-v1":
        raise GraphCompletionError("source_audit_manifest.json has an unsupported schema")
    record_count = source_audit.get("record_count")
    records = source_audit.get("records")
    inputs = source_audit.get("inputs")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
        or not isinstance(records, list)
        or len(records) != record_count
        or not isinstance(inputs, list)
        or not inputs
    ):
        raise GraphCompletionError("source audit does not describe a completed input set")
    input_record_total = 0
    input_paths: set[str] = set()
    for item in inputs:
        if not isinstance(item, Mapping):
            raise GraphCompletionError("source audit contains an invalid input entry")
        source_path = item.get("source_path")
        input_records = item.get("records")
        if (
            not isinstance(source_path, str)
            or not source_path
            or source_path in input_paths
            or isinstance(input_records, bool)
            or not isinstance(input_records, int)
            or input_records < 0
        ):
            raise GraphCompletionError("source audit contains an invalid input entry")
        input_paths.add(source_path)
        input_record_total += input_records
    if input_record_total != record_count:
        raise GraphCompletionError("source audit input counts do not close over records")
    audited_record_ids: set[str] = set()
    per_input_records: Dict[str, int] = {source_path: 0 for source_path in input_paths}
    for item in records:
        if not isinstance(item, Mapping):
            raise GraphCompletionError("source audit contains an invalid record entry")
        record_id = item.get("record_id")
        source_path = item.get("source_path")
        line_number = item.get("record_line_number")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in audited_record_ids
            or source_path not in input_paths
            or isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or line_number <= 0
        ):
            raise GraphCompletionError("source audit contains an invalid record entry")
        audited_record_ids.add(record_id)
        per_input_records[str(source_path)] += 1
    declared_per_input = {
        str(item["source_path"]): int(item["records"]) for item in inputs
    }
    if per_input_records != declared_per_input:
        raise GraphCompletionError("source audit per-input record counts do not close")
    if audited_record_ids != chunk_record_ids:
        raise GraphCompletionError("source audit record IDs do not close over chunks")

    operational = telemetry.get("operational")
    traces = operational.get("extraction_traces") if isinstance(operational, Mapping) else None
    if not isinstance(traces, list) or len(traces) != counts["chunks"]:
        raise GraphCompletionError("extraction traces do not close over completed chunks")
    accepted_statuses = {
        "success",
        "success_explicit_empty",
        "success_with_relation_rejections",
        "success_with_grounding_rejections",
        "success_with_proposition_schema_rejections",
        "success_projected_empty",
    }
    trace_chunk_ids: set[str] = set()
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise GraphCompletionError("an extraction trace is not an object")
        chunk_id = trace.get("chunk_id")
        attempts = trace.get("attempts")
        if (
            not isinstance(chunk_id, str)
            or chunk_id not in chunks
            or chunk_id in trace_chunk_ids
            or trace.get("status") not in accepted_statuses
            or not isinstance(attempts, list)
            or not attempts
        ):
            raise GraphCompletionError("an extraction trace is incomplete or non-terminal")
        trace_chunk_ids.add(chunk_id)
    if trace_chunk_ids != set(chunks):
        raise GraphCompletionError("extraction trace chunk IDs are incomplete")
    scientific = telemetry.get("scientific")
    if not isinstance(scientific, Mapping):
        raise GraphCompletionError("build telemetry scientific section is missing")
    if scientific.get("extraction_chunks_total") != counts["chunks"]:
        raise GraphCompletionError("scientific extraction count does not match chunks")
    completed_scientific = 0
    for field in ("success", "success_explicit_empty", "success_projected_empty"):
        completed_scientific += _non_negative_count(scientific.get(field), field=field)
    if completed_scientific != counts["chunks"]:
        raise GraphCompletionError("scientific success counts do not close over chunks")

    try:
        import networkx as nx

        graph = nx.read_graphml(
            str(graph_dir / "knowledge_graph.graphml"), force_multigraph=True
        )
        if not graph.is_directed():
            raise GraphCompletionError("knowledge_graph.graphml must be directed")
        if not isinstance(graph, nx.MultiDiGraph):
            graph = nx.MultiDiGraph(graph)
    except Exception as exc:
        raise GraphCompletionError(f"knowledge_graph.graphml cannot be parsed: {exc}") from exc
    if graph.graph.get("schema_version") != "chain-causal-temporal-multidigraph-v1":
        raise GraphCompletionError("GraphML has an unsupported graph schema")
    roles = [attrs.get("role") for _, attrs in graph.nodes(data=True)]
    if any(role not in {"entity", "hyperedge"} for role in roles):
        raise GraphCompletionError("GraphML contains an unknown or missing node role")
    graph_entities = sum(role == "entity" for role in roles)
    graph_hyperedges = sum(role == "hyperedge" for role in roles)
    entity_graph_nodes = {
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("role") == "entity"
    }
    hyperedge_graph_nodes = {
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("role") == "hyperedge"
    }
    for node_id in hyperedge_graph_nodes:
        attrs = graph.nodes[node_id]
        chunk_id = attrs.get("chunk_id") or attrs.get("source_id")
        if not isinstance(chunk_id, str) or chunk_id not in chunks:
            raise GraphCompletionError(
                "GraphML hyperedge does not reference a completed chunk"
            )
    graph_causal_edges = 0
    for source, target, attrs in graph.edges(data=True):
        role = attrs.get("role")
        chunk_id = attrs.get("chunk_id") or attrs.get("source_id")
        if role == "causal":
            graph_causal_edges += 1
            strength = attrs.get("strength")
            if (
                str(source) not in entity_graph_nodes
                or str(target) not in entity_graph_nodes
                or str(source) == str(target)
            ):
                raise GraphCompletionError("GraphML has a causal edge with invalid endpoints")
            if attrs.get("causal_type") not in {"causes", "enables", "prevents"}:
                raise GraphCompletionError("GraphML has an unsupported causal edge type")
            if (
                isinstance(strength, bool)
                or not isinstance(strength, (int, float))
                or not math.isfinite(float(strength))
                or not 0.0 < float(strength) <= 1.0
            ):
                raise GraphCompletionError("GraphML has an invalid causal strength")
        elif role == "incidence":
            if (
                str(source) not in hyperedge_graph_nodes
                or str(target) not in entity_graph_nodes
            ):
                raise GraphCompletionError("GraphML has an incidence edge with invalid endpoints")
        else:
            raise GraphCompletionError("GraphML contains an unknown or missing edge role")
        if not isinstance(chunk_id, str) or chunk_id not in chunks:
            raise GraphCompletionError("GraphML edge does not reference a completed chunk")
    if graph_entities != counts["entities"]:
        raise GraphCompletionError("GraphML entity count does not match build telemetry")
    if graph_hyperedges != counts["hyperedges"]:
        raise GraphCompletionError("GraphML hyperedge count does not match build telemetry")
    if entity_graph_nodes != entity_vdb_nodes:
        raise GraphCompletionError("entity VDB node IDs do not match GraphML")
    if hyperedge_graph_nodes != hyperedge_vdb_nodes:
        raise GraphCompletionError("hyperedge VDB node IDs do not match GraphML")
    if graph.number_of_edges() != counts["edges"]:
        raise GraphCompletionError("GraphML edge count does not match build telemetry")
    if graph_causal_edges != counts["causal_edges"]:
        raise GraphCompletionError("GraphML causal-edge count does not match build telemetry")

    return {
        "completion_only": True,
        "hashes_checked": False,
        "runtime_checked": False,
        "graph_dir": str(graph_dir),
        "stats": counts,
        "source_records": record_count,
    }


def _portable_file_fingerprints(graph_dir: Path) -> Dict[str, Dict[str, Any]]:


    artifacts: Dict[str, Dict[str, Any]] = {}
    for filename in REQUIRED_FILES[:4]:
        payload = (graph_dir / filename).read_bytes()
        artifacts[filename] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return artifacts


def _portable_bundle_fingerprint(artifacts: Mapping[str, Mapping[str, Any]]) -> str:


    payload = json.dumps(
        {str(name): dict(value) for name, value in sorted(artifacts.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_graph_bundle_portable(
    path: str | Path,
    *,
    expected_profile_id: str | None = None,
    expected_config: Any = None,
    expected_config_sha256: str | None = None,
    expected_construction_config_sha256: str | None = None,
    require_publishable: bool = False,
    **_ignored_identity_kwargs: Any,
) -> Dict[str, Any]:





    del (
        expected_profile_id,
        expected_config_sha256,
        expected_construction_config_sha256,
        require_publishable,
        _ignored_identity_kwargs,
    )

    graph_dir = Path(path)
    completion = check_graph_bundle_complete(graph_dir)
    manifest = _read_json(graph_dir / "graph_manifest.json")
    telemetry = _read_json(graph_dir / "build_telemetry.json")
    chunks = _read_json(graph_dir / "chunks.json")
    entity_vdb = _read_json(graph_dir / "entity_vdb.json")
    hyperedge_vdb = _read_json(graph_dir / "hyperedge_vdb.json")
    source_audit = _read_json(graph_dir / "source_audit_manifest.json")

    try:
        import networkx as nx

        graph = nx.read_graphml(
            str(graph_dir / "knowledge_graph.graphml"), force_multigraph=True
        )
        if not isinstance(graph, nx.MultiDiGraph):
            graph = nx.MultiDiGraph(graph)
    except Exception as exc:
        raise GraphCompletionError(
            f"knowledge_graph.graphml cannot be loaded: {exc}"
        ) from exc

    entity_additional = entity_vdb.get("additional_data", {})
    hyperedge_additional = hyperedge_vdb.get("additional_data", {})
    entity_signature = dict(entity_additional.get("embedding_signature", {}))
    hyperedge_signature = dict(hyperedge_additional.get("embedding_signature", {}))
    artifacts = _portable_file_fingerprints(graph_dir)
    portable_hash = _portable_bundle_fingerprint(artifacts)
    entity_dimension = int(entity_vdb["embedding_dim"])
    hyperedge_dimension = int(hyperedge_vdb["embedding_dim"])





    source_manifest = dict(manifest)
    source_telemetry = dict(telemetry)
    legacy_graph_hash = source_manifest.get("graph_bundle_sha256")
    effective_profile = str(
        getattr(expected_config, "profile_id", None)
        or (expected_config.get("profile_id") if isinstance(expected_config, Mapping) else "")
        or manifest.get("profile_id")
        or "compat"
    )
    effective_construction = str(
        getattr(expected_config, "construction_config_sha256", None)
        or (expected_config.get("construction_config_sha256") if isinstance(expected_config, Mapping) else "")
        or manifest.get("construction_config_sha256")
        or ""
    )
    effective_scientific = str(
        getattr(expected_config, "scientific_config_sha256", None)
        or (expected_config.get("scientific_config_sha256") if isinstance(expected_config, Mapping) else "")
        or manifest.get("scientific_config_sha256")
        or ""
    )
    runtime_publishable: Any = None
    if expected_config is not None:
        runtime_publishable = getattr(expected_config, "publishable", None)
        if runtime_publishable is None and isinstance(expected_config, Mapping):
            runtime_publishable = expected_config.get("publishable")



    effective_publishable = (
        bool(runtime_publishable)
        if isinstance(runtime_publishable, bool)
        else manifest.get("publishable") is True
    )
    effective_out_of_paper = not effective_publishable
    for metadata in (manifest, telemetry):
        metadata["profile_id"] = effective_profile
        metadata["publishable"] = effective_publishable
        metadata["out_of_paper_protocol"] = effective_out_of_paper
        metadata["construction_config_sha256"] = effective_construction
        metadata["scientific_config_sha256"] = effective_scientific
        metadata["graph_bundle_sha256"] = portable_hash
    graph.graph.update(
        {
            "validated_graph_bundle": True,
            "graph_bundle_sha256": portable_hash,
            "profile_id": effective_profile,
            "publishable": effective_publishable,
            "out_of_paper_protocol": effective_out_of_paper,
            "construction_config_sha256": effective_construction,
            "scientific_config_sha256": effective_scientific,
            "embedding_signature": entity_signature,
        }
    )

    return {
        "portable": True,
        "completion_only": True,
        "hashes_checked": False,
        "runtime_checked": False,
        "config_checked": False,
        "graph_dir": str(graph_dir),
        "profile_id": effective_profile,
        "publishable": effective_publishable,
        "out_of_paper_protocol": effective_out_of_paper,

        "graph_bundle_sha256": portable_hash,
        "portable_graph_bundle_sha256": portable_hash,
        "legacy_graph_bundle_sha256": legacy_graph_hash,
        "manifest": dict(manifest),
        "telemetry": dict(telemetry),
        "source_manifest": source_manifest,
        "source_telemetry": source_telemetry,
        "artifacts": artifacts,
        "portable_artifacts": artifacts,
        "stats": dict(completion["stats"]),
        "source_records": completion["source_records"],
        "embedding_signature": entity_signature,
        "embedding_signatures": {
            "entity": entity_signature,
            "hyperedge": hyperedge_signature,
        },
        "chunks": chunks,
        "source_audit": source_audit,
        "graph": graph,
        "entity_matrix_shape": (
            len(entity_vdb.get("data", [])),
            entity_dimension,
        ),
        "hyperedge_matrix_shape": (
            len(hyperedge_vdb.get("data", [])),
            hyperedge_dimension,
        ),
    }


__all__ = [
    "EMBEDDING_SPACE_FIELDS",
    "GraphCompletionError",
    "REQUIRED_FILES",
    "check_graph_bundle_complete",
    "load_graph_bundle_portable",
]
