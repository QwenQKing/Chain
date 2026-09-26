from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import deque
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from weakref import WeakKeyDictionary

import networkx as nx

from chain.config import ResolvedConfig, canonical_json, validate_binary_axis
from chain.skills.temporal_validity import (
    TemporalValidityScorer,
    assess_evidence_admissibility,
    parse_cutoff,
)


_CAUSAL_TYPES = frozenset({"causes", "enables", "prevents"})
_PAPER_PROFILE = "paper"
_ADMISSIBILITY_POLICY = "chain-cutoff-admissibility-v1"






_GRAPH_CACHE_LOCK = RLock()
_TEMPORAL_INDEX_CACHE: "WeakKeyDictionary[Any, Dict[str, Any]]" = (
    WeakKeyDictionary()
)
_SORTED_EDGE_CACHE: "WeakKeyDictionary[Any, Tuple[Tuple[Any, Any, str, Dict[str, Any]], ...]]" = (
    WeakKeyDictionary()
)


def _stable_id(prefix: str, *parts: Any) -> str:
    return prefix + hashlib.sha256(
        canonical_json(list(parts)).encode("utf-8")
    ).hexdigest()


def _config_value(
    config: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    if config is None:
        return default
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _profile_name(config: Any) -> str:
    return str(
        _config_value(
            config,
            ("profile_id", "profile", "profile_name", "schema_profile"),
            "",
        )
        or ""
    )


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number, not {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return parsed


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validated_causal_precompute(
    precomputed: Any,
    *,
    graph: Any,
    case_id: str,
    cutoff: Any,
    graph_bundle_sha256: str,
    config: ResolvedConfig,
    target_signs: Mapping[str, int],
    tvf_scores: Optional[Mapping[str, float]],
) -> Dict[str, Any]:


    if not isinstance(precomputed, Mapping):
        raise ValueError("causal precomputation must be an object")
    if tvf_scores is not None:
        raise ValueError("causal precomputation cannot be combined with tvf_scores")
    if precomputed.get("schema_version") != "chain-causal-precompute-v1":
        raise ValueError("causal precomputation schema mismatch")
    if precomputed.get("active_graph") is not graph:
        raise ValueError("causal precomputation active graph identity mismatch")
    expected_cutoff = parse_cutoff(cutoff).canonical
    expected_targets = {
        str(key): int(value)
        for key, value in sorted(target_signs.items(), key=lambda item: str(item[0]))
    }
    expected = {
        "active_view_id": str(getattr(graph, "graph", {}).get("active_view_id", "")),
        "case_id": str(case_id),
        "cutoff": expected_cutoff,
        "graph_bundle_sha256": str(graph_bundle_sha256),
        "scientific_config_sha256": str(config.scientific_config_sha256),
        "target_signs": expected_targets,
        "ablation_mode": str(config.ablation_mode),
        "d_max": int(config.d_max),
        "eta": float(config.eta),
        "B_pi": int(config.B_pi),
        "F_max": config.F_max,
        "tau_phi": float(config.tau_phi),
    }
    for field, expected_value in expected.items():
        actual = precomputed.get(field)
        if field in {"eta", "tau_phi"}:
            try:
                matches = math.isclose(float(actual), float(expected_value), rel_tol=0.0, abs_tol=0.0)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(f"causal precomputation {field} identity mismatch")

    distances = precomputed.get("causal_distances")
    scores = precomputed.get("node_tvf_scores")
    decisions = precomputed.get("ctvf_decisions")
    n_q = precomputed.get("n_q")
    if not isinstance(distances, Mapping):
        raise ValueError("causal precomputation distances must be an object")
    if not isinstance(scores, Mapping):
        raise ValueError("causal precomputation node scores must be an object")
    if not isinstance(decisions, Mapping):
        raise ValueError("causal precomputation CTVF decisions must be an object")
    if not isinstance(n_q, Mapping):
        raise ValueError("causal precomputation frequency table must be an object")
    validated_distances: Dict[str, float] = {}
    for key, value in distances.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("causal precomputation distance must be numeric")
        parsed = float(value)
        if math.isnan(parsed) or parsed < 0.0:
            raise ValueError("causal precomputation distance is invalid")
        validated_distances[str(key)] = parsed
    validated_scores: Dict[str, float] = {}
    for key, value in scores.items():
        validated_scores[str(key)] = _finite_probability(value, "cached CTVF score")
    graph_node_ids = {str(node_id) for node_id in graph.nodes}
    graph_entity_ids = {
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("role") == "entity"
    }
    graph_hyperedge_ids = {
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("role") == "hyperedge"
    }
    if set(validated_scores) != graph_node_ids:
        raise ValueError("causal precomputation node-score coverage mismatch")
    if not graph_entity_ids <= set(validated_distances):
        raise ValueError("causal precomputation entity-distance coverage mismatch")
    if not set(validated_distances) <= graph_node_ids:
        raise ValueError("causal precomputation contains an unknown distance node")
    normalized_decisions = {str(key): value for key, value in decisions.items()}
    if set(normalized_decisions) != graph_hyperedge_ids:
        raise ValueError("causal precomputation CTVF-decision coverage mismatch")
    if not all(isinstance(value, Mapping) for value in normalized_decisions.values()):
        raise ValueError("cached CTVF decisions must be objects")
    validated_n_q: Dict[str, int] = {}
    for key, value in n_q.items():
        validated_n_q[str(key)] = _nonnegative_int(value, "cached N_q")
    return {
        "causal_distances": validated_distances,
        "node_tvf_scores": validated_scores,
        "ctvf_decisions": copy.deepcopy(normalized_decisions),
        "n_q": validated_n_q,
    }


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _iter_keyed_edges(graph: Any) -> Iterable[Tuple[Any, Any, str, Dict[str, Any]]]:
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(keys=True, data=True):
            yield source, target, str(key), dict(data)
    else:
        for source, target, data in graph.edges(data=True):
            attrs = dict(data)
            key = str(
                attrs.get("edge_id")
                or attrs.get("logical_edge_id")
                or _stable_id(
                    "compat_edge_",
                    str(source),
                    str(target),
                    attrs.get("role", ""),
                    attrs.get("causal_type", attrs.get("relation", "")),
                    attrs,
                )
            )
            yield source, target, key, attrs


def _sorted_keyed_edges(
    graph: Any,
) -> Tuple[Tuple[Any, Any, str, Dict[str, Any]], ...]:


    with _GRAPH_CACHE_LOCK:
        cached = _SORTED_EDGE_CACHE.get(graph)
        if cached is not None:
            return cached
        ordered = tuple(
            sorted(
                _iter_keyed_edges(graph),
                key=lambda item: (str(item[0]), str(item[1]), item[2]),
            )
        )
        _SORTED_EDGE_CACHE[graph] = ordered
        return ordered


def _hyperedge_temporal_index(graph: Any) -> Dict[str, Any]:


    with _GRAPH_CACHE_LOCK:
        cached = _TEMPORAL_INDEX_CACHE.get(graph)
        if cached is not None:
            return cached

        temporal_fields = (
            "record_id",
            "timestamp_raw",
            "timestamp_iso",
            "date",
            "availability_upper_bound_iso",
            "availability_upper_bound_date",
            "availability_bound_kind",
            "availability_bound_source",
            "availability_bound_source_sha256",
        )
        first_by_chunk: Dict[str, Dict[str, Any]] = {}
        signatures_by_chunk: Dict[str, set] = {}
        for node_id, node_data in graph.nodes(data=True):



            if node_data.get("role") != "hyperedge":
                continue
            chunk_id = str(
                node_data.get("chunk_id") or node_data.get("source_id") or ""
            )
            if not chunk_id:
                continue
            first_by_chunk.setdefault(chunk_id, dict(node_data))
            signatures_by_chunk.setdefault(chunk_id, set()).add(
                tuple(str(node_data.get(field, "")) for field in temporal_fields)
            )

        index: Dict[str, Any] = {
            "first_by_chunk": first_by_chunk,
            "signatures_by_chunk": signatures_by_chunk,
        }
        _TEMPORAL_INDEX_CACHE[graph] = index
        return index


def _edges_between(graph: Any, source: Any, target: Any) -> Iterable[Dict[str, Any]]:
    data = graph.get_edge_data(source, target, default={})
    if graph.is_multigraph():
        for key in sorted(data, key=str):
            yield dict(data[key])
    elif data:
        yield dict(data)


def _is_causal(data: Mapping[str, Any]) -> bool:
    return data.get("role") == "causal" or data.get("causal_type") in _CAUSAL_TYPES


def _causal_type(data: Mapping[str, Any]) -> str:
    value = str(data.get("causal_type", data.get("type", ""))).strip().casefold()
    if value not in _CAUSAL_TYPES:
        raise ValueError(f"unsupported causal type: {value!r}")
    return value


def _edge_strength(data: Mapping[str, Any]) -> float:
    if "strength" in data:
        value = data["strength"]
    elif "weight" in data:
        value = data["weight"]
    else:
        raise ValueError("causal edge is missing strength")
    strength = _finite_probability(value, "causal strength")
    if strength <= 0.0:
        raise ValueError("causal strength must be strictly positive")
    return strength


def _occurrence_id(key: str, data: Mapping[str, Any]) -> str:
    return str(
        data.get("edge_id")
        or data.get("occurrence_id")
        or data.get("causal_occurrence_id")
        or key
    )


def _occurrence_temporal_data(
    graph: Any,
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for field in ("provenance", "source_provenance", "temporal_provenance"):
        value = data.get(field)
        if isinstance(value, Mapping):
            merged.update(value)
    for node_field in (
        "source_hyperedge_id",
        "hyperedge_id",
        "source_occurrence_id",
        "proposition_occurrence_id",
    ):
        node_id = data.get(node_field)
        if node_id in graph:
            merged.update(dict(graph.nodes[node_id]))
            break
    else:
        chunk_id = str(data.get("chunk_id") or data.get("source_id") or "")
        if chunk_id:
            temporal_index = _hyperedge_temporal_index(graph)
            first_match = temporal_index["first_by_chunk"].get(chunk_id)
            if first_match is None:
                raise ValueError(
                    f"causal occurrence refers to chunk without hyperedge provenance: {chunk_id}"
                )
            signatures = temporal_index["signatures_by_chunk"][chunk_id]
            if len(signatures) != 1:
                raise ValueError(
                    f"chunk has inconsistent temporal provenance across occurrences: {chunk_id}"
                )
            merged.update(first_match)
    merged.update(dict(data))
    return merged


def _admitted_aliases(
    node_data: Mapping[str, Any],
    admitted_occurrence_ids: set,
    *,
    normalized: bool = False,
) -> List[str]:




    if normalized:
        if not isinstance(admitted_occurrence_ids, (set, frozenset)):
            raise TypeError("normalized occurrence IDs must be a set")
        admitted = admitted_occurrence_ids
    else:
        admitted = {str(item) for item in admitted_occurrence_ids}
    provenance = node_data.get("alias_provenance")
    aliases: List[str] = []
    if isinstance(provenance, Mapping):
        for alias, supports in provenance.items():
            if isinstance(supports, Mapping):
                supports = (
                    supports.get("occurrence_ids")
                    or supports.get("source_occurrence_ids")
                    or supports.get("hyperedge_ids")
                    or []
                )
            if isinstance(supports, str):
                supports = [supports]
            support_ids: set[str] = set()
            if isinstance(supports, (list, tuple, set)):
                for item in supports:
                    if isinstance(item, Mapping):
                        for key in (
                            "occurrence_id",
                            "hyperedge_id",
                            "source_occurrence_id",
                            "source_hyperedge_id",
                            "record_id",
                            "chunk_id",
                        ):
                            value = item.get(key)
                            if value is not None and str(value).strip():
                                support_ids.add(str(value))
                    else:
                        support_ids.add(str(item))
            if support_ids & admitted:
                aliases.append(str(alias))
    elif isinstance(provenance, list):
        for item in provenance:
            if not isinstance(item, Mapping):
                continue
            occurrence_id = str(
                item.get("occurrence_id")
                or item.get("hyperedge_id")
                or item.get("source_occurrence_id")
                or ""
            )
            alias = str(item.get("alias") or item.get("name") or "").strip()
            if occurrence_id in admitted and alias:
                aliases.append(alias)
    return sorted(set(aliases), key=lambda item: (item.casefold(), item))


def _safe_entity_data(
    node_id: Any,
    node_data: Mapping[str, Any],
    admitted_occurrence_ids: set,
    *,
    admitted_occurrence_ids_normalized: bool = False,
) -> Dict[str, Any]:
    canonical_key = str(
        node_data.get("canonical_entity_key")
        or node_data.get("entity_key")
        or node_data.get("canonical_key")
        or node_id
    ).strip()
    aliases = _admitted_aliases(
        node_data,
        admitted_occurrence_ids,
        normalized=admitted_occurrence_ids_normalized,
    )
    display_alias = aliases[0] if aliases else canonical_key
    safe = {
        "role": "entity",
        "canonical_entity_key": canonical_key,
        "name": display_alias,
        "display_alias": display_alias,
        "admitted_aliases": aliases,
    }
    for key in ("entity_vdb_id", "vector_id", "embedding_signature_sha256"):
        if key in node_data:
            safe[key] = node_data[key]
    return safe


def _copy_admitted_hyperedges(
    source_graph: Any,
    target_graph: nx.MultiDiGraph,
    cutoff: Any,
) -> Tuple[set, Dict[str, Dict[str, Any]], set]:
    admitted_hyperedges: set = set()
    decisions: Dict[str, Dict[str, Any]] = {}
    admitted_occurrence_ids: set = set()
    seen_occurrence_ids: set = set()
    for node_id, data in sorted(source_graph.nodes(data=True), key=lambda item: str(item[0])):
        role = str(data.get("role", data.get("type", ""))).casefold()
        if role != "hyperedge" and not str(node_id).casefold().startswith("<hyperedge>"):
            continue
        occurrence_id = str(data.get("occurrence_id") or node_id)
        if occurrence_id in seen_occurrence_ids:
            raise ValueError(f"duplicate hyperedge occurrence identity: {occurrence_id}")
        seen_occurrence_ids.add(occurrence_id)
        for field in ("record_id", "chunk_id", "proposition_key"):
            if not str(data.get(field, "")).strip():
                raise ValueError(f"hyperedge {node_id!r} is missing {field}")
        decision = assess_evidence_admissibility(data, cutoff)
        decisions[str(node_id)] = decision.to_dict()
        if not decision.admitted:
            continue
        admitted_hyperedges.add(node_id)
        admitted_occurrence_ids.add(str(node_id))
        admitted_occurrence_ids.add(occurrence_id)
        attrs = dict(data)
        attrs["admissibility"] = decision.to_dict()
        target_graph.add_node(node_id, **attrs)
    return admitted_hyperedges, decisions, admitted_occurrence_ids


def build_active_logical_view(
    graph: Any,
    *,
    cutoff: Any,
    case_id: str,
    graph_bundle_sha256: str,
    admissibility_config_hash: str = "",
    scientific_config_sha256: str = "",
    construction_config_sha256: str = "",
) -> nx.MultiDiGraph:


    if not str(case_id).strip():
        raise ValueError("case_id is required for a cutoff-conditioned active view")
    if not str(graph_bundle_sha256).strip():
        raise ValueError("graph_bundle_sha256 is required for an active view")
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("validated graph bundles must load as nx.MultiDiGraph")
    if graph.graph.get("validated_graph_bundle") is not True:
        raise ValueError("active view requires a validated graph bundle")
    if str(graph.graph.get("graph_bundle_sha256", "")) != str(graph_bundle_sha256):
        raise ValueError("validated graph bundle identity mismatch")
    graph_construction_hash = str(
        graph.graph.get("construction_config_sha256")
        or graph.graph.get("scientific_config_sha256", "")
        or construction_config_sha256
        or scientific_config_sha256
        or graph_bundle_sha256
    ).strip()
    cutoff_value = parse_cutoff(cutoff)
    policy_hash = admissibility_config_hash or hashlib.sha256(
        canonical_json(
            {
                "policy": _ADMISSIBILITY_POLICY,
                "fallback_score": 0.25,
                "known_equal_or_post": "exclude",
                "missing_requires_safe_bound": True,
            }
        ).encode("utf-8")
    ).hexdigest()
    active_view_id = _stable_id(
        "view_",
        str(graph_bundle_sha256),
        str(case_id),
        cutoff_value.canonical,
        policy_hash,
    )
    active = nx.MultiDiGraph()
    active.graph.update(
        {
            "schema_version": "chain-active-logical-view-v1",
            "active_view_id": active_view_id,
            "case_id": str(case_id),
            "cutoff": cutoff_value.canonical,
            "graph_bundle_sha256": str(graph_bundle_sha256),
            "admissibility_config_hash": policy_hash,
            "construction_config_sha256": graph_construction_hash,
            "scientific_config_sha256": str(scientific_config_sha256),
            "validated_graph_bundle": True,
            "profile_id": str(graph.graph.get("profile_id", "")),
            "publishable": graph.graph.get("publishable") is True,
        }
    )

    admitted_hyperedges, hyperedge_decisions, admitted_occurrence_ids = (
        _copy_admitted_hyperedges(graph, active, cutoff)
    )
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    causal_decisions: Dict[str, Dict[str, Any]] = {}
    seen_occurrence_ids: set = set()
    sorted_edges = _sorted_keyed_edges(graph)
    for source, target, key, data in sorted_edges:
        if not _is_causal(data):
            continue
        occurrence_id = _occurrence_id(key, data)
        if occurrence_id in seen_occurrence_ids:
            raise ValueError(f"duplicate causal occurrence identity: {occurrence_id}")
        seen_occurrence_ids.add(occurrence_id)
        if source not in graph or graph.nodes[source].get("role") != "entity":
            raise ValueError(f"causal source is not a validated entity: {source!r}")
        if target not in graph or graph.nodes[target].get("role") != "entity":
            raise ValueError(f"causal target is not a validated entity: {target!r}")
        temporal_data = _occurrence_temporal_data(graph, data)
        if not str(temporal_data.get("record_id", "")).strip() or not str(
            temporal_data.get("chunk_id") or temporal_data.get("source_id") or ""
        ).strip():
            raise ValueError(
                f"causal occurrence lacks record/chunk provenance: {occurrence_id}"
            )
        decision = assess_evidence_admissibility(temporal_data, cutoff)
        causal_decisions[occurrence_id] = decision.to_dict()
        if not decision.admitted:
            continue
        relation_type = _causal_type(data)
        occurrence = {
            "source": str(source),
            "target": str(target),
            "key": key,
            "occurrence_id": occurrence_id,
            "strength": _edge_strength(data),
            "causal_type": relation_type,
            "description": str(data.get("description", "")),
            "provenance": temporal_data,
            "admissibility": decision.to_dict(),
        }
        groups.setdefault((str(source), str(target), relation_type), []).append(
            occurrence
        )
        admitted_occurrence_ids.add(occurrence_id)




    admitted_occurrence_ids = frozenset(admitted_occurrence_ids)

    active_entity_ids: set = set()
    for (source, target, relation_type), occurrences in sorted(groups.items()):
        occurrences.sort(key=lambda item: item["occurrence_id"])
        representative = min(
            occurrences,
            key=lambda item: (-item["strength"], item["key"]),
        )
        occurrence_ids = [item["occurrence_id"] for item in occurrences]
        logical_edge_id = _stable_id(
            "ledge_",
            active_view_id,
            source,
            target,
            relation_type,
            occurrence_ids,
        )
        active_entity_ids.update((source, target))
        active.add_edge(
            source,
            target,
            key=logical_edge_id,
            edge_id=logical_edge_id,
            logical_edge_id=logical_edge_id,
            role="causal",
            causal_type=relation_type,
            strength=representative["strength"],
            description=representative["description"],
            representative_occurrence_id=representative["occurrence_id"],
            representative_occurrence_key=representative["key"],
            admitted_occurrence_ids=occurrence_ids,
            admitted_occurrences=occurrences,
        )



    for source, target, key, data in sorted_edges:
        if _is_causal(data):
            continue
        if data.get("role") != "incidence":
            raise ValueError(f"unsupported non-causal edge role: {data.get('role')!r}")
        if source not in admitted_hyperedges:
            continue
        if target not in graph or graph.nodes[target].get("role") != "entity":
            raise ValueError("incidence target is not a validated entity")
        active_entity_ids.add(str(target))
        active.add_edge(source, target, key=key, **dict(data))

    for node_id in sorted(active_entity_ids, key=str):
        original_id: Any = node_id
        if original_id not in graph:

            original_id = next(
                (candidate for candidate in graph.nodes if str(candidate) == node_id),
                node_id,
            )
        if original_id not in graph:
            raise ValueError(f"causal endpoint missing from graph: {node_id!r}")
        active.add_node(
            node_id,
            **_safe_entity_data(
                node_id,
                graph.nodes[original_id],
                admitted_occurrence_ids,
                admitted_occurrence_ids_normalized=True,
            ),
        )

    active.graph["hyperedge_admissibility"] = hyperedge_decisions
    active.graph["causal_occurrence_admissibility"] = causal_decisions
    active.graph["admitted_occurrence_ids"] = sorted(admitted_occurrence_ids)
    return active


def unique_neighbor_count(graph: Any, node_id: Any) -> int:
    if node_id not in graph:
        return 0
    return len(set(graph.predecessors(node_id)) | set(graph.successors(node_id)))


def _has_causal_edge(graph: Any, source: Any, target: Any) -> bool:
    return any(_is_causal(data) for data in _edges_between(graph, source, target))


def compute_causal_distances(
    graph: Any,
    target_entities: List[str],
    entity_ids: List[str],
    max_depth: int = 4,
) -> Dict[str, int]:


    max_depth = _positive_int(max_depth, "d_max")
    distances: Dict[str, int] = {}
    for target in sorted(set(target_entities), key=str):
        if target not in graph:
            continue
        queue = deque([(target, 0)])
        visited = {target}
        while queue:
            current, distance = queue.popleft()
            if distance > max_depth:
                continue
            current_key = str(current)
            distances[current_key] = min(distances.get(current_key, math.inf), distance)
            if distance == max_depth:
                continue
            for predecessor in sorted(set(graph.predecessors(current)), key=str):
                if predecessor in visited or not _has_causal_edge(
                    graph, predecessor, current
                ):
                    continue
                visited.add(predecessor)
                queue.append((predecessor, distance + 1))
    for entity_id in entity_ids:
        distances.setdefault(str(entity_id), math.inf)

    entity_set = {str(entity_id) for entity_id in entity_ids}
    for node_id in sorted(graph.nodes, key=str):
        node_key = str(node_id)
        if node_key in distances:
            continue
        connected = [
            distances[str(neighbor)]
            for neighbor in (
                set(graph.predecessors(node_id)) | set(graph.successors(node_id))
            )
            if str(neighbor) in entity_set and str(neighbor) in distances
        ]
        if connected:
            distances[node_key] = min(connected)
    return distances


def _typed_edge_identity(edge: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(edge["cause_id"]),
        str(edge["effect_id"]),
        str(edge["causal_type"]),
    )


def _edge_logical_id(edge: Mapping[str, Any]) -> str:
    return str(
        edge.get("logical_edge_id")
        or edge.get("edge_key")
        or _stable_id(
            "compat_ledge_",
            str(edge.get("cause_id", "")),
            str(edge.get("effect_id", "")),
            str(edge.get("causal_type", "")),
        )
    )


def _typed_path_identity(chain: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(_edge_logical_id(edge) for edge in chain.get("edges", []))


def dedupe_chains_by_overlap(
    chains: List[Dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    threshold = float(overlap_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("theta must be in (0, 1)")
    sorted_chains = sorted(
        chains,
        key=lambda chain: (-float(chain.get("score", 0.0)), _typed_path_identity(chain)),
    )
    kept: List[Dict[str, Any]] = []
    edge_sets_by_direction: Dict[str, List[frozenset]] = {
        "supports": [],
        "opposes": [],
    }
    for chain in sorted_chains:
        direction = str(chain.get("direction", ""))
        if direction not in edge_sets_by_direction:
            raise ValueError(f"invalid chain direction: {direction!r}")
        edge_set = frozenset(_typed_edge_identity(edge) for edge in chain["edges"])
        if not edge_set:
            continue
        redundant = False
        for existing in edge_sets_by_direction[direction]:
            overlap = len(edge_set & existing) / len(edge_set | existing)
            if overlap > threshold:
                redundant = True
                break
        if not redundant:
            kept.append(chain)
            edge_sets_by_direction[direction].append(edge_set)
    return kept


def compute_chain_diversity(
    scored_chains: List[Dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> int:
    return len(dedupe_chains_by_overlap(scored_chains, overlap_threshold))


def score_causal_chain(
    chain_edges: List[Dict[str, Any]],
    tvf_scores: Optional[Dict[str, float]] = None,
    causal_distances: Optional[Dict[str, int]] = None,
    max_depth: int = 4,
) -> float:
    max_depth = _positive_int(max_depth, "d_max")
    if not chain_edges:
        return 0.0
    product = 1.0
    for edge in chain_edges:
        strength = _edge_strength(edge)
        if tvf_scores is None:
            edge_validity = 1.0
        else:
            cause_id = str(edge["cause_id"])
            effect_id = str(edge["effect_id"])
            mean_validity = 0.5 * (
                _finite_probability(
                    tvf_scores[cause_id], "cause CTVF"
                )
                + _finite_probability(
                    tvf_scores[effect_id], "effect CTVF"
                )
            )
            if causal_distances is None:
                rho_edge = 1.0
            else:
                distance_sum = float(
                    causal_distances.get(cause_id, math.inf)
                ) + float(causal_distances.get(effect_id, math.inf))
                rho_edge = min(max(distance_sum / (2.0 * max_depth), 0.0), 1.0)
            edge_validity = mean_validity ** rho_edge
        product *= strength * edge_validity
    return max(0.0, min(1.0, product))


def _logical_causal_edges(graph: Any) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for source, target, key, data in _iter_keyed_edges(graph):
        if not _is_causal(data):
            continue
        relation_type = _causal_type(data)
        logical_id = str(
            data.get("logical_edge_id")
            or data.get("edge_id")
            or _stable_id("compat_ledge_", str(source), str(target), relation_type, key)
        )
        edge = {
            "cause_id": str(source),
            "effect_id": str(target),
            "causal_type": relation_type,
            "strength": _edge_strength(data),
            "description": str(data.get("description", "")),
            "edge_key": key,
            "logical_edge_id": logical_id,
            "admitted_occurrence_ids": list(data.get("admitted_occurrence_ids", [])),
        }
        grouped.setdefault((str(source), str(target), relation_type), []).append(edge)
    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for typed_key, occurrences in sorted(grouped.items()):
        representative = min(
            occurrences,
            key=lambda edge: (-edge["strength"], edge["logical_edge_id"], edge["edge_key"]),
        )
        if len(occurrences) > 1:
            occurrence_ids = sorted(
                {
                    occurrence_id
                    for edge in occurrences
                    for occurrence_id in (
                        edge["admitted_occurrence_ids"] or [edge["logical_edge_id"]]
                    )
                }
            )
            representative = dict(representative)
            representative["admitted_occurrence_ids"] = occurrence_ids
            representative["logical_edge_id"] = _stable_id(
                "ledge_", typed_key[0], typed_key[1], typed_key[2], occurrence_ids
            )
        adjacency.setdefault(typed_key[0], []).append(representative)
    for source in adjacency:
        adjacency[source].sort(
            key=lambda edge: (
                -edge["strength"],
                edge["effect_id"],
                edge["causal_type"],
                edge["logical_edge_id"],
            )
        )
    return adjacency


def _clean_name(value: Any) -> str:
    text = re.sub(r"<[^>]+>", "", str(value))
    text = re.sub(
        r"\s*\|\s*(Domain|Source):.*$", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def extract_scored_chains(
    graph: Any,
    entity_ids: List[str],
    tvf_scores: Optional[Dict[str, float]] = None,
    causal_distances: Optional[Dict[str, int]] = None,
    max_depth: int = 4,
    max_chains: int = 200,
    target_signs: Optional[Dict[str, int]] = None,
    *,
    target_anchors: Optional[Dict[str, int]] = None,
    F_max: Optional[int] = None,
    tau_phi: float = 0.005,
    hard_cap: Optional[int] = None,
) -> List[Dict[str, Any]]:


    d_max = _positive_int(max_depth, "d_max")
    B_pi = _positive_int(max_chains, "B_pi")
    if F_max is not None:
        F_max = _positive_int(F_max, "F_max")
    tau = float(tau_phi)
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau_phi must be in [0, 1]")
    H_pi = max(5 * B_pi, 1000)
    if hard_cap is not None and hard_cap != H_pi:
        raise ValueError("H_pi is fixed to max(5*B_pi, 1000)")
    raw_targets = target_signs if target_signs is not None else target_anchors
    resolved_targets: Dict[str, int] = {}
    for target_id, raw_sign in (raw_targets or {}).items():
        if isinstance(raw_sign, bool) or raw_sign not in (-1, 1):
            continue
        if target_id in graph:
            resolved_targets[str(target_id)] = int(raw_sign)
    if not resolved_targets:
        return []

    adjacency = _logical_causal_edges(graph)
    if F_max is not None:
        adjacency = {
            source: edges[:F_max] for source, edges in adjacency.items()
        }
    accepted: List[Dict[str, Any]] = []
    seen_typed_paths: set = set()
    for start_id in sorted({str(item) for item in entity_ids}):



        if start_id in resolved_targets or start_id not in adjacency:
            continue
        queue = deque([(start_id, (start_id,), tuple(), tuple())])
        while queue and len(accepted) < H_pi:
            current, node_path, edge_path, edge_ids = queue.popleft()
            if current in resolved_targets:
                continue
            for edge in adjacency.get(current, []):
                next_id = edge["effect_id"]
                if next_id in node_path:
                    continue
                next_edge_ids = edge_ids + (edge["logical_edge_id"],)
                typed_path_key = (node_path + (next_id,), next_edge_ids)
                if typed_path_key in seen_typed_paths:
                    continue
                seen_typed_paths.add(typed_path_key)
                next_edges = edge_path + (edge,)
                prefix_score = score_causal_chain(
                    list(next_edges),
                    tvf_scores,
                    causal_distances,
                    max_depth=d_max,
                )
                if prefix_score < tau:
                    continue
                next_nodes = node_path + (next_id,)
                if next_id in resolved_targets:
                    prevents_count = sum(
                        1 for item in next_edges if item["causal_type"] == "prevents"
                    )
                    path_sign = -1 if prevents_count % 2 else 1
                    target_sign = resolved_targets[next_id]
                    direction = "supports" if target_sign * path_sign > 0 else "opposes"
                    accepted.append(
                        {
                            "chain": list(next_nodes),
                            "edges": [dict(item) for item in next_edges],
                            "score": prefix_score,
                            "direction": direction,
                            "target_id": next_id,
                            "target_sign": target_sign,
                            "prevents_count": prevents_count,
                            "entities": [
                                _clean_name(graph.nodes[node_id].get("name", node_id))
                                for node_id in next_nodes
                                if node_id in graph
                            ],
                            "depth": len(next_edges),
                            "typed_path_identity": list(next_edge_ids),
                        }
                    )
                    if len(accepted) >= H_pi:
                        break
                    continue
                if len(next_edges) < d_max:
                    queue.append((next_id, next_nodes, next_edges, next_edge_ids))
        if len(accepted) >= H_pi:
            break
    accepted.sort(
        key=lambda chain: (-chain["score"], _typed_path_identity(chain))
    )
    return accepted[:B_pi]


def noisy_or_aggregate(chain_scores: List[float]) -> float:
    complement = 1.0
    for score in chain_scores:
        complement *= 1.0 - _finite_probability(score, "chain score")
    return 1.0 - complement


def mean_aggregate(chain_scores: List[float]) -> float:
    if not chain_scores:
        return 0.0
    values = [_finite_probability(score, "chain score") for score in chain_scores]
    return sum(values) / len(values)


def _logit(probability: float, epsilon: float = 1e-6) -> float:
    clipped = max(epsilon, min(1.0 - epsilon, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def combine_directional_scores(
    supporting_score: float,
    opposing_score: float,
    eps: float = 1e-6,
) -> float:
    support = _finite_probability(supporting_score, "supporting channel")
    oppose = _finite_probability(opposing_score, "opposing channel")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5)")
    return _sigmoid(
        _logit((1.0 + support) / 2.0, epsilon)
        - _logit((1.0 + oppose) / 2.0, epsilon)
    )


def compute_causal_probability(
    scored_chains: List[Dict[str, Any]],
    *,
    theta: float = 0.5,
    epsilon: float = 1e-6,
    aggregation_mode: str = "noisy_or",
) -> Dict[str, Any]:
    supporting_all = [
        chain for chain in scored_chains if chain.get("direction") == "supports"
    ]
    opposing_all = [
        chain for chain in scored_chains if chain.get("direction") == "opposes"
    ]
    supporting = dedupe_chains_by_overlap(supporting_all, theta)
    opposing = dedupe_chains_by_overlap(opposing_all, theta)
    if aggregation_mode == "noisy_or":
        aggregate = noisy_or_aggregate
    elif aggregation_mode == "mean":
        aggregate = mean_aggregate
    else:
        raise ValueError(f"unsupported polarity aggregation: {aggregation_mode!r}")
    p_plus = aggregate([chain["score"] for chain in supporting])
    p_minus = aggregate([chain["score"] for chain in opposing])
    p_causal_event = combine_directional_scores(p_plus, p_minus, epsilon)
    return {
        "p_causal_event": p_causal_event,
        "p_complement_causal": 1.0 - p_causal_event,
        "p_plus": p_plus,
        "p_minus": p_minus,
        "n_supporting": len(supporting),
        "n_opposing": len(opposing),
        "n_supporting_raw": len(supporting_all),
        "n_opposing_raw": len(opposing_all),
        "aggregation_mode": aggregation_mode,
        "deduplicated_chains": supporting + opposing,
    }


def _causal_probability_from_pool(
    supporting: List[Dict[str, Any]],
    opposing: List[Dict[str, Any]],
    *,
    epsilon: float,
    aggregation_mode: str,
) -> float:
    aggregate = noisy_or_aggregate if aggregation_mode == "noisy_or" else mean_aggregate
    return combine_directional_scores(
        aggregate([chain["score"] for chain in supporting]),
        aggregate([chain["score"] for chain in opposing]),
        epsilon,
    )


def compute_attribution(
    scored_chains: List[Dict[str, Any]],
    top_k: int = 5,
    *,
    theta: float = 0.5,
    epsilon: float = 1e-6,
    aggregation_mode: str = "noisy_or",
) -> List[Dict[str, Any]]:
    pool = compute_causal_probability(
        scored_chains,
        theta=theta,
        epsilon=epsilon,
        aggregation_mode=aggregation_mode,
    )["deduplicated_chains"]
    supporting = [chain for chain in pool if chain["direction"] == "supports"]
    opposing = [chain for chain in pool if chain["direction"] == "opposes"]
    full = _causal_probability_from_pool(
        supporting, opposing, epsilon=epsilon, aggregation_mode=aggregation_mode
    )
    output: List[Dict[str, Any]] = []
    for chain in pool:
        support_without = [item for item in supporting if item is not chain]
        oppose_without = [item for item in opposing if item is not chain]
        counterfactual = _causal_probability_from_pool(
            support_without,
            oppose_without,
            epsilon=epsilon,
            aggregation_mode=aggregation_mode,
        )
        output.append(
            {
                "typed_path_identity": list(_typed_path_identity(chain)),
                "chain_score": chain["score"],
                "direction": chain["direction"],
                "contribution": full - counterfactual,
                "p_causal_event": full,
                "p_causal_event_without": counterfactual,
                "entities": list(chain.get("entities", [])),
                "depth": chain["depth"],
            }
        )
    output.sort(
        key=lambda item: (-abs(item["contribution"]), tuple(item["typed_path_identity"]))
    )
    return output[:_positive_int(top_k, "attribution top_k")]


def counterfactual_analysis(
    scored_chains: List[Dict[str, Any]],
    top_k: int = 3,
    *,
    theta: float = 0.5,
    epsilon: float = 1e-6,
    aggregation_mode: str = "noisy_or",
) -> List[Dict[str, Any]]:
    pool = compute_causal_probability(
        scored_chains,
        theta=theta,
        epsilon=epsilon,
        aggregation_mode=aggregation_mode,
    )["deduplicated_chains"]
    supporting = [chain for chain in pool if chain["direction"] == "supports"]
    opposing = [chain for chain in pool if chain["direction"] == "opposes"]
    full = _causal_probability_from_pool(
        supporting, opposing, epsilon=epsilon, aggregation_mode=aggregation_mode
    )
    edge_by_id: Dict[str, Dict[str, Any]] = {}
    for chain in pool:
        for edge in chain["edges"]:
            edge_by_id.setdefault(_edge_logical_id(edge), edge)
    output: List[Dict[str, Any]] = []
    for logical_edge_id, edge in sorted(edge_by_id.items()):
        support_without = [
            chain
            for chain in supporting
            if all(
                _edge_logical_id(item) != logical_edge_id
                for item in chain["edges"]
            )
        ]
        oppose_without = [
            chain
            for chain in opposing
            if all(
                _edge_logical_id(item) != logical_edge_id
                for item in chain["edges"]
            )
        ]
        counterfactual = _causal_probability_from_pool(
            support_without,
            oppose_without,
            epsilon=epsilon,
            aggregation_mode=aggregation_mode,
        )
        output.append(
            {
                "removed_logical_edge_id": logical_edge_id,
                "removed_edge": {
                    "cause_id": edge["cause_id"],
                    "effect_id": edge["effect_id"],
                    "causal_type": edge["causal_type"],
                },
                "removed_edge_label": (
                    f"{edge['cause_id']} \u2192 {edge['effect_id']} "
                    f"({edge['causal_type']})"
                ),
                "edge_strength": edge["strength"],
                "p_causal_event": full,
                "p_causal_event_counterfactual": counterfactual,
                "delta_p": full - counterfactual,
            }
        )
    output.sort(
        key=lambda item: (-abs(item["delta_p"]), item["removed_logical_edge_id"])
    )
    return output[:_positive_int(top_k, "counterfactual top_k")]


def aggregate_predictions(
    p_llm: Optional[float] = None,
    p_causal: Optional[float] = None,
    alpha_base: float = 0.5,
    n_causal_chains: int = 0,
    chain_diversity: int = 0,
    avg_chain_ctvf: float = 0.0,
    adaptive: bool = True,
    n_supporting: int = 0,
    n_opposing: int = 0,
    *,
    p_llm_event: Optional[float] = None,
    p_causal_event: Optional[float] = None,
    alpha_b: Optional[float] = None,
    alpha_0: float = 0.6,
    k_sat: int = 10,
    beta_0: float = 0.3,
    Z: float = 5.0,
    Omega_0: float = 0.3,
    zeta: float = 3.0,
    fusion_mode: Optional[str] = None,
    fixed_alpha: Optional[float] = None,
) -> Dict[str, Any]:


    llm = _finite_probability(
        p_llm_event if p_llm_event is not None else p_llm, "p_llm_event"
    )
    causal = _finite_probability(
        p_causal_event if p_causal_event is not None else p_causal,
        "p_causal_event",
    )
    alpha_b_value = float(alpha_base if alpha_b is None else alpha_b)
    alpha_0_value = float(alpha_0)
    beta_floor = float(beta_0)
    normalizer = float(Z)
    threshold = float(Omega_0)
    sharpness = float(zeta)
    if not math.isfinite(alpha_b_value) or alpha_b_value <= 0.0:
        raise ValueError("alpha_b must be finite and > 0")
    if not math.isfinite(alpha_0_value) or not 0.0 < alpha_0_value <= 1.0:
        raise ValueError("alpha_0 must be finite and in (0, 1]")
    if not math.isfinite(beta_floor) or not 0.0 < beta_floor < 1.0:
        raise ValueError("beta_0 must be finite and in (0, 1)")
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("Z must be finite and > 0")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Omega_0 must be finite and in [0, 1]")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("zeta must be finite and > 0")
    saturation = _positive_int(k_sat, "k_sat")
    k = _nonnegative_int(
        n_causal_chains if n_causal_chains is not None else chain_diversity,
        "number of causal chains",
    )
    supporting = _nonnegative_int(n_supporting, "n_supporting")
    opposing = _nonnegative_int(n_opposing, "n_opposing")
    if supporting + opposing != k:
        raise ValueError(
            "n_supporting + n_opposing must equal number of causal chains"
        )
    mean_score = 0.0 if k == 0 else _finite_probability(
        avg_chain_ctvf, "mean retained chain score"
    )
    reliability = (
        0.0
        if k == 0
        else min(math.log1p(k) / math.log1p(saturation), 1.0)
    )
    balance = min(supporting, opposing) / max(
        supporting, opposing, 1
    )
    floored_balance = beta_floor + (1.0 - beta_floor) * balance
    coverage = min(
        k * mean_score * reliability * floored_balance / normalizer, 1.0
    )
    mode = fusion_mode or ("adaptive" if adaptive else "fixed")
    if k == 0:
        alpha = 0.0
    elif mode == "adaptive":
        alpha = min(
            2.0 * alpha_b_value * _sigmoid(sharpness * (coverage - threshold)),
            alpha_0_value,
        )
    elif mode == "without_adaptive_alpha":



        alpha = 0.5
    elif mode == "fixed":
        alpha = _finite_probability(
            alpha_b_value if fixed_alpha is None else fixed_alpha, "fixed_alpha"
        )
    else:
        raise ValueError(f"unsupported fusion mode: {mode!r}")
    final = alpha * causal + (1.0 - alpha) * llm
    event_side = final >= 0.5
    answer = "event" if event_side else "complement"
    confidence = final if event_side else 1.0 - final
    return {
        "p_llm_event": llm,
        "p_causal_event": causal,
        "p_final_event": final,
        "p_event": final,
        "alpha": alpha,
        "coverage": coverage,
        "reliability": reliability,
        "balance": balance,
        "floored_balance": floored_balance,
        "mean_chain_score": mean_score,
        "chain_diversity": k,
        "n_supporting": supporting,
        "n_opposing": opposing,
        "fusion_mode": mode,
        "answer": answer,
        "confidence": confidence,

        "p_llm": llm,
        "p_causal": causal,
        "p_final": final,
        "alpha_used": alpha,
    }


def _axis_positive_is_yes(binary_axis: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(binary_axis, Mapping):
        return False
    positive = binary_axis.get("positive") or binary_axis.get("event")
    if not isinstance(positive, Mapping):
        return False
    label = str(
        positive.get("label")
        or positive.get("name")
        or positive.get("value")
        or positive.get("id")
        or ""
    ).strip().casefold()
    positive_id = str(positive.get("id", "")).strip().casefold()
    negative = binary_axis.get("negative") or binary_axis.get("complement")
    negative_id = (
        str(negative.get("id", "")).strip().casefold()
        if isinstance(negative, Mapping)
        else ""
    )
    negative_label = (
        str(negative.get("label", "")).strip().casefold()
        if isinstance(negative, Mapping)
        else ""
    )
    legacy_adapter = (
        str(binary_axis.get("adapter_id", ""))
        == "chain-legacy-native-yes-no-v1"
    )
    return (
        positive_id == "yes"
        and label == "yes"
        and negative_id == "no"
        and negative_label == "no"
        and (legacy_adapter or str(binary_axis.get("source", "")) != "legacy_adapter")
    )


def _axis_label(binary_axis: Mapping[str, Any], positive: bool) -> str:
    side = binary_axis["positive" if positive else "negative"]
    return str(side["label"]).strip()


def run_causal_inference(
    graph: Any,
    entity_ids: List[str],
    p_llm_yes: Optional[float] = None,
    tvf_scores: Optional[Dict[str, float]] = None,
    alpha: float = 0.5,
    max_depth: int = 4,
    max_chains: int = 200,
    target_entities: Optional[List[str]] = None,
    adaptive_alpha: bool = True,
    target_signs: Optional[Dict[str, int]] = None,
    *,
    p_llm_event: Optional[float] = None,
    cutoff_date: Any = "",
    case_id: str = "",
    graph_bundle_sha256: str = "",
    inference_config: Any = None,
    binary_axis: Optional[Mapping[str, Any]] = None,
    precomputed: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:


    if not isinstance(inference_config, ResolvedConfig):
        raise ValueError("causal inference requires one ResolvedConfig")
    required_config_fields = (
        "d_max",
        "B_pi",
        "F_max",
        "tau_phi",
        "theta",
        "eta",
        "k_sat",
        "beta_0",
        "Z",
        "alpha_b",
        "alpha_0",
        "Omega_0",
        "zeta",
        "epsilon",
        "fusion_mode",
        "fixed_alpha",
        "ablation_mode",
        "attribution_top_k",
        "counterfactual_top_k",
        "displayed_top_chain_cap",
    )
    missing_sentinel = object()
    missing_config_fields = [
        field
        for field in required_config_fields
        if _config_value(inference_config, (field,), missing_sentinel)
        is missing_sentinel
    ]
    if missing_config_fields:
        raise ValueError(
            "resolved inference configuration is missing scientific fields: "
            + ", ".join(sorted(missing_config_fields))
        )
    profile = _profile_name(inference_config)
    paper = profile in {_PAPER_PROFILE, "paper"}
    if not cutoff_date:
        raise ValueError("causal inference requires a per-case cutoff")
    if not str(case_id).strip():
        raise ValueError("causal inference requires case_id")
    if not isinstance(binary_axis, Mapping):
        raise ValueError("causal inference requires canonical binary_axis")
    canonical_axis = validate_binary_axis(binary_axis)
    axis_hash = str(canonical_axis.get("binary_axis_sha256", "")).strip()
    if not axis_hash:
        raise ValueError("canonical binary_axis is missing binary_axis_sha256")
    config_hash = str(
        _config_value(inference_config, ("scientific_config_sha256",), "")
    ).strip()
    if not config_hash:
        raise ValueError("resolved configuration is missing scientific identity")
    config_profile_id = str(
        _config_value(inference_config, ("profile_id",), profile)
    ).strip()
    bundle_hash = str(
        graph_bundle_sha256
        or getattr(graph, "graph", {}).get("graph_bundle_sha256", "")
        or getattr(graph, "graph", {}).get("bundle_sha256", "")
    )
    if not bundle_hash:
        raise ValueError("causal inference requires graph_bundle_sha256")
    graph_metadata = getattr(graph, "graph", {})
    if graph_metadata.get("validated_graph_bundle") is not True:
        raise ValueError("causal inference requires a validated graph bundle")
    graph_construction_hash = str(
        graph_metadata.get("construction_config_sha256")
        or graph_metadata.get("scientific_config_sha256", "")
        or _config_value(
            inference_config, ("construction_config_sha256",), ""
        )
        or config_hash
        or bundle_hash
    ).strip()

    d_max = int(_config_value(inference_config, ("d_max",)))
    B_pi = int(_config_value(inference_config, ("B_pi", "b_pi")))
    F_max = _config_value(inference_config, ("F_max", "f_max"))
    tau_phi = float(_config_value(inference_config, ("tau_phi",)))
    theta = float(_config_value(inference_config, ("theta",)))
    eta = float(_config_value(inference_config, ("eta",)))
    k_sat = int(_config_value(inference_config, ("k_sat",)))
    beta_0 = float(_config_value(inference_config, ("beta_0",)))
    Z = float(_config_value(inference_config, ("Z", "coverage_normalizer")))
    alpha_b = float(_config_value(inference_config, ("alpha_b",)))
    alpha_0 = float(_config_value(inference_config, ("alpha_0",)))
    Omega_0 = float(_config_value(inference_config, ("Omega_0", "omega_0")))
    zeta = float(_config_value(inference_config, ("zeta",)))
    epsilon = float(_config_value(inference_config, ("epsilon",)))
    fusion_mode = str(_config_value(inference_config, ("fusion_mode",)))
    fixed_alpha = _config_value(inference_config, ("fixed_alpha",), None)
    ablation_mode = str(
        _config_value(inference_config, ("ablation_mode",), "none") or "none"
    )

    llm_probability = _finite_probability(
        p_llm_event if p_llm_event is not None else p_llm_yes, "p_llm_event"
    )
    effective_targets = dict(target_signs or {})
    dropped_targets: List[Dict[str, str]] = []
    if not effective_targets and isinstance(target_entities, Mapping):
        effective_targets = dict(target_entities)
    elif not effective_targets and target_entities:


        dropped_targets.extend(
            {"target_id": str(target_id), "reason": "missing_orientation"}
            for target_id in target_entities
        )
        effective_targets = {}
    validated_targets: Dict[str, int] = {}
    for target_id, sign in sorted(effective_targets.items(), key=lambda item: str(item[0])):
        target_key = str(target_id)
        if isinstance(sign, bool) or sign not in (-1, 1):
            dropped_targets.append(
                {"target_id": target_key, "reason": "invalid_orientation"}
            )
        elif target_id not in graph:
            dropped_targets.append(
                {"target_id": target_key, "reason": "missing_from_validated_graph"}
            )
        else:
            validated_targets[target_key] = int(sign)
    effective_targets = validated_targets

    active_graph = graph
    already_active = (
        getattr(graph, "graph", {}).get("schema_version")
        == "chain-active-logical-view-v1"
    )
    if cutoff_date and already_active:
        requested_cutoff = parse_cutoff(cutoff_date).canonical
        if str(graph.graph.get("case_id", "")) != str(case_id):
            raise ValueError("active view case_id does not match inference case")
        if str(graph.graph.get("cutoff", "")) != requested_cutoff:
            raise ValueError("active view cutoff does not match inference cutoff")
        if str(graph.graph.get("graph_bundle_sha256", "")) != bundle_hash:
            raise ValueError("active view graph bundle identity mismatch")
        bundle_hash = str(graph.graph.get("graph_bundle_sha256", bundle_hash))
    elif cutoff_date:
        active_graph = build_active_logical_view(
            graph,
            cutoff=cutoff_date,
            case_id=str(case_id),
            graph_bundle_sha256=bundle_hash,
            scientific_config_sha256=config_hash,
        )
    active_entities = sorted(
        str(node_id)
        for node_id, data in active_graph.nodes(data=True)
        if data.get("role") == "entity"
    )
    active_targets: Dict[str, int] = {}
    for target_id, sign in effective_targets.items():
        if target_id not in active_graph or target_id not in active_entities:
            dropped_targets.append(
                {"target_id": target_id, "reason": "no_admitted_pre_cutoff_support"}
            )
        else:
            active_targets[target_id] = sign
    effective_targets = active_targets




    infer_tvf = ablation_mode == "infer_tvf"
    neutral_validity = ablation_mode in {
        "without_ctvf",
        "without_tvf",
        "without_all",
        "w/o CTVF",
        "w_o_ctvf",
    }
    cached = None
    if precomputed is not None:
        cached = _validated_causal_precompute(
            precomputed,
            graph=active_graph,
            case_id=str(case_id),
            cutoff=cutoff_date,
            graph_bundle_sha256=bundle_hash,
            config=inference_config,
            target_signs=effective_targets,
            tvf_scores=tvf_scores,
        )
    causal_distances = (
        dict(cached["causal_distances"])
        if cached is not None
        else (
            compute_causal_distances(
                active_graph, list(effective_targets), active_entities, d_max
            )
            if effective_targets
            else {entity_id: math.inf for entity_id in active_entities}
        )
    )
    node_tvf_scores: Dict[str, float] = dict(tvf_scores or {})
    ctvf_decisions: Dict[str, Any] = {}
    n_q: Dict[str, int] = {}
    if cached is not None:
        node_tvf_scores = dict(cached["node_tvf_scores"])
        ctvf_decisions = dict(cached["ctvf_decisions"])
        n_q = dict(cached["n_q"])
    elif cutoff_date:
        scorer = TemporalValidityScorer(eta=eta, d_max=d_max)
        hyperedges = [
            (str(node_id), dict(data))
            for node_id, data in active_graph.nodes(data=True)
            if data.get("role") == "hyperedge"
        ]
        scored, decisions, n_q = scorer.score_nodes_with_decisions(
            hyperedges,
            cutoff_date,
            None if infer_tvf else causal_distances,
            d_max,
            strict_support_ledger=True,
        )
        node_tvf_scores.update({node_id: score for node_id, _, score in scored})
        ctvf_decisions = {
            node_id: decision.to_dict() for node_id, decision in decisions.items()
        }
        hyperedge_ids = {node_id for node_id, _, _ in scored}
        for entity_id in active_entities:
            adjacent = sorted(
                {
                    str(neighbor)
                    for neighbor in (
                        set(active_graph.predecessors(entity_id))
                        | set(active_graph.successors(entity_id))
                    )
                    if str(neighbor) in hyperedge_ids
                }
            )
            values = [node_tvf_scores[node_id] for node_id in adjacent]
            node_tvf_scores[entity_id] = (
                sum(values) / len(values) if values else 0.5
            )
    if neutral_validity:
        node_tvf_scores = {node_id: 1.0 for node_id in active_graph.nodes}

    aggregation_mode = (
        "mean"
        if ablation_mode in {
            "without_noisy_or",
            "without_all",
            "w/o Noisy-OR",
            "w_o_noisy_or",
        }
        else "noisy_or"
    )
    if ablation_mode in {
        "without_adaptive_alpha",
        "w/o Adaptive alpha",
        "w_o_adaptive_alpha",
    }:
        fusion_mode = "without_adaptive_alpha"
        fixed_alpha = 0.5
    elif ablation_mode == "without_all":
        fusion_mode = "fixed"
        fixed_alpha = 0.0

    scored_chains = extract_scored_chains(
        active_graph,
        active_entities,
        tvf_scores=node_tvf_scores or None,
        causal_distances=None if infer_tvf else causal_distances,
        max_depth=d_max,
        max_chains=B_pi,
        target_signs=effective_targets,
        F_max=F_max,
        tau_phi=tau_phi,
    )
    causal_probability = compute_causal_probability(
        scored_chains,
        theta=theta,
        epsilon=epsilon,
        aggregation_mode=aggregation_mode,
    )
    deduplicated = causal_probability.pop("deduplicated_chains")
    k = len(deduplicated)
    mean_score = sum(chain["score"] for chain in deduplicated) / k if k else 0.0
    prediction = aggregate_predictions(
        p_llm_event=llm_probability,
        p_causal_event=causal_probability["p_causal_event"],
        n_causal_chains=k,
        chain_diversity=k,
        avg_chain_ctvf=mean_score,
        n_supporting=causal_probability["n_supporting"],
        n_opposing=causal_probability["n_opposing"],
        alpha_b=alpha_b,
        alpha_0=alpha_0,
        k_sat=k_sat,
        beta_0=beta_0,
        Z=Z,
        Omega_0=Omega_0,
        zeta=zeta,
        fusion_mode=fusion_mode,
        fixed_alpha=fixed_alpha,
    )
    attribution_cap = int(
        _config_value(inference_config, ("attribution_top_k",), 5)
    )
    counterfactual_cap = int(
        _config_value(inference_config, ("counterfactual_top_k",), 3)
    )
    top_chain_cap = int(
        _config_value(inference_config, ("displayed_top_chain_cap",), 8)
    )
    attribution = compute_attribution(
        scored_chains,
        top_k=attribution_cap,
        theta=theta,
        epsilon=epsilon,
        aggregation_mode=aggregation_mode,
    )
    counterfactual = counterfactual_analysis(
        scored_chains,
        top_k=counterfactual_cap,
        theta=theta,
        epsilon=epsilon,
        aggregation_mode=aggregation_mode,
    )
    prediction["answer"] = _axis_label(
        canonical_axis, prediction["p_event"] >= 0.5
    )
    inference_identity = _stable_id(
        "infer_",
        str(active_graph.graph.get("active_view_id", "")),
        config_hash,
        axis_hash,
        sorted(effective_targets.items()),
    )
    config_publishable = bool(
        _config_value(inference_config, ("publishable",), False)
    )
    graph_publishable = graph_metadata.get("publishable") is True
    publishable = bool(paper and config_publishable and graph_publishable)
    result: Dict[str, Any] = {
        **{
            key: prediction[key]
            for key in (
                "p_llm_event",
                "p_causal_event",
                "p_final_event",
                "p_event",
                "alpha",
            )
        },
        "prediction": prediction,
        "causal_probability": causal_probability,
        "n_chains": k,
        "n_chains_raw": len(scored_chains),
        "chain_diversity": k,
        "avg_chain_ctvf": mean_score,




        "scored_chains": copy.deepcopy(scored_chains),
        "deduplicated_chains": copy.deepcopy(deduplicated),
        "top_chains": deduplicated[:top_chain_cap],
        "attribution": attribution,
        "counterfactual": counterfactual,
        "target_signs": effective_targets,
        "dropped_targets": sorted(
            dropped_targets, key=lambda item: (item["target_id"], item["reason"])
        ),
        "causal_status": "valid" if k else (
            "valid_no_target" if not effective_targets else "valid_no_chain"
        ),
        "active_view_id": str(active_graph.graph.get("active_view_id", "")),
        "graph_bundle_sha256": bundle_hash,
        "case_id": str(case_id),
        "binary_axis_sha256": axis_hash,
        "construction_config_sha256": graph_construction_hash,
        "scientific_config_sha256": config_hash,
        "inference_identity": inference_identity,
        "cutoff": str(active_graph.graph.get("cutoff", cutoff_date or "")),
        "ctvf_decisions": ctvf_decisions,
        "n_q": n_q,
        "resolved_inference": {
            "d_max": d_max,
            "B_pi": B_pi,
            "F_max": F_max,
            "tau_phi": tau_phi,
            "theta": theta,
            "eta": eta,
            "k_sat": k_sat,
            "beta_0": beta_0,
            "Z": Z,
            "alpha_b": alpha_b,
            "alpha_0": alpha_0,
            "Omega_0": Omega_0,
            "zeta": zeta,
            "epsilon": epsilon,
            "fusion_mode": fusion_mode,
            "fixed_alpha": fixed_alpha,
            "ablation_mode": ablation_mode,
            "polarity_aggregation": aggregation_mode,
        },
        "profile": _PAPER_PROFILE if paper else (profile or "compat"),
        "profile_id": config_profile_id,
        "publishable": publishable,
    }
    if _axis_positive_is_yes(canonical_axis):
        result["p_yes"] = result["p_event"]
        prediction["p_yes"] = prediction["p_event"]
    canonical_json(result)
    return result


__all__ = [
    "aggregate_predictions",
    "build_active_logical_view",
    "combine_directional_scores",
    "compute_attribution",
    "compute_causal_distances",
    "compute_causal_probability",
    "compute_chain_diversity",
    "counterfactual_analysis",
    "dedupe_chains_by_overlap",
    "extract_scored_chains",
    "mean_aggregate",
    "noisy_or_aggregate",
    "run_causal_inference",
    "score_causal_chain",
    "unique_neighbor_count",
]
