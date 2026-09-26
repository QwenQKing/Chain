from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

from chain.config import (
    ResolvedConfig,
    canonical_json,
    validate_binary_axis,
)
from chain.graph.hypergraph import MMDTHypergraph
from chain.graph.reasoning_graph import ReasoningGraph
from chain.llm import chat_json, cosine_sim, embed_texts
from chain.skills.causal_inference import (
    build_active_logical_view,
    compute_causal_distances,
    extract_scored_chains,
    run_causal_inference,
    unique_neighbor_count,
)
from chain.skills.temporal_validity import TemporalValidityScorer, parse_cutoff


logger = logging.getLogger(__name__)

_ACTIVE_REASONING_PROMPT_VERSION = "chain-reasoning-prompt-v3"
_OPERATIONAL_RUNTIME_TELEMETRY_SCHEMA = "chain-operational-runtime-telemetry-v1"




_COSINE_BOUND_TOLERANCE = 1e-6
_CAUSAL_TYPES = {"causes", "enables", "prevents"}
_POSITIVE_WORDS = {
    "accept",
    "advance",
    "agree",
    "ceasefire",
    "continue",
    "deal",
    "gain",
    "improve",
    "increase",
    "lead",
    "peace",
    "progress",
    "reach",
    "resume",
    "sign",
    "win",
}
_NEGATIVE_WORDS = {
    "attack",
    "collapse",
    "decline",
    "fail",
    "hostile",
    "injury",
    "loss",
    "miss",
    "reject",
    "retire",
    "sanction",
    "stall",
    "suspend",
    "war",
    "withdraw",
}


def _runtime_spans(rg: Optional[ReasoningGraph]) -> List[Dict[str, Any]]:


    if rg is None:
        return []
    spans = getattr(rg, "_operational_runtime_spans", None)
    if spans is None:
        spans = []
        rg._operational_runtime_spans = spans
    if not isinstance(spans, list):
        raise TypeError("reasoning runtime span sink must be a list")
    return spans


def _append_runtime_span(
    spans: List[Dict[str, Any]],
    *,
    stage: str,
    round_num: int,
    started: float,
    status: str,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    spans.append(
        {
            "stage": str(stage),
            "elapsed_seconds": float(elapsed),
            "status": str(status),
            "round": int(round_num),
        }
    )


@contextmanager
def _timed_runtime_span(spans: List[Dict[str, Any]], stage: str, round_num: int):
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        _append_runtime_span(
            spans,
            stage=stage,
            round_num=round_num,
            started=started,
            status="error",
        )
        raise
    else:
        _append_runtime_span(
            spans,
            stage=stage,
            round_num=round_num,
            started=started,
            status="ok",
        )


def _operational_runtime_telemetry(rg: ReasoningGraph) -> Dict[str, Any]:
    spans = copy.deepcopy(_runtime_spans(rg))
    stage_totals: Dict[str, Dict[str, Any]] = {}
    for span in spans:
        stage = str(span["stage"])
        row = stage_totals.setdefault(
            stage,
            {
                "count": 0,
                "elapsed_seconds": 0.0,
                "status_counts": {"ok": 0, "error": 0},
            },
        )
        row["count"] += 1
        row["elapsed_seconds"] += float(span["elapsed_seconds"])
        row["status_counts"][str(span["status"])] += 1
    return {
        "schema_version": _OPERATIONAL_RUNTIME_TELEMETRY_SCHEMA,
        "spans": spans,
        "stage_totals": {stage: stage_totals[stage] for stage in sorted(stage_totals)},
    }


def _canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.strip().split())


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = canonical_json(list(parts)).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()


def _norm_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", _canonical_text(value))


def _name_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token
        for token in re.split(r"[^a-z0-9]+", _canonical_text(value))
        if len(token) >= 3
    )


def _names_match(left: Any, right: Any) -> bool:
    left_text = str(left or "")
    right_text = str(right or "")
    if re.search(r"\d", left_text) or re.search(r"\d", right_text):
        return _canonical_text(left_text) == _canonical_text(right_text)
    if _norm_name(left_text) == _norm_name(right_text):
        return True
    left_tokens = _name_tokens(left_text)
    right_tokens = _name_tokens(right_text)
    if not left_tokens or not right_tokens:
        return False
    intersection = len(left_tokens & right_tokens)
    return intersection >= 2 and intersection / len(left_tokens | right_tokens) >= 0.6


def _clean_node_label(raw: Any) -> str:
    value = re.sub(r"<[^>]+>", "", str(raw or ""))
    value = re.sub(r"\s*\|\s*(?:Domain|Source):.*$", "", value, flags=re.IGNORECASE)
    return " ".join(value.strip().split())


def _entity_label(graph: nx.MultiDiGraph, node_id: str) -> str:
    data = graph.nodes[node_id]
    return _clean_node_label(
        data.get("display_alias")
        or data.get("canonical_entity_key")
        or data.get("name")
        or node_id
    )


def _entity_search_texts(graph: nx.MultiDiGraph, node_id: str) -> List[str]:
    data = graph.nodes[node_id]
    values = [
        str(data.get("canonical_entity_key", "")),
        str(data.get("display_alias", "")),
        str(data.get("name", "")),
    ]
    aliases = data.get("admitted_aliases", [])
    if isinstance(aliases, (list, tuple)):
        values.extend(str(item) for item in aliases)
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        cleaned = _clean_node_label(value)
        key = _canonical_text(cleaned)
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result or [str(node_id)]


def _graph_entity_label_summary(
    graph: nx.MultiDiGraph,
    question: str,
    max_entities: Optional[int] = 80,
    max_chars: Optional[int] = 6000,
) -> str:

    if max_entities is not None:
        if isinstance(max_entities, bool) or not isinstance(max_entities, int):
            raise TypeError("max_entities must be an integer or None")
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")
    if max_chars is not None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise TypeError("max_chars must be an integer or None")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")

    query_tokens = _name_tokens(question)
    ranked: List[Tuple[int, int, str, str, str]] = []
    for node_id, data in graph.nodes(data=True):
        if data.get("role") != "entity":
            continue
        label = _entity_label(graph, str(node_id))
        if not label:
            continue
        overlap = len(query_tokens & _name_tokens(label))
        ranked.append(
            (
                overlap,
                unique_neighbor_count(graph, node_id),
                label.casefold(),
                str(node_id),
                label,
            )
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    labels: List[str] = []
    seen: Set[str] = set()
    for _, _, _, _, label in ranked:
        key = _canonical_text(label)
        if not key or key in seen:
            continue




        if max_chars is not None:
            candidate = canonical_json(labels + [label])
            if len(candidate) > max_chars:
                continue
        labels.append(label)
        seen.add(key)
        if max_entities is not None and len(labels) >= max_entities:
            break
    return canonical_json(labels)


def _require_resolved_config(
    config: Optional[ResolvedConfig],
    entrypoint: str,
) -> ResolvedConfig:
    if not isinstance(config, ResolvedConfig):
        raise ValueError(f"{entrypoint} requires one ResolvedConfig")
    if config.reasoning_prompt_version != _ACTIVE_REASONING_PROMPT_VERSION:
        raise ValueError(
            f"{entrypoint} reasoning prompt version mismatch: "
            f"resolved={config.reasoning_prompt_version!r}, "
            f"active={_ACTIVE_REASONING_PROMPT_VERSION!r}"
        )
    return config


def _require_active_reasoning_identity(
    rg: ReasoningGraph,
    *,
    question: str,
    cutoff_date: str,
    case_id: str,
    binary_axis: Mapping[str, Any],
    inference_config: ResolvedConfig,
    entrypoint: str,
) -> nx.MultiDiGraph:


    if rg.case_id != str(case_id):
        raise ValueError(f"{entrypoint} case identity mismatch")
    expected_question = _stable_id("query_", _canonical_text(question))
    if rg.question_identity != expected_question:
        raise ValueError(f"{entrypoint} question identity mismatch")
    if getattr(rg, "_binary_axis_sha256", "") != binary_axis["binary_axis_sha256"]:
        raise ValueError(f"{entrypoint} binary-axis identity mismatch")
    active_graph = getattr(rg, "_active_fact_graph", None)
    if not isinstance(active_graph, nx.MultiDiGraph):
        raise ValueError(f"{entrypoint} lost its active fact view")
    metadata = active_graph.graph
    if metadata.get("schema_version") != "chain-active-logical-view-v1":
        raise ValueError(f"{entrypoint} active-view schema mismatch")
    if metadata.get("validated_graph_bundle") is not True:
        raise ValueError(f"{entrypoint} active view is not validated")

    requested_cutoff = parse_cutoff(cutoff_date).canonical
    if str(metadata.get("cutoff", "")) != requested_cutoff:
        raise ValueError(f"{entrypoint} cutoff identity mismatch")
    if str(getattr(rg, "_cutoff", "")) != requested_cutoff:
        raise ValueError(f"{entrypoint} cached cutoff identity mismatch")
    if str(metadata.get("case_id", "")) != str(case_id):
        raise ValueError(f"{entrypoint} active-view case identity mismatch")
    if str(metadata.get("graph_bundle_sha256", "")) != rg.graph_bundle_sha256:
        raise ValueError(f"{entrypoint} graph-bundle identity mismatch")
    policy_hash = str(metadata.get("admissibility_config_hash", ""))
    if not policy_hash:
        raise ValueError(f"{entrypoint} active view lacks its policy identity")
    expected_view_id = _stable_id(
        "view_",
        rg.graph_bundle_sha256,
        str(case_id),
        requested_cutoff,
        policy_hash,
    )
    if str(metadata.get("active_view_id", "")) != expected_view_id:
        raise ValueError(f"{entrypoint} active-view identity mismatch")
    return active_graph


def _parse_yes_orientation(value: Any) -> Optional[int]:


    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value in (-1, 1):
        return int(value)
    raw = _canonical_text(value)
    if raw in {"1", "+1"}:
        return 1
    if raw == "-1":
        return -1
    normalized = raw.replace("_", " ").replace("-", " ").strip()
    if normalized in {"yes", "supports yes", "positive"}:
        return 1
    if normalized in {"no", "supports no", "negative"}:
        return -1
    return None


def _signature_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError("embedding signature must be a JSON object")


_EMBEDDING_SPACE_FIELDS = (
    "model",
    "revision",
    "dimension",
    "dtype",
    "serialization_precision",
    "preprocess_version",
    "normalization",
    "distance",
)


def _embedding_space_signature(value: Any) -> Dict[str, Any]:


    signature = _signature_object(value)
    return {field: signature.get(field) for field in _EMBEDDING_SPACE_FIELDS}


def _validate_embedding_boundary(
    graph: nx.MultiDiGraph,
    hypergraph: Any,
    config: ResolvedConfig,
) -> None:


    graph_signature = _signature_object(graph.graph.get("embedding_signature"))
    expected = config.embedding_signature
    graph_space = _embedding_space_signature(graph_signature)
    expected_space = _embedding_space_signature(expected)
    if canonical_json(graph_space) != canonical_json(expected_space):
        raise ValueError("graph and query embedding signatures do not match")
    runtime_signature = getattr(hypergraph, "embedding_signature", graph_signature)
    if callable(runtime_signature):
        runtime_signature = runtime_signature()
    if _embedding_space_signature(runtime_signature) != graph_space:
        raise ValueError(
            "runtime entity VDB and graph embedding signatures do not match"
        )


def _keyword_candidates(
    graph: nx.MultiDiGraph,
    entity_ids: Sequence[str],
    query: str,
    cap: int,
) -> List[str]:
    query_tokens = _name_tokens(query)
    ranked: List[Tuple[int, int, str]] = []
    for entity_id in sorted(set(str(item) for item in entity_ids)):
        if entity_id not in graph or graph.nodes[entity_id].get("role") != "entity":
            continue
        texts = _entity_search_texts(graph, entity_id)
        overlap = max(
            (len(query_tokens & _name_tokens(text)) for text in texts),
            default=0,
        )
        ranked.append((overlap, unique_neighbor_count(graph, entity_id), entity_id))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    positive = [entity_id for overlap, _, entity_id in ranked if overlap > 0]
    return (positive or [item[2] for item in ranked])[:cap]


def _find_target_entities(
    graph: nx.MultiDiGraph,
    entity_ids: List[str],
    question: str,
    top_k: int = 5,
    use_embedding: bool = True,
    min_similarity: float = 0.45,
    min_margin: float = 0.03,
    *,
    candidate_cap: int = 100,
    hypergraph: Any = None,
    inference_config: Optional[ResolvedConfig] = None,
) -> List[str]:


    available = [
        str(entity_id)
        for entity_id in sorted(set(entity_ids), key=str)
        if entity_id in graph and graph.nodes[entity_id].get("role") == "entity"
    ]
    if not available:
        return []
    if question in available:
        return [str(question)]

    query_key = _canonical_text(question)
    exact = [
        entity_id
        for entity_id in available
        if any(
            _canonical_text(text) == query_key
            for text in _entity_search_texts(graph, entity_id)
        )
    ]
    if len(exact) == 1 and not use_embedding:
        return exact
    fuzzy = [
        entity_id
        for entity_id in available
        if any(
            _names_match(question, text)
            for text in _entity_search_texts(graph, entity_id)
        )
    ]
    if len(fuzzy) == 1 and not use_embedding:
        return fuzzy
    if not use_embedding:
        return []
    if hypergraph is None or inference_config is None:
        raise ValueError(
            "dense target resolution requires the sealed entity VDB and config"
        )

    candidates = _keyword_candidates(
        graph,
        exact or fuzzy or available,
        question,
        candidate_cap,
    )
    if not candidates:
        return []
    query_vectors = embed_texts(
        [question],
        config=inference_config,
        stage="target_resolution.query_embedding",
        expected_dim=inference_config.embed_dim,
    )
    if len(query_vectors) != 1:
        raise ValueError("target query embedding returned the wrong vector count")
    search = getattr(hypergraph, "search_entity_vectors", None)
    if not callable(search):
        raise ValueError("validated hypergraph does not expose entity vector search")
    search_k = min(len(candidates), max(top_k + 1, 2))
    hits = search(
        query_vectors[0],
        candidate_ids=set(candidates),
        top_k=search_k,
    )
    if not isinstance(hits, list):
        raise TypeError("entity vector search must return a list")
    validated: List[Tuple[str, float, str]] = []
    runtime_signature = getattr(hypergraph, "embedding_signature", None)
    if callable(runtime_signature):
        runtime_signature = runtime_signature()
    expected_signature = _embedding_space_signature(
        runtime_signature or inference_config.embedding_signature
    )
    for hit in hits:
        if not isinstance(hit, Mapping):
            raise TypeError("entity vector hit must be an object")
        entity_id = str(hit.get("entity_id") or hit.get("node_id") or "")
        if entity_id not in candidates:
            raise ValueError("entity vector search escaped the admitted candidate set")
        similarity = hit.get("similarity")
        if isinstance(similarity, bool):
            raise ValueError("entity similarity cannot be boolean")
        score = float(similarity)
        if not math.isfinite(score):
            raise ValueError("entity similarity must be finite and in [-1,1]")
        if (
            score < -1.0 - _COSINE_BOUND_TOLERANCE
            or score > 1.0 + _COSINE_BOUND_TOLERANCE
        ):
            raise ValueError("entity similarity must be finite and in [-1,1]")
        score = min(1.0, max(-1.0, score))
        signature = _embedding_space_signature(hit.get("embedding_signature"))
        if signature != expected_signature:
            raise ValueError("entity hit embedding signature mismatch")
        canonical_key = str(
            hit.get("canonical_entity_key")
            or graph.nodes[entity_id].get("canonical_entity_key")
            or entity_id
        )
        validated.append((entity_id, score, _canonical_text(canonical_key)))
    validated.sort(key=lambda item: (-item[1], item[0]))
    if not validated or validated[0][1] < min_similarity:
        return []
    top_key = validated[0][2]
    runner_up = next((item for item in validated[1:] if item[2] != top_key), None)
    if runner_up is not None and validated[0][1] - runner_up[1] < min_margin:
        return []
    return [
        entity_id
        for entity_id, _, canonical_key in validated
        if canonical_key == top_key
    ][:top_k]


def _axis_side_to_sign(value: Any, binary_axis: Mapping[str, Any]) -> Optional[int]:
    token = _canonical_text(value)
    if token == "positive":
        return 1
    if token == "negative":
        return -1
    for side_name, sign in (("positive", 1), ("negative", -1)):
        side = binary_axis[side_name]
        allowed = {
            _canonical_text(side["id"]),
            _canonical_text(side["label"]),
            _canonical_text(side["semantics"]),
            *(_canonical_text(alias) for alias in side.get("aliases", [])),
        }
        if token and token in allowed:
            return sign
    return None


def _resolve_target_anchors(
    graph: nx.MultiDiGraph,
    hypergraph: Any,
    question: str,
    direction: Mapping[str, Any],
    binary_axis: Mapping[str, Any],
    config: ResolvedConfig,
) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    entity_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("role") == "entity"
    )
    anchors = direction.get("target_anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("target_anchors must be an array")
    resolved: Dict[str, int] = {}
    conflicted: Set[str] = set()
    dropped: List[Dict[str, str]] = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            raise TypeError("each target anchor must be an object")
        label = str(
            anchor.get("entity") or anchor.get("target") or anchor.get("name") or ""
        ).strip()
        side_value = anchor.get("axis_side")
        sign = _axis_side_to_sign(side_value, binary_axis)
        if sign is None and "yes_orientation" in anchor:
            sign = _parse_yes_orientation(anchor.get("yes_orientation"))
        if not label:
            dropped.append({"anchor": str(index), "reason": "missing_entity"})
            continue
        if sign is None:
            dropped.append(
                {"anchor": label, "reason": "missing_or_invalid_orientation"}
            )
            continue
        target_ids = _find_target_entities(
            graph,
            entity_ids,
            label,
            top_k=config.target_top_k,
            use_embedding=True,
            min_similarity=config.target_min_similarity,
            min_margin=config.target_min_margin,
            candidate_cap=config.target_candidate_cap,
            hypergraph=hypergraph,
            inference_config=config,
        )
        if not target_ids:
            dropped.append({"anchor": label, "reason": "weak_or_ambiguous_match"})
            continue
        for target_id in target_ids:
            if target_id in resolved and resolved[target_id] != sign:
                conflicted.add(target_id)
            else:
                resolved[target_id] = sign
    for target_id in sorted(conflicted):
        resolved.pop(target_id, None)
        dropped.append({"anchor": target_id, "reason": "conflicting_orientation"})
    return dict(sorted(resolved.items())), sorted(
        dropped, key=lambda item: (item["anchor"], item["reason"])
    )


def _find_target_anchors(
    graph: nx.MultiDiGraph,
    entity_ids: List[str],
    question: str,
    direction: Dict[str, Any],
    *,
    binary_axis: Optional[Mapping[str, Any]] = None,
    hypergraph: Any = None,
    inference_config: Optional[ResolvedConfig] = None,
) -> Dict[str, int]:


    if binary_axis is None or hypergraph is None or inference_config is None:
        raw_anchors = (
            direction.get("target_anchors", [])
            if isinstance(direction, Mapping)
            else []
        )
        resolved: Dict[str, int] = {}
        conflicted: Set[str] = set()
        for anchor in raw_anchors if isinstance(raw_anchors, list) else []:
            if not isinstance(anchor, Mapping):
                continue
            label = str(anchor.get("entity") or anchor.get("target") or "").strip()
            sign = _parse_yes_orientation(anchor.get("yes_orientation"))
            if not label or sign is None:
                continue
            targets = (
                [label]
                if label in entity_ids
                else _find_target_entities(
                    graph, entity_ids, label, use_embedding=False
                )
            )
            for target in targets:
                if target in resolved and resolved[target] != sign:
                    conflicted.add(target)
                else:
                    resolved[target] = sign
        for target in conflicted:
            resolved.pop(target, None)
        return dict(sorted(resolved.items()))
    resolved, _ = _resolve_target_anchors(
        graph,
        hypergraph,
        question,
        direction,
        binary_axis,
        inference_config,
    )
    return resolved


def _validate_direction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("reasoning direction must be an object")
    required = {
        "entities",
        "target_anchors",
        "hypotheses",
        "dimensions",
        "search_queries",
        "confidence_prior",
    }
    unknown = set(payload) - required
    missing = required - set(payload)
    if unknown:
        raise ValueError(f"reasoning direction has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"reasoning direction is missing fields: {sorted(missing)}")
    anchors = payload.get("target_anchors")
    if not isinstance(anchors, list):
        raise ValueError("reasoning direction requires target_anchors")
    clean_anchors: List[Dict[str, str]] = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise TypeError("target anchor must be an object")
        if set(anchor) != {"entity", "axis_side"}:
            raise ValueError(
                "target anchor schema must be exactly entity and axis_side"
            )
        entity = str(anchor.get("entity", "")).strip()
        side_value = anchor.get("axis_side")
        if not isinstance(side_value, str):
            raise TypeError("target anchor axis_side must be a string")
        side = _canonical_text(side_value)
        if not entity or side not in {"positive", "negative"}:
            raise ValueError(
                "target anchor requires entity and positive/negative axis_side"
            )
        clean_anchors.append({"entity": entity, "axis_side": side})

    def string_list(name: str) -> List[str]:
        value = payload.get(name, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{name} must be a string array")
        return [item.strip() for item in value if item.strip()]

    prior = payload["confidence_prior"]
    if isinstance(prior, bool) or not isinstance(prior, (int, float)):
        raise TypeError("confidence_prior must be a JSON number")
    prior_value = float(prior)
    if not math.isfinite(prior_value) or not 0.0 <= prior_value <= 1.0:
        raise ValueError("confidence_prior must be finite and in [0,1]")
    return {
        "entities": string_list("entities"),
        "target_anchors": clean_anchors,
        "hypotheses": string_list("hypotheses"),
        "dimensions": string_list("dimensions"),
        "search_queries": string_list("search_queries"),
        "confidence_prior": prior_value,
    }


def generate_reasoning_direction(
    question: str,
    graph_summary: str = "",
    *,
    cutoff_date: str = "",
    binary_axis: Optional[Mapping[str, Any]] = None,
    inference_config: Optional[ResolvedConfig] = None,
) -> Dict[str, Any]:
    config = _require_resolved_config(inference_config, "direction generation")
    if binary_axis is None or not cutoff_date:
        raise ValueError("direction generation requires cutoff and binary axis")
    axis = validate_binary_axis(binary_axis)
    prompt = f"""You are resolving causal targets for a probabilistic forecast.

Question: {question}
Prediction cutoff: {cutoff_date}
Only evidence strictly admissible before this cutoff may be used.

Canonical binary axis:
- positive/event: id={axis["positive"]["id"]}; label={axis["positive"]["label"]}; semantics={axis["positive"]["semantics"]}
- negative/complement: id={axis["negative"]["id"]}; label={axis["negative"]["label"]}; semantics={axis["negative"]["semantics"]}

Admitted graph entity labels:
{graph_summary or "[]"}

Choose only unambiguous question-resolving target anchors. Copy each entity
label exactly from the admitted list. Mark axis_side as positive when an
increase or occurrence supports the event side, and negative when it supports
the complement side. If no anchor is unambiguous, return an empty list.

Return exactly this JSON schema:
{{
  "entities": ["relevant admitted entity"],
  "target_anchors": [
    {{"entity": "exact admitted label", "axis_side": "positive"}}
  ],
  "hypotheses": ["causal hypothesis"],
  "dimensions": ["forecast dimension"],
  "search_queries": ["query"],
  "confidence_prior": 0.5
}}"""
    if len(prompt) > config.max_prompt_chars:
        raise ValueError(
            f"direction prompt exceeds max_prompt_chars={config.max_prompt_chars}"
        )
    return chat_json(
        prompt,
        config=config,
        stage="reasoning.direction",
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        schema_validator=_validate_direction_payload,
        correction_attempts=config.json_correction_attempts,
    )


def _event_text(data: Mapping[str, Any], node_id: str) -> str:
    return _clean_node_label(
        data.get("name")
        or data.get("text")
        or data.get("sentence")
        or data.get("proposition_key")
        or node_id
    )


def _event_time(data: Mapping[str, Any]) -> str:
    return str(
        data.get("timestamp_iso")
        or data.get("timestamp")
        or data.get("date")
        or data.get("availability_upper_bound_iso")
        or data.get("availability_upper_bound")
        or ""
    )


def _hyperedge_distance_map(
    graph: nx.MultiDiGraph,
    entity_distances: Mapping[str, float],
) -> Dict[str, float]:
    distances = {str(key): float(value) for key, value in entity_distances.items()}
    for node_id, data in graph.nodes(data=True):
        if data.get("role") != "hyperedge":
            continue
        incident = {
            str(neighbor)
            for neighbor in set(graph.predecessors(node_id))
            | set(graph.successors(node_id))
            if neighbor in graph and graph.nodes[neighbor].get("role") == "entity"
        }
        values = [entity_distances.get(entity_id, math.inf) for entity_id in incident]
        distances[str(node_id)] = min(values) if values else math.inf
    return distances


def _trend(events: Sequence[Mapping[str, Any]]) -> str:
    if len(events) < 2:
        return "insufficient"

    def score(items: Sequence[Mapping[str, Any]]) -> int:
        total = 0
        for item in items:
            words = set(re.findall(r"\b[a-z]+\b", str(item.get("text", "")).casefold()))
            total += len(words & _POSITIVE_WORDS) - len(words & _NEGATIVE_WORDS)
        return total

    split = max(1, min(3, len(events) // 2))
    recent = score(events[:split])
    older = score(events[split : split * 2])
    if recent > older + 1:
        return "improving"
    if recent < older - 1:
        return "deteriorating"
    return "stable"


def _contradictions(
    source_events: Sequence[Mapping[str, Any]],
    cap: int,
) -> List[Dict[str, str]]:
    positive: List[str] = []
    negative: List[str] = []
    for event in source_events:
        text = str(event["text"])
        words = set(re.findall(r"\b[a-z]+\b", text.casefold()))
        has_positive = bool(words & _POSITIVE_WORDS)
        has_negative = bool(words & _NEGATIVE_WORDS)
        if has_positive and not has_negative:
            positive.append(text)
        elif has_negative and not has_positive:
            negative.append(text)
    pairs = [
        {"positive_signal": left, "negative_signal": right}
        for left in positive
        for right in negative
    ]
    return pairs[:cap]


def _bounded_json_items(
    items: Sequence[Mapping[str, Any]],
    *,
    count_cap: int,
    char_cap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items[:count_cap]:
        candidate = result + [dict(item)]
        if char_cap is not None and len(canonical_json(candidate)) > char_cap:
            break
        result = candidate
    return result


def _build_scored_context(
    rg: ReasoningGraph,
    question: str,
    target_signs: Mapping[str, int],
    cutoff_date: str,
    config: ResolvedConfig,
) -> Dict[str, Any]:
    graph = getattr(rg, "_active_fact_graph", None)
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("reasoning graph is missing its active fact view")
    cache_key = _stable_id(
        "ctx_",
        graph.graph.get("active_view_id", ""),
        rg.case_id,
        _canonical_text(question),
        cutoff_date,
        sorted(target_signs.items()),
        config.scientific_config_sha256,
    )
    cache = getattr(rg, "_context_cache", {})
    if cache_key in cache:



        precompute_cache = getattr(rg, "_context_causal_precompute_cache", {})
        if cache_key in precompute_cache:
            rg._causal_precompute = precompute_cache[cache_key]
        else:


            rg._causal_precompute = None
        return copy.deepcopy(cache[cache_key])

    entity_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("role") == "entity"
    )
    entity_distances = (
        compute_causal_distances(graph, list(target_signs), entity_ids, config.d_max)
        if target_signs
        else {entity_id: math.inf for entity_id in entity_ids}
    )
    all_distances = _hyperedge_distance_map(graph, entity_distances)
    hyperedges = [
        (str(node_id), dict(data))
        for node_id, data in graph.nodes(data=True)
        if data.get("role") == "hyperedge"
    ]
    scorer = TemporalValidityScorer(eta=config.eta, d_max=config.d_max)
    infer_tvf = config.ablation_mode == "infer_tvf"
    neutral_validity = config.ablation_mode in {
        "without_ctvf",
        "without_tvf",
        "without_all",
        "w/o CTVF",
        "w_o_ctvf",
    }
    scored, decisions, n_q = scorer.score_nodes_with_decisions(
        hyperedges,
        cutoff_date,
        None if infer_tvf else all_distances,
        config.d_max,
        strict_support_ledger=True,
    )
    score_by_id = {node_id: score for node_id, _, score in scored}
    if neutral_validity:
        score_by_id = {node_id: 1.0 for node_id in score_by_id}

    source_events: List[Dict[str, Any]] = []
    for node_id, data, original_score in scored:
        text = _event_text(data, node_id)
        if not text:
            continue
        source_events.append(
            {
                "node_id": node_id,
                "time": _event_time(data),
                "text": text,
                "ctvf": score_by_id[node_id],
                "admissibility": decisions[node_id].status,
                "record_id": str(data.get("record_id", "")),
                "proposition_key": str(data.get("proposition_key", "")),
                "original_ctvf": original_score,
            }
        )
    source_events.sort(key=lambda item: (-item["ctvf"], item["node_id"]))
    source_events = source_events[: config.hyperedge_working_cap]

    narratives: List[Dict[str, Any]] = []
    for entity_id in entity_ids:
        incident_ids = sorted(
            {
                str(neighbor)
                for neighbor in set(graph.predecessors(entity_id))
                | set(graph.successors(entity_id))
                if str(neighbor) in score_by_id
            }
        )
        events = [event for event in source_events if event["node_id"] in incident_ids][
            : config.timeline_per_entity_cap
        ]
        if not events:
            continue
        entity_ctvf = sum(score_by_id[node_id] for node_id in incident_ids) / len(
            incident_ids
        )
        entity_neighbors = {
            str(neighbor)
            for neighbor in set(graph.predecessors(entity_id))
            | set(graph.successors(entity_id))
            if neighbor in graph and graph.nodes[neighbor].get("role") == "entity"
        }
        structural_weight = unique_neighbor_count(graph, entity_id) * max(
            1,
            sum(
                unique_neighbor_count(graph, neighbor) for neighbor in entity_neighbors
            ),
        )
        narratives.append(
            {
                "entity_id": entity_id,
                "entity": _entity_label(graph, entity_id),
                "structural_weight": structural_weight,
                "ctvf": entity_ctvf,
                "combined_score": structural_weight * (1.0 + entity_ctvf),
                "trend": _trend(events),
                "timeline": events,
            }
        )
    narratives.sort(key=lambda item: (-item["combined_score"], item["entity_id"]))
    narratives = narratives[: config.reasoning_hub_cap]

    trajectory = sorted(
        source_events,
        key=lambda item: (item["time"], item["node_id"]),
        reverse=True,
    )

    causal_links: List[Dict[str, Any]] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        if data.get("role") != "causal":
            continue
        relation = str(data.get("causal_type", "")).casefold()
        if relation not in _CAUSAL_TYPES:
            raise ValueError(f"unsupported active causal type: {relation!r}")
        strength = data.get("strength")
        if isinstance(strength, bool):
            raise ValueError("causal strength cannot be boolean")
        strength_value = float(strength)
        if not math.isfinite(strength_value) or not 0.0 < strength_value <= 1.0:
            raise ValueError("causal strength must be finite and strictly in (0,1]")
        causal_links.append(
            {
                "logical_edge_id": str(data.get("logical_edge_id") or key),
                "cause_id": str(source),
                "cause": _entity_label(graph, str(source)),
                "effect_id": str(target),
                "effect": _entity_label(graph, str(target)),
                "type": relation,
                "strength": strength_value,
                "description": str(data.get("description", "")),
            }
        )
    causal_links.sort(key=lambda item: (-item["strength"], item["logical_edge_id"]))
    causal_links = causal_links[: config.causal_link_working_cap]
    causal_links = causal_links[: config.causal_link_returned_cap]

    node_scores = dict(score_by_id)
    for entity_id in entity_ids:
        adjacent = sorted(
            {
                str(neighbor)
                for neighbor in set(graph.predecessors(entity_id))
                | set(graph.successors(entity_id))
                if str(neighbor) in score_by_id
            }
        )
        node_scores[entity_id] = (
            sum(score_by_id[node_id] for node_id in adjacent) / len(adjacent)
            if adjacent
            else 0.5
        )
    if neutral_validity:
        node_scores = {str(node_id): 1.0 for node_id in graph.nodes}

    context_chains = extract_scored_chains(
        graph,
        entity_ids,
        tvf_scores=node_scores,
        causal_distances=None if infer_tvf else entity_distances,
        max_depth=config.d_max,
        max_chains=config.chain_working_cap,
        target_signs=dict(target_signs),
        F_max=config.F_max,
        tau_phi=config.tau_phi,
    )
    context_chains = context_chains[: config.chain_returned_cap]
    contradictions = _contradictions(source_events, config.contradiction_returned_cap)
    prompt_context = {
        "narratives": copy.deepcopy(narratives),
        "trajectory": _bounded_json_items(
            trajectory,
            count_cap=config.hyperedge_prompt_cap,
            char_cap=config.trajectory_char_cap,
        ),
        "contradictions": copy.deepcopy(
            contradictions[: config.contradiction_prompt_cap]
        ),
        "causal_context": {
            "links": copy.deepcopy(causal_links[: config.causal_link_prompt_cap]),
            "chains": copy.deepcopy(context_chains[: config.chain_prompt_cap]),
        },
        "source_events": copy.deepcopy(source_events[: config.hyperedge_prompt_cap]),
    }
    result = {
        "context_id": cache_key,
        "narratives": narratives,
        "trajectory": trajectory,
        "contradictions": contradictions,
        "causal_context": {
            "links": causal_links,
            "chains": context_chains,
        },
        "source_events": source_events,
        "prompt_context": prompt_context,
        "prompt_caps": {
            "narratives": config.reasoning_hub_cap,
            "trajectory_chars": config.trajectory_char_cap,
            "trajectory_items": config.hyperedge_prompt_cap,
            "contradictions": config.contradiction_prompt_cap,
            "causal_links": config.causal_link_prompt_cap,
            "causal_chains": config.chain_prompt_cap,
            "source_events": config.hyperedge_prompt_cap,
        },
        "node_ctvf_scores": node_scores,
        "entity_distances": {
            key: (value if math.isfinite(value) else None)
            for key, value in sorted(entity_distances.items())
        },
        "ctvf_decisions": {
            node_id: decision.to_dict()
            for node_id, decision in sorted(decisions.items())
        },
        "n_q": n_q,
    }



    active_target_signs = {
        str(target_id): int(sign)
        for target_id, sign in sorted(target_signs.items(), key=lambda item: str(item[0]))
        if str(target_id) in graph
        and graph.nodes[str(target_id)].get("role") == "entity"
    }
    precompute = {
        "schema_version": "chain-causal-precompute-v1",



        "active_graph": graph,
        "active_view_id": str(graph.graph.get("active_view_id", "")),
        "case_id": str(rg.case_id),
        "cutoff": str(graph.graph.get("cutoff", cutoff_date)),
        "graph_bundle_sha256": str(graph.graph.get("graph_bundle_sha256", "")),
        "scientific_config_sha256": str(config.scientific_config_sha256),
        "target_signs": active_target_signs,
        "ablation_mode": str(config.ablation_mode),
        "d_max": int(config.d_max),
        "eta": float(config.eta),
        "B_pi": int(config.B_pi),
        "F_max": config.F_max,
        "tau_phi": float(config.tau_phi),
        "causal_distances": dict(entity_distances),
        "node_tvf_scores": {
            str(key): float(value) for key, value in node_scores.items()
        },
        "ctvf_decisions": copy.deepcopy(
            {node_id: decision.to_dict() for node_id, decision in sorted(decisions.items())}
        ),
        "n_q": {str(key): int(value) for key, value in sorted(n_q.items())},
    }
    precompute_cache = getattr(rg, "_context_causal_precompute_cache", {})
    precompute_cache[cache_key] = precompute
    rg._context_causal_precompute_cache = precompute_cache
    rg._causal_precompute = precompute
    cache[cache_key] = copy.deepcopy(result)
    rg._context_cache = cache
    return result


def _bounded_prior_outcomes(
    rg: ReasoningGraph, config: ResolvedConfig
) -> List[Dict[str, Any]]:
    candidates = [node_id for node_id in rg.speculative_nodes if node_id in rg.graph]
    candidates.sort(
        key=lambda node_id: (
            -int(rg.graph.nodes[node_id].get("round_num", 0)),
            int(rg.graph.nodes[node_id].get("stable_local_index", 0)),
            node_id,
        )
    )
    result: List[Dict[str, Any]] = []
    for node_id in candidates[: config.prior_outcome_cap]:
        data = rg.graph.nodes[node_id]
        item = {
            "outcome_id": str(data.get("outcome_id", "")),
            "name": str(data.get("name", "")),
            "probability": float(data.get("confidence", 0.0)),
            "description": str(data.get("description", ""))[
                : config.prior_description_char_cap
            ],
        }
        if len(canonical_json(result + [item])) > config.prior_serialized_char_cap:
            break
        result.append(item)
    return result


def _canonical_outcome_key(name: str, description: str) -> str:
    return canonical_json(
        {
            "name": _canonical_text(name),
            "description": _canonical_text(description),
        }
    )


def _outcome_validator(
    *,
    case_id: str,
    question: str,
    round_num: int,
    tolerance: float,
):
    def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
        required_root = {
            "outcomes",
            "pivot_factor",
            "need_more_info",
            "info_queries",
            "reasoning",
        }
        unknown_root = set(payload) - required_root
        missing_root = required_root - set(payload)
        if unknown_root:
            raise ValueError(
                f"outcome response has unknown fields: {sorted(unknown_root)}"
            )
        if missing_root:
            raise ValueError(
                f"outcome response is missing fields: {sorted(missing_root)}"
            )
        outcomes = payload.get("outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise ValueError("outcomes must be a non-empty array")
        parsed: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()
        for raw in outcomes:
            if not isinstance(raw, Mapping):
                raise TypeError("each outcome must be an object")
            allowed_item = {"id", "name", "description", "indicators", "probability"}
            unknown_item = set(raw) - allowed_item
            required_item = {"name", "description", "indicators", "probability"}
            missing_item = required_item - set(raw)
            if unknown_item:
                raise ValueError(f"outcome has unknown fields: {sorted(unknown_item)}")
            if missing_item:
                raise ValueError(f"outcome is missing fields: {sorted(missing_item)}")
            if not isinstance(raw.get("name"), str) or not isinstance(
                raw.get("description"), str
            ):
                raise TypeError("outcome name and description must be strings")
            name = raw["name"].strip()
            description = raw["description"].strip()
            if not name or not description:
                raise ValueError("each outcome requires non-empty name and description")
            probability = raw.get("probability")
            if isinstance(probability, bool) or not isinstance(
                probability, (int, float)
            ):
                raise TypeError("outcome probability must be a JSON number")
            value = float(probability)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("outcome probability must be finite and in [0,1]")
            indicators = raw.get("indicators", [])
            if not isinstance(indicators, list) or not all(
                isinstance(item, str) for item in indicators
            ):
                raise ValueError("outcome indicators must be a string array")
            key = _canonical_outcome_key(name, description)
            if key in seen_keys:
                raise ValueError("duplicate canonical outcome in one round")
            seen_keys.add(key)
            parsed.append(
                {
                    "provider_outcome_id": str(raw.get("id", "")),
                    "name": name,
                    "description": description,
                    "indicators": [item.strip() for item in indicators if item.strip()],
                    "probability": value,
                    "canonical_outcome_key": key,
                }
            )
        total = sum(item["probability"] for item in parsed)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("outcome probability mass must be finite and positive")
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"outcome probability mass differs from 1 by more than {tolerance}"
            )
        parsed.sort(key=lambda item: item["canonical_outcome_key"])
        question_identity = _stable_id("query_", _canonical_text(question))
        for index, item in enumerate(parsed):
            item["probability"] = item["probability"] / total
            item["stable_local_index"] = index
            item["outcome_id"] = _stable_id(
                "out_",
                case_id,
                question_identity,
                round_num,
                index,
                item["canonical_outcome_key"],
            )
        if not isinstance(payload["pivot_factor"], str):
            raise TypeError("pivot_factor must be a string")
        if not isinstance(payload["reasoning"], str):
            raise TypeError("reasoning must be a string")
        need_more_info = payload["need_more_info"]
        if not isinstance(need_more_info, bool):
            raise ValueError("need_more_info must be boolean")
        info_queries = payload["info_queries"]
        if not isinstance(info_queries, list) or not all(
            isinstance(item, str) for item in info_queries
        ):
            raise ValueError("info_queries must be a string array")
        return {
            "outcomes": parsed,
            "pivot_factor": payload["pivot_factor"].strip(),
            "need_more_info": need_more_info,
            "info_queries": [item.strip() for item in info_queries if item.strip()],
            "reasoning": payload["reasoning"].strip(),
        }

    return validate


def _reasoning_prompt(
    question: str,
    round_num: int,
    cutoff_date: str,
    binary_axis: Mapping[str, Any],
    context: Mapping[str, Any],
    priors: Sequence[Mapping[str, Any]],
) -> str:
    axis = binary_axis
    prompt_context = context["prompt_context"]
    return f"""You are a probabilistic future-event prediction agent.

Round: {round_num}
Question: {question}
Prediction cutoff: {cutoff_date}
Canonical positive/event side: {axis["positive"]["label"]} — {axis["positive"]["semantics"]}
Canonical negative/complement side: {axis["negative"]["label"]} — {axis["negative"]["semantics"]}

All evidence below comes from one cutoff-safe active graph view. Known evidence
at or after the prediction cutoff is excluded. Missing event time is included
only with a verified availability upper bound strictly before the cutoff.
Scientific evidence weights use the Causal-Temporal Validity Function (CTVF).

CONTEXT 1 — ENTITY NARRATIVES
{canonical_json(prompt_context["narratives"])}

CONTEXT 2 — OVERALL TRAJECTORY
{canonical_json(prompt_context["trajectory"])}

CONTEXT 3 — CONTRADICTIONS
{canonical_json(prompt_context["contradictions"])}

CONTEXT 4 — LIGHTWEIGHT CAUSAL-CONTEXT CHAINS
{canonical_json(prompt_context["causal_context"])}

CONTEXT 5 — SOURCE EVENTS
{canonical_json(prompt_context["source_events"])}

PRIOR ROUND OUTCOMES
{canonical_json(list(priors))}

Define mutually exclusive, collectively exhaustive outcomes that directly
answer the question at its resolution horizon. Use causal mechanisms, CTVF
weighted evidence, trajectory, contradictions, and structural constraints.
Do not describe merely the current trend. Probabilities must be finite,
non-negative, and sum to exactly 1 within the requested numerical tolerance.

Return exactly this JSON schema:
{{
  "outcomes": [
    {{
      "id": "provider-local optional id",
      "name": "concise resolved future state",
      "description": "one sentence world state at resolution",
      "indicators": ["evidence type"],
      "probability": 0.25
    }}
  ],
  "pivot_factor": "single decisive variable",
  "need_more_info": false,
  "info_queries": [],
  "reasoning": "concise causal explanation"
}}"""


def _add_outcomes_and_derivations(
    rg: ReasoningGraph,
    outcomes: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    round_num: int,
    config: ResolvedConfig,
) -> List[str]:
    outcome_nodes: List[str] = []
    for outcome in outcomes:
        node_id = rg.add_speculative_node(
            name=str(outcome["name"]),
            description=str(outcome["description"]),
            confidence=float(outcome["probability"]),
            round_num=round_num,
            outcome_id=str(outcome["outcome_id"]),
            stable_local_index=int(outcome["stable_local_index"]),
            canonical_outcome_key=str(outcome["canonical_outcome_key"]),
            provider_outcome_id=str(outcome.get("provider_outcome_id", "")),
            indicators=list(outcome.get("indicators", [])),
        )
        outcome_nodes.append(node_id)

    if config.chain_skip_viz_embed or not outcomes:
        return outcome_nodes
    events = list(context["source_events"])
    if not events:
        return outcome_nodes
    outcome_texts = [
        ". ".join(outcome.get("indicators", []))
        or f"{outcome['name']}. {outcome['description']}"
        for outcome in outcomes
    ]
    event_texts = [str(event["text"]) for event in events]
    vectors = embed_texts(
        outcome_texts + event_texts,
        config=config,
        stage="reasoning.derivation_embedding",
        expected_dim=config.embed_dim,
    )
    if len(vectors) != len(outcome_texts) + len(event_texts):
        raise ValueError("derivation embedding returned the wrong vector count")
    outcome_vectors = vectors[: len(outcome_texts)]
    event_vectors = vectors[len(outcome_texts) :]
    local_index = 0
    for event_index, event in enumerate(events):
        best_index = max(
            range(len(outcomes)),
            key=lambda index: (
                cosine_sim(event_vectors[event_index], outcome_vectors[index]),
                -index,
            ),
        )
        similarity = cosine_sim(event_vectors[event_index], outcome_vectors[best_index])
        if not math.isfinite(similarity):
            raise ValueError("derivation similarity is non-finite")
        if similarity < config.derivation_similarity_threshold:
            continue
        source = str(event["node_id"])
        target = outcome_nodes[best_index]
        rg.add_derivation_edge(
            source,
            target,
            confidence=similarity,
            round_num=round_num,
            local_index=local_index,
            evidence_id=source,
            outcome_id=str(outcomes[best_index]["outcome_id"]),
        )
        local_index += 1
    return outcome_nodes


def extract_subgraph(
    hypergraph: MMDTHypergraph,
    direction: Optional[Mapping[str, Any]] = None,
) -> nx.MultiDiGraph:
    del direction
    graph = hypergraph.get_graph()
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("validated CHAIN graph must be an nx.MultiDiGraph")
    return graph


def extract_and_fork(
    hypergraph: MMDTHypergraph,
    direction: Dict[str, Any],
    *,
    case_id: str = "",
    question: str = "",
) -> ReasoningGraph:
    return ReasoningGraph(
        extract_subgraph(hypergraph, direction),
        case_id=case_id,
        question=question,
    )


def reasoning_step(
    rg: ReasoningGraph,
    question: str,
    round_num: int,
    *,
    cutoff_date: str,
    binary_axis: Mapping[str, Any],
    inference_config: ResolvedConfig,
) -> Dict[str, Any]:
    inference_config = _require_resolved_config(inference_config, "reasoning_step")
    axis = validate_binary_axis(binary_axis)
    _require_active_reasoning_identity(
        rg,
        question=question,
        cutoff_date=cutoff_date,
        case_id=rg.case_id,
        binary_axis=axis,
        inference_config=inference_config,
        entrypoint="reasoning_step",
    )
    runtime_spans = _runtime_spans(rg)
    target_signs = dict(getattr(rg, "_target_signs", {}))
    with _timed_runtime_span(runtime_spans, "evidence_retrieval_ctvf", round_num):
        context = _build_scored_context(
            rg,
            question,
            target_signs,
            cutoff_date,
            inference_config,
        )
    priors = _bounded_prior_outcomes(rg, inference_config)
    prompt = _reasoning_prompt(
        question,
        round_num,
        cutoff_date,
        axis,
        context,
        priors,
    )
    if len(prompt) > inference_config.max_prompt_chars:
        raise ValueError(
            f"reasoning prompt exceeds max_prompt_chars={inference_config.max_prompt_chars}"
        )
    with _timed_runtime_span(runtime_spans, "outcome_generation", round_num):
        result = chat_json(
            prompt,
            config=inference_config,
            stage=f"reasoning.outcomes.round_{round_num}",
            temperature=inference_config.llm_temperature,
            max_tokens=inference_config.llm_max_tokens,
            schema_validator=_outcome_validator(
                case_id=rg.case_id,
                question=question,
                round_num=round_num,
                tolerance=inference_config.outcome_sum_tolerance,
            ),
            correction_attempts=inference_config.json_correction_attempts,
        )
    outcomes = result["outcomes"]
    with _timed_runtime_span(runtime_spans, "derivation_processing", round_num):
        outcome_nodes = _add_outcomes_and_derivations(
            rg,
            outcomes,
            context,
            round_num,
            inference_config,
        )
    top = max(
        outcomes,
        key=lambda item: (item["probability"], -item["stable_local_index"]),
    )
    result["confidence"] = float(top["probability"])
    result["event_name"] = str(top["name"])
    result["inferences"] = [
        {
            "outcome_id": item["outcome_id"],
            "event_name": item["name"],
            "probability": item["probability"],
            "description": item["description"],
        }
        for item in outcomes
    ]





    result["scored_context"] = copy.deepcopy(context)
    rg._last_context = context
    rg.log_round(
        round_num,
        "reason",
        {
            "new_nodes": outcome_nodes,
            "inferences": result["inferences"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "pivot": result["pivot_factor"],
            "need_more_info": result["need_more_info"],
            "info_queries": result["info_queries"],
            "context_id": context["context_id"],
        },
    )


    canonical_json(result)
    return result


def _reason_step_impl(
    question: str,
    hypergraph: MMDTHypergraph,
    rg: Optional[ReasoningGraph],
    round_num: int,
    cutoff_date: str = "",
    case_id: str = "",
    binary_axis: Optional[Mapping[str, Any]] = None,
    inference_config: Optional[ResolvedConfig] = None,
    _runtime_spans_sink: Optional[List[Dict[str, Any]]] = None,
) -> tuple[ReasoningGraph, Dict[str, Any]]:
    inference_config = _require_resolved_config(inference_config, "reason_step")
    if not cutoff_date or not str(case_id).strip():
        raise ValueError("reason_step requires per-case cutoff and case_id")
    if binary_axis is None:
        raise ValueError("reason_step requires a canonical binary axis")
    axis = validate_binary_axis(binary_axis)
    runtime_spans: List[Dict[str, Any]]
    if rg is None:
        full_graph = extract_subgraph(hypergraph)
        if full_graph.graph.get("validated_graph_bundle") is not True:
            raise ValueError("reasoning requires a validated graph bundle")
        _validate_embedding_boundary(full_graph, hypergraph, inference_config)
        bundle_hash = str(full_graph.graph.get("graph_bundle_sha256", ""))
        if not bundle_hash:
            raise ValueError("validated graph is missing graph_bundle_sha256")
        runtime_spans = _runtime_spans_sink if _runtime_spans_sink is not None else []
        with _timed_runtime_span(runtime_spans, "active_graph_view", round_num):
            active_graph = build_active_logical_view(
                full_graph,
                cutoff=cutoff_date,
                case_id=str(case_id),
                graph_bundle_sha256=bundle_hash,
                scientific_config_sha256=inference_config.scientific_config_sha256,
            )
        with _timed_runtime_span(runtime_spans, "reasoning_direction", round_num):





            graph_summary = _graph_entity_label_summary(
                active_graph,
                question,
                max_entities=inference_config.direction_entity_cap,
                max_chars=inference_config.direction_char_cap,
            )
            direction = generate_reasoning_direction(
                question,
                graph_summary,
                cutoff_date=cutoff_date,
                binary_axis=axis,
                inference_config=inference_config,
            )
        rg = ReasoningGraph(
            active_graph,
            case_id=str(case_id),
            question=question,
            graph_bundle_sha256=bundle_hash,
            path_cutoff=inference_config.serialized_path_cap,
            speculative_merge_threshold=inference_config.speculative_merge_threshold,
            speculative_merge_min_tokens=inference_config.speculative_merge_min_tokens,
        )
        rg._operational_runtime_spans = runtime_spans







        rg._active_fact_graph = active_graph
        rg._direction = direction
        rg._cutoff = str(active_graph.graph["cutoff"])
        rg._binary_axis_sha256 = axis["binary_axis_sha256"]
        rg._scientific_config_sha256 = inference_config.scientific_config_sha256
        with _timed_runtime_span(runtime_spans, "target_resolution", round_num):
            target_signs, dropped = _resolve_target_anchors(
                active_graph,
                hypergraph,
                question,
                direction,
                axis,
                inference_config,
            )
        rg._target_signs = target_signs
        rg._target_resolution_drops = dropped
        rg._cached_target_anchors = target_signs
        rg._cached_target_entities = list(target_signs)
    else:
        runtime_spans = (
            _runtime_spans_sink
            if _runtime_spans_sink is not None
            else _runtime_spans(rg)
        )
        _require_active_reasoning_identity(
            rg,
            question=question,
            cutoff_date=cutoff_date,
            case_id=str(case_id),
            binary_axis=axis,
            inference_config=inference_config,
            entrypoint="reason_step",
        )
    step_result = reasoning_step(
        rg,
        question,
        round_num,
        cutoff_date=cutoff_date,
        binary_axis=axis,
        inference_config=inference_config,
    )
    return rg, step_result


def reason_step(
    question: str,
    hypergraph: MMDTHypergraph,
    rg: Optional[ReasoningGraph],
    round_num: int,
    cutoff_date: str = "",
    case_id: str = "",
    binary_axis: Optional[Mapping[str, Any]] = None,
    inference_config: Optional[ResolvedConfig] = None,
) -> tuple[ReasoningGraph, Dict[str, Any]]:


    runtime_spans = _runtime_spans(rg) if rg is not None else []
    started = time.perf_counter()
    try:
        result_rg, step_result = _reason_step_impl(
            question,
            hypergraph,
            rg,
            round_num,
            cutoff_date=cutoff_date,
            case_id=case_id,
            binary_axis=binary_axis,
            inference_config=inference_config,
            _runtime_spans_sink=runtime_spans,
        )
    except BaseException:
        _append_runtime_span(
            runtime_spans,
            stage="reasoning_round_total",
            round_num=round_num,
            started=started,
            status="error",
        )
        raise
    result_spans = _runtime_spans(result_rg)
    if not any(
        span.get("stage") == "reasoning_round_total" and span.get("round") == round_num
        for span in result_spans
        if isinstance(span, Mapping)
    ):
        _append_runtime_span(
            result_spans,
            stage="reasoning_round_total",
            round_num=round_num,
            started=started,
            status="ok",
        )
    return result_rg, step_result


def _latest_outcomes(
    rg: ReasoningGraph,
    tolerance: float,
) -> Tuple[List[Dict[str, Any]], int]:
    nodes = [node_id for node_id in rg.speculative_nodes if node_id in rg.graph]
    if not nodes:
        raise ValueError("reasoning graph contains no outcome nodes")
    max_round = max(int(rg.graph.nodes[node].get("round_num", 0)) for node in nodes)
    latest = [
        node
        for node in nodes
        if int(rg.graph.nodes[node].get("round_num", 0)) == max_round
    ]
    latest.sort(
        key=lambda node: (
            int(rg.graph.nodes[node].get("stable_local_index", 0)),
            node,
        )
    )
    outcomes: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node in latest:
        data = rg.graph.nodes[node]
        outcome_id = str(data.get("outcome_id", "")).strip()
        if not outcome_id or outcome_id in seen:
            raise ValueError("latest outcome IDs must be non-empty and unique")
        seen.add(outcome_id)
        probability = data.get("confidence")
        if isinstance(probability, bool):
            raise ValueError("stored outcome probability cannot be boolean")
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("stored outcome probability must be finite and in [0,1]")
        outcomes.append(
            {
                "outcome_id": outcome_id,
                "name": str(data.get("name", "")),
                "description": str(data.get("description", "")),
                "probability": value,
            }
        )
    total = sum(item["probability"] for item in outcomes)
    if total <= 0.0 or abs(total - 1.0) > tolerance:
        raise ValueError("latest outcome distribution is incomplete or invalid")
    for item in outcomes:
        item["probability"] = item["probability"] / total
    return outcomes, max_round


def _mapping_validator(
    outcomes: Sequence[Mapping[str, Any]],
    binary_axis: Mapping[str, Any],
):
    if not outcomes:
        raise ValueError("outcome mapping requires at least one outcome")
    ordered_ids = [str(item["outcome_id"]).strip() for item in outcomes]
    if any(not outcome_id for outcome_id in ordered_ids):
        raise ValueError("outcome mapping requires non-empty canonical outcome IDs")
    expected = set(ordered_ids)
    if len(expected) != len(ordered_ids):
        raise ValueError("outcome mapping requires unique canonical outcome IDs")
    allowed_answers = _mapping_allowed_answers(binary_axis)
    allowed_answer_set = set(allowed_answers)

    def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
        if set(payload) != {"mappings"}:
            raise ValueError("mapping response schema must contain only mappings")
        mappings = payload.get("mappings")
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("mappings must be a non-empty array")
        resolved: Dict[str, str] = {}
        for item in mappings:
            if not isinstance(item, Mapping):
                raise TypeError("each mapping must be an object")
            if set(item) != {"outcome_id", "answer"}:
                raise ValueError(
                    "mapping item schema must be exactly outcome_id and answer"
                )
            if not isinstance(item["outcome_id"], str) or not isinstance(
                item["answer"], str
            ):
                raise TypeError("mapping outcome_id and answer must be strings")
            outcome_id = item["outcome_id"].strip()
            if outcome_id not in expected:
                raise ValueError(f"unknown outcome_id in mapping: {outcome_id!r}")
            if outcome_id in resolved:
                raise ValueError(f"duplicate outcome_id in mapping: {outcome_id}")
            answer = item["answer"]
            if answer not in allowed_answer_set:
                raise ValueError(
                    "mapping answer must exactly match one canonical axis id or label"
                )
            sign = _axis_side_to_sign(answer, binary_axis)
            if sign is None:
                raise ValueError("mapping answer is not one canonical axis side")
            resolved[outcome_id] = "positive" if sign == 1 else "negative"
        if set(resolved) != expected:
            missing = sorted(expected - set(resolved))
            raise ValueError(
                f"outcome mapping is not an exact cover; missing={missing}"
            )
        ordered = [
            {
                "outcome_id": outcome_id,
                "axis_side": resolved[outcome_id],
            }
            for outcome_id in ordered_ids
        ]
        return {"mappings": ordered}

    return validate


def _mapping_allowed_answers(binary_axis: Mapping[str, Any]) -> List[str]:
    allowed: List[str] = []
    for side_name in ("positive", "negative"):
        side = binary_axis[side_name]
        for field_name in ("id", "label"):
            value = str(side[field_name])
            if value not in allowed:
                allowed.append(value)
    return allowed


def _mapping_prompt(
    question: str,
    cutoff_date: str,
    outcomes: Sequence[Mapping[str, Any]],
    binary_axis: Mapping[str, Any],
) -> str:
    outcome_ids = [str(item["outcome_id"]).strip() for item in outcomes]
    if not outcome_ids or any(not outcome_id for outcome_id in outcome_ids):
        raise ValueError("mapping prompt requires non-empty canonical outcome IDs")
    if len(set(outcome_ids)) != len(outcome_ids):
        raise ValueError("mapping prompt requires unique canonical outcome IDs")
    allowed_answers = _mapping_allowed_answers(binary_axis)
    lines = [
        {
            "outcome_id": outcome_id,
            "name": item["name"],
            "description": item["description"],
            "probability": item["probability"],
        }
        for item, outcome_id in zip(outcomes, outcome_ids)
    ]
    skeleton = {
        "mappings": [
            {
                "outcome_id": outcome_id,
                "answer": "",
            }
            for outcome_id in outcome_ids
        ]
    }
    return f"""Map every forecast outcome to exactly one side of the canonical binary axis.

Question: {question}
Prediction cutoff: {cutoff_date}
Positive/event side: id={binary_axis["positive"]["id"]}; label={binary_axis["positive"]["label"]}; semantics={binary_axis["positive"]["semantics"]}
Negative/complement side: id={binary_axis["negative"]["id"]}; label={binary_axis["negative"]["label"]}; semantics={binary_axis["negative"]["semantics"]}

Allowed answer strings (canonical JSON; copy one item verbatim):
{canonical_json(allowed_answers)}

Outcome distribution:
{canonical_json(lines)}

The prefilled JSON skeleton below contains every required canonical outcome_id
exactly once. Return that complete skeleton; never return an empty mappings
array and never delete, omit, duplicate, reorder, rename, or invent an object,
outcome_id, or field. Change only each empty answer string. Every answer must
be copied character-for-character from Allowed answer strings. Never use the
generic substitutes positive, negative, true, or false; never add an
explanation, qualification, paraphrase, or rewritten label. Do not change
probabilities.

Return this prefilled JSON skeleton with only the answer string values changed:
{canonical_json(skeleton)}"""


def _axis_positive_is_yes(binary_axis: Mapping[str, Any]) -> bool:
    positive = binary_axis["positive"]
    negative = binary_axis["negative"]
    return (
        _canonical_text(positive["id"]) == "yes"
        and _canonical_text(positive["label"]) == "yes"
        and _canonical_text(negative["id"]) == "no"
        and _canonical_text(negative["label"]) == "no"
    )


def _reason_aggregate_impl(
    rg: ReasoningGraph,
    question: str,
    cutoff_date: str = "",
    case_id: str = "",
    binary_axis: Optional[Mapping[str, Any]] = None,
    inference_config: Optional[ResolvedConfig] = None,
    _runtime_spans_sink: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    inference_config = _require_resolved_config(inference_config, "reason_aggregate")
    if not cutoff_date or not str(case_id).strip():
        raise ValueError("reason_aggregate requires per-case cutoff and case_id")
    if binary_axis is None:
        raise ValueError("reason_aggregate requires a canonical binary axis")
    axis = validate_binary_axis(binary_axis)
    active_graph = _require_active_reasoning_identity(
        rg,
        question=question,
        cutoff_date=cutoff_date,
        case_id=str(case_id),
        binary_axis=axis,
        inference_config=inference_config,
        entrypoint="reason_aggregate",
    )

    outcomes, latest_round = _latest_outcomes(
        rg, inference_config.outcome_sum_tolerance
    )
    mapping_prompt = _mapping_prompt(question, cutoff_date, outcomes, axis)
    runtime_spans = (
        _runtime_spans_sink if _runtime_spans_sink is not None else _runtime_spans(rg)
    )
    with _timed_runtime_span(runtime_spans, "outcome_mapping", int(latest_round)):
        mapping = chat_json(
            mapping_prompt,
            config=inference_config,
            stage="reasoning.outcome_mapping",
            temperature=inference_config.llm_temperature,
            max_tokens=inference_config.outcome_mapping_max_tokens,
            schema_validator=_mapping_validator(outcomes, axis),
            correction_attempts=inference_config.json_correction_attempts,
        )
        side_by_id = {
            item["outcome_id"]: item["axis_side"] for item in mapping["mappings"]
        }
        positive_mass = sum(
            item["probability"]
            for item in outcomes
            if side_by_id[item["outcome_id"]] == "positive"
        )
        negative_mass = sum(
            item["probability"]
            for item in outcomes
            if side_by_id[item["outcome_id"]] == "negative"
        )
        denominator = positive_mass + negative_mass
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "mapped binary probability mass must be finite and positive"
            )
        if abs(denominator - 1.0) > inference_config.outcome_sum_tolerance:
            raise ValueError("exact-cover mapping did not preserve total outcome mass")
        p_llm_event = positive_mass / denominator

    entity_ids = sorted(
        str(node_id)
        for node_id, data in active_graph.nodes(data=True)
        if data.get("role") == "entity"
    )
    target_signs = dict(getattr(rg, "_target_signs", {}))
    with _timed_runtime_span(runtime_spans, "causal_inference", int(latest_round)):
        causal_result = run_causal_inference(
            active_graph,
            entity_ids,
            p_llm_event=p_llm_event,
            cutoff_date=cutoff_date,
            case_id=str(case_id),
            graph_bundle_sha256=rg.graph_bundle_sha256,
            inference_config=inference_config,
            binary_axis=axis,
            target_signs=target_signs,
            precomputed=getattr(rg, "_causal_precompute", None),
        )
    p_event = float(causal_result["p_event"])
    p_final = float(causal_result["p_final_event"])
    p_causal = float(causal_result["p_causal_event"])
    if p_event != p_final:
        raise ValueError("causal core violated p_event == p_final_event")
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (p_llm_event, p_causal, p_final)
    ):
        raise ValueError("aggregate probabilities must be finite and in [0,1]")
    side = axis["positive"] if p_event >= 0.5 else axis["negative"]
    confidence = p_event if p_event >= 0.5 else 1.0 - p_event
    last_reasoning = str(rg.round_log[-1].get("reasoning", "")) if rg.round_log else ""
    context = getattr(rg, "_last_context", {})
    key_evidence = [str(item["text"]) for item in context.get("source_events", [])[:3]]
    result: Dict[str, Any] = {
        "answer": str(side["label"]),
        "confidence": confidence,
        "p_llm_event": p_llm_event,
        "p_causal_event": p_causal,
        "p_final_event": p_final,
        "p_event": p_event,
        "reasoning_summary": last_reasoning,
        "key_evidence": key_evidence,
        "reasoning_graph": rg.to_dict(),
        "direction": dict(getattr(rg, "_direction", {})),
        "rounds": len(rg.round_log),
        "latest_round": latest_round,
        "outcomes": outcomes,
        "outcome_mapping": mapping,
        "causal_inference": causal_result,
        "target_resolution": {
            "target_signs": target_signs,
            "dropped": list(getattr(rg, "_target_resolution_drops", [])),
        },
        "case_id": str(case_id),
        "cutoff": str(cutoff_date),
        "binary_axis_sha256": axis["binary_axis_sha256"],
        "scientific_config_sha256": inference_config.scientific_config_sha256,
        "graph_bundle_sha256": rg.graph_bundle_sha256,
        "profile_id": inference_config.profile_id,
        "publishable": bool(causal_result.get("publishable", False)),
    }
    if _axis_positive_is_yes(axis):
        result["p_yes"] = p_event
    return result


def reason_aggregate(
    rg: ReasoningGraph,
    question: str,
    cutoff_date: str = "",
    case_id: str = "",
    binary_axis: Optional[Mapping[str, Any]] = None,
    inference_config: Optional[ResolvedConfig] = None,
) -> Dict[str, Any]:


    runtime_spans = _runtime_spans(rg)
    started = time.perf_counter()
    fallback_round = max(
        (
            int(item.get("round", 0))
            for item in getattr(rg, "round_log", [])
            if isinstance(item, Mapping)
        ),
        default=1,
    )
    try:
        result = _reason_aggregate_impl(
            rg,
            question,
            cutoff_date=cutoff_date,
            case_id=case_id,
            binary_axis=binary_axis,
            inference_config=inference_config,
            _runtime_spans_sink=runtime_spans,
        )
    except BaseException:
        _append_runtime_span(
            runtime_spans,
            stage="aggregation_total",
            round_num=fallback_round,
            started=started,
            status="error",
        )
        raise
    latest_round = int(result.get("latest_round", fallback_round))
    if not any(
        span.get("stage") == "aggregation_total" and span.get("round") == latest_round
        for span in runtime_spans
        if isinstance(span, Mapping)
    ):
        _append_runtime_span(
            runtime_spans,
            stage="aggregation_total",
            round_num=latest_round,
            started=started,
            status="ok",
        )
    result["operational_runtime_telemetry"] = _operational_runtime_telemetry(rg)
    canonical_json(result)
    return result


__all__ = [
    "_find_target_anchors",
    "_find_target_entities",
    "_graph_entity_label_summary",
    "_parse_yes_orientation",
    "extract_and_fork",
    "extract_subgraph",
    "generate_reasoning_direction",
    "reason_aggregate",
    "reason_step",
    "reasoning_step",
]
