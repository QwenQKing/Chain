from __future__ import annotations

import copy
import hashlib
import math
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Tuple

import networkx as nx

from chain.config import canonical_json


EDGE_SERIALIZATION = "chain-reasoning-edge-attributes-v1"


def _canonical_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(value.strip().split())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("reasoning graph contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("reasoning graph JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in result:
                raise ValueError(
                    f"duplicate canonical reasoning graph key: {normalized_key!r}"
                )
            result[normalized_key] = _json_safe(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_safe(item) for item in value]
        return sorted(
            items,
            key=canonical_json,
        )
    raise TypeError(f"reasoning graph attribute is not JSON serializable: {type(value)!r}")


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = canonical_json(_json_safe(list(parts))).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()


def _as_multidigraph(graph: Any) -> nx.MultiDiGraph:

    if isinstance(graph, nx.MultiDiGraph):
        return graph.copy(as_view=False)
    result = nx.MultiDiGraph()
    result.graph.update(dict(getattr(graph, "graph", {})))
    for node_id, data in graph.nodes(data=True):
        result.add_node(node_id, **dict(data))
    for source, target, data in graph.edges(data=True):
        edge_data = dict(data)
        key = str(
            edge_data.get("edge_id")
            or edge_data.get("logical_edge_id")
            or _stable_id(
                "edge_",
                source,
                target,
                edge_data.get("role", ""),
                edge_data.get("causal_type", edge_data.get("relation", "")),
                edge_data,
            )
        )
        result.add_edge(source, target, key=key, **edge_data)
    return result


class ReasoningGraph:


    def __init__(
        self,
        subgraph: Any,
        *,
        case_id: str = "",
        question: str = "",
        graph_bundle_sha256: str = "",
        path_cutoff: int = 10,
        speculative_merge_threshold: float = 0.35,
        speculative_merge_min_tokens: int = 2,
    ):
        if isinstance(path_cutoff, bool) or not isinstance(path_cutoff, int) or path_cutoff <= 0:
            raise ValueError("path_cutoff must be a positive integer")
        self.graph = _as_multidigraph(subgraph)
        self.case_id = str(case_id or "")
        self.question_identity = _stable_id("query_", _canonical_text(question))
        self.graph_bundle_sha256 = str(
            graph_bundle_sha256
            or self.graph.graph.get("graph_bundle_sha256", "")
            or self.graph.graph.get("bundle_sha256", "")
        )
        self.path_cutoff = path_cutoff
        self.speculative_merge_threshold = float(speculative_merge_threshold)
        self.speculative_merge_min_tokens = int(speculative_merge_min_tokens)
        if not 0.0 < self.speculative_merge_threshold < 1.0:
            raise ValueError("speculative_merge_threshold must be in (0, 1)")
        if self.speculative_merge_min_tokens <= 0:
            raise ValueError("speculative_merge_min_tokens must be positive")
        self.speculative_nodes: List[str] = sorted(
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("role") == "speculative"
        )
        self.derivation_edges: List[Tuple[str, str, str]] = sorted(
            (
                str(source),
                str(target),
                str(key),
            )
            for source, target, key, data in self.graph.edges(
                keys=True, data=True
            )
            if data.get("role") == "derivation"
        )
        self.round_log: List[Dict[str, Any]] = []

    @property
    def fact_nodes(self) -> List[str]:
        speculative = set(self.speculative_nodes)
        return sorted(
            (node for node in self.graph.nodes if node not in speculative), key=str
        )

    def _outcome_similar(self, a: str, b: str) -> bool:
        if re.search(r"\d", a) or re.search(r"\d", b):
            return _canonical_text(a) == _canonical_text(b)

        def tokens(value: str) -> set:
            return {
                token
                for token in re.split(r"[^a-z0-9]+", value.casefold())
                if len(token) >= 3
            }

        left, right = tokens(a), tokens(b)
        if not left or not right:
            return False
        intersection = len(left & right)
        return (
            intersection >= self.speculative_merge_min_tokens
            and intersection / len(left | right) >= self.speculative_merge_threshold
        )

    def _speculative_id(
        self,
        *,
        round_num: int,
        stable_outcome_id: str,
    ) -> str:
        case_identity = self.case_id or self.question_identity
        return _stable_id(
            "spec_", case_identity, self.question_identity, round_num, stable_outcome_id
        )

    def update_or_add_speculative_node(
        self,
        name: str,
        description: str,
        confidence: float = 0.5,
        candidate_nodes: Optional[Dict[str, str]] = None,
        exclude_ids: Optional[set] = None,
        **attrs: Any,
    ) -> str:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("speculative confidence must be finite and in [0, 1]")
        normalized_name = _canonical_text(name)
        excluded = exclude_ids or set()
        if candidate_nodes is not None:
            items = sorted(
                (
                    (node_id, original)
                    for node_id, original in candidate_nodes.items()
                    if node_id not in excluded
                    and node_id in self.graph
                    and node_id in self.speculative_nodes
                ),
                key=lambda item: str(item[0]),
            )
        else:
            items = [
                (node_id, str(self.graph.nodes[node_id].get("name", "")))
                for node_id in sorted(self.speculative_nodes)
                if node_id not in excluded and node_id in self.graph
            ]
        for node_id, original_name in items:
            existing = _canonical_text(original_name)
            if existing == normalized_name or self._outcome_similar(
                existing, normalized_name
            ):
                self.graph.nodes[node_id].update(
                    {
                        "confidence": confidence,
                        "description": str(description),
                        "name": str(name),
                        **attrs,
                    }
                )
                return node_id
        return self.add_speculative_node(
            name=name,
            description=description,
            confidence=confidence,
            **attrs,
        )

    def add_speculative_node(
        self,
        name: str,
        description: str,
        confidence: float = 0.5,
        timestamp: str = "",
        **attrs: Any,
    ) -> str:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("speculative confidence must be finite and in [0, 1]")
        node_attrs = dict(attrs)
        round_num = int(node_attrs.get("round_num", 0))
        stable_outcome_id = str(
            node_attrs.get("outcome_id")
            or node_attrs.get("stable_outcome_id")
            or _stable_id(
                "out_",
                _canonical_text(name),
                _canonical_text(description),
                node_attrs.get("stable_local_index", 0),
            )
        )
        node_id = self._speculative_id(
            round_num=round_num, stable_outcome_id=stable_outcome_id
        )
        if node_id in self.graph and node_id not in self.speculative_nodes:
            raise ValueError(f"stable speculative node ID collides with a fact: {node_id}")
        node_attrs.update(
            {
                "role": "speculative",
                "name": str(name),
                "description": str(description),
                "confidence": confidence,
                "timestamp": str(timestamp),
                "outcome_id": stable_outcome_id,
            }
        )
        self.graph.add_node(node_id, **node_attrs)
        if node_id not in self.speculative_nodes:
            self.speculative_nodes.append(node_id)
            self.speculative_nodes.sort()
        return node_id

    def add_derivation_edge(
        self,
        source: str,
        target: str,
        relation: str = "causes",
        confidence: float = 0.5,
        *,
        round_num: int = 0,
        local_index: int = 0,
        parent_edge_id: str = "",
        **attrs: Any,
    ) -> str:
        if source not in self.graph or target not in self.graph:
            raise KeyError("derivation endpoints must already exist")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("derivation confidence must be finite and in [0, 1]")
        edge_key = _stable_id(
            "deriv_",
            self.case_id or self.question_identity,
            round_num,
            "derivation",
            source,
            target,
            relation,
            local_index,
            parent_edge_id,
        )
        edge_attrs = dict(attrs)
        edge_attrs.update(
            {
                "edge_id": edge_key,
                "relation": str(relation),
                "confidence": confidence,
                "role": "derivation",
                "round_num": round_num,
                "parent_edge_id": str(parent_edge_id),
            }
        )
        self.graph.add_edge(source, target, key=edge_key, **edge_attrs)
        record = (str(source), str(target), edge_key)
        if record not in self.derivation_edges:
            self.derivation_edges.append(record)
            self.derivation_edges.sort()
        return edge_key

    def update_confidence(self, node_id: str, confidence: float) -> None:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if node_id in self.graph:
            if node_id not in self.speculative_nodes:
                raise ValueError("confidence updates are restricted to speculative nodes")
            self.graph.nodes[node_id]["confidence"] = confidence

    def prune(self, threshold: float) -> List[str]:
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("prune threshold must be in [0, 1]")
        to_remove = sorted(
            node_id
            for node_id in self.speculative_nodes
            if node_id in self.graph
            and float(self.graph.nodes[node_id].get("confidence", 0.0)) < threshold
        )
        for node_id in to_remove:
            self.graph.remove_node(node_id)
            self.speculative_nodes.remove(node_id)
        removed = set(to_remove)
        self.derivation_edges = [
            edge
            for edge in self.derivation_edges
            if edge[0] not in removed and edge[1] not in removed
        ]
        return to_remove

    def log_round(self, round_num: int, action: str, details: Dict[str, Any]) -> None:
        self.round_log.append(
            _json_safe({"round": int(round_num), "action": str(action), **details})
        )

    def _sorted_out_edges(self, node_id: str) -> List[Tuple[str, str, str, Dict]]:
        return sorted(
            (
                (str(source), str(target), str(key), dict(data))
                for source, target, key, data in self.graph.out_edges(
                    node_id, keys=True, data=True
                )
            ),
            key=lambda item: (item[1], item[2], str(item[3].get("role", ""))),
        )

    def get_reasoning_chains(self) -> List[Dict[str, Any]]:
        leaves = sorted(
            node_id
            for node_id in self.speculative_nodes
            if node_id in self.graph and self.graph.out_degree(node_id) == 0
        )
        leaf_set = set(leaves)
        chains: List[Dict[str, Any]] = []
        for start in self.fact_nodes:
            stack = [(start, (start,), tuple())]
            while stack:
                current, node_path, edge_path = stack.pop()
                if current in leaf_set and edge_path:
                    chains.append(
                        {
                            "nodes": list(node_path),
                            "edges": [
                                {"source": u, "target": v, "key": key}
                                for u, v, key in edge_path
                            ],
                        }
                    )
                    continue
                if len(edge_path) >= self.path_cutoff:
                    continue
                outgoing = self._sorted_out_edges(current)
                for source, target, key, data in reversed(outgoing):
                    if data.get("role") != "derivation":
                        continue
                    if target in node_path:
                        continue
                    stack.append(
                        (
                            target,
                            node_path + (target,),
                            edge_path + ((source, target, key),),
                        )
                    )
        chains.sort(
            key=lambda chain: tuple(
                (edge["source"], edge["target"], edge["key"])
                for edge in chain["edges"]
            )
        )
        return chains

    def rebase_facts(self, new_subgraph: Any) -> None:
        speculative_data = {
            node_id: copy.deepcopy(dict(self.graph.nodes[node_id]))
            for node_id in self.speculative_nodes
            if node_id in self.graph
        }
        derivations = [
            (source, target, key, copy.deepcopy(dict(self.graph.edges[source, target, key])))
            for source, target, key in self.derivation_edges
            if self.graph.has_edge(source, target, key)
        ]
        self.graph = _as_multidigraph(new_subgraph)
        for node_id, data in speculative_data.items():
            self.graph.add_node(node_id, **data)
        restored: List[Tuple[str, str, str]] = []
        for source, target, key, data in derivations:
            if target not in self.graph:
                continue


            if source not in self.graph:
                continue
            self.graph.add_edge(source, target, key=key, **data)
            restored.append((str(source), str(target), str(key)))
        self.derivation_edges = sorted(restored)

    def to_dict(self) -> Dict[str, Any]:
        nodes = []
        for node_id, data in sorted(
            self.graph.nodes(data=True), key=lambda item: str(item[0])
        ):
            if "id" in data or "speculative" in data:
                raise ValueError("node attributes may not shadow reserved serialization fields")
            nodes.append(
                _json_safe(
                    {
                        **dict(data),
                        "id": node_id,
                        "speculative": node_id in self.speculative_nodes,
                    }
                )
            )
        edges = []
        for source, target, key, data in sorted(
            self.graph.edges(keys=True, data=True),
            key=lambda item: (str(item[0]), str(item[1]), str(item[2])),
        ):
            endpoint_values = (source, target, key)
            if not all(isinstance(value, str) and value for value in endpoint_values):
                raise ValueError("reasoning graph edge endpoints and key must be non-empty strings")
            edges.append(
                _json_safe(
                    {
                        "source": source,
                        "target": target,
                        "key": key,





                        "attributes": _json_safe(dict(data)),
                    }
                )
            )
        return {
            "schema_version": "chain-reasoning-graph-v1",
            "edge_serialization": EDGE_SERIALIZATION,
            "case_id": self.case_id,
            "question_identity": self.question_identity,
            "graph_bundle_sha256": self.graph_bundle_sha256,
            "serialized_path_cap": self.path_cutoff,
            "speculative_merge_threshold": self.speculative_merge_threshold,
            "speculative_merge_min_tokens": self.speculative_merge_min_tokens,
            "nodes": nodes,
            "edges": edges,
            "round_log": _json_safe(self.round_log),
            "chains": self.get_reasoning_chains(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningGraph":
        if not isinstance(payload, Mapping):
            raise TypeError("reasoning graph payload must be an object")
        allowed_fields = {
            "schema_version",
            "edge_serialization",
            "case_id",
            "question_identity",
            "graph_bundle_sha256",
            "serialized_path_cap",
            "speculative_merge_threshold",
            "speculative_merge_min_tokens",
            "nodes",
            "edges",
            "round_log",
            "chains",
        }
        required_fields = allowed_fields - {"edge_serialization", "chains"}
        unknown_fields = set(payload) - allowed_fields
        missing_fields = required_fields - set(payload)
        if unknown_fields:
            raise ValueError(
                f"reasoning graph has unknown top-level fields: {sorted(unknown_fields)}"
            )
        if missing_fields:
            raise ValueError(
                f"reasoning graph is missing top-level fields: {sorted(missing_fields)}"
            )
        if payload.get("schema_version") != "chain-reasoning-graph-v1":
            raise ValueError("unsupported reasoning graph schema")
        edge_serialization = payload.get("edge_serialization", "")
        if edge_serialization not in {"", EDGE_SERIALIZATION}:
            raise ValueError("unsupported reasoning graph edge serialization")
        nested_edge_attributes = edge_serialization == EDGE_SERIALIZATION
        graph = nx.MultiDiGraph()
        raw_nodes = payload.get("nodes", [])
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise TypeError("reasoning graph nodes/edges must be arrays")
        seen_nodes = set()
        for node in raw_nodes:
            if not isinstance(node, Mapping):
                raise TypeError("reasoning graph node must be an object")
            data = dict(node)
            node_id = data.pop("id", None)
            speculative = data.pop("speculative", None)
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("reasoning graph node id must be a non-empty string")
            if node_id in seen_nodes:
                raise ValueError(f"duplicate reasoning graph node id: {node_id}")
            seen_nodes.add(node_id)
            if not isinstance(speculative, bool):
                raise ValueError("reasoning graph speculative flag must be boolean")
            if not isinstance(data.get("role"), str) or not data.get("role"):
                raise ValueError("reasoning graph node role is required")
            if speculative != (data.get("role") == "speculative"):
                raise ValueError("speculative flag and node role disagree")
            _json_safe(data)
            graph.add_node(node_id, **data)
        seen_edges = set()
        for edge in raw_edges:
            if not isinstance(edge, Mapping):
                raise TypeError("reasoning graph edge must be an object")
            edge_keys = set(edge)
            data = dict(edge)
            source = data.pop("source", None)
            target = data.pop("target", None)
            key = data.pop("key", None)
            if not all(isinstance(value, str) and value for value in (source, target, key)):
                raise ValueError("reasoning graph edge source/target/key are required strings")
            if source not in seen_nodes or target not in seen_nodes:
                raise ValueError("reasoning graph edge endpoint is missing")
            identity = (source, target, key)
            if identity in seen_edges:
                raise ValueError(f"duplicate reasoning graph edge identity: {identity!r}")
            seen_edges.add(identity)
            if nested_edge_attributes:



                if edge_keys != {"source", "target", "key", "attributes"}:
                    raise ValueError("ambiguous reasoning graph edge serialization")
                attributes = edge.get("attributes")
                if not isinstance(attributes, Mapping):
                    raise TypeError("reasoning graph edge attributes must be an object")
                data = dict(attributes)
            if not isinstance(data.get("role"), str) or not data.get("role"):
                raise ValueError("reasoning graph edge role is required")
            _json_safe(data)



            graph.add_edge(source, target, key=key)
            graph.edges[source, target, key].update(data)
        round_log = payload.get("round_log", [])
        if not isinstance(round_log, list) or not all(
            isinstance(item, Mapping) for item in round_log
        ):
            raise TypeError("reasoning graph round_log must be an array of objects")
        _json_safe(round_log)
        instance = cls(
            graph,
            case_id=str(payload.get("case_id", "")),
            graph_bundle_sha256=str(payload.get("graph_bundle_sha256", "")),
            path_cutoff=int(payload.get("serialized_path_cap", 10)),
            speculative_merge_threshold=float(
                payload.get("speculative_merge_threshold", 0.35)
            ),
            speculative_merge_min_tokens=int(
                payload.get("speculative_merge_min_tokens", 2)
            ),
        )
        instance.question_identity = str(payload.get("question_identity", ""))
        instance.round_log = [dict(item) for item in round_log]
        if "chains" in payload:
            supplied_chains = payload["chains"]
            if not isinstance(supplied_chains, list):
                raise TypeError("reasoning graph chains must be an array")
            _json_safe(supplied_chains)
            if supplied_chains != instance.get_reasoning_chains():
                raise ValueError("reasoning graph chains do not match graph topology")
        return instance


__all__ = ["ReasoningGraph"]
