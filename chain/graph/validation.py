from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


GRAPH_BUNDLE_SCHEMA = "chain-graph-bundle-v1"
GRAPH_SCHEMA = "chain-causal-temporal-multidigraph-v1"
CHUNKS_SCHEMA = "chain-chunks-v1"
VECTOR_SCHEMA = "chain-vector-store-v1"
TELEMETRY_SCHEMA = "chain-build-telemetry-v1"
SOURCE_AUDIT_SCHEMA = "chain-source-audit-v1"

KG_ARTIFACTS: Tuple[str, ...] = (
    "knowledge_graph.graphml",
    "entity_vdb.json",
    "hyperedge_vdb.json",
    "chunks.json",
)
METADATA_ARTIFACTS: Tuple[str, ...] = (
    "graph_manifest.json",
    "build_telemetry.json",
)
SOURCE_AUDIT_FILE = "source_audit_manifest.json"
ALLOWED_CAUSAL_TYPES = frozenset({"causes", "enables", "prevents"})


def _xml_char_allowed(value: str) -> bool:


    return all(
        char in "\t\n\r"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        for char in value
    )


def _require_xml_safe_text(value: str, *, context: str) -> None:
    if not isinstance(value, str) or not _xml_char_allowed(value):
        raise GraphBundleValidationError(
            f"{context} contains a character that cannot be represented in XML 1.0 GraphML"
        )





EXPECTED_CONSTRUCTION_CONFIG_SCHEMA = "chain-construction-config-v2"
EXPECTED_CONTENT_NORMALIZATION_VERSION = "unicode-nfc-content-answer-safe-v3"
EXPECTED_KEY_NORMALIZATION_VERSION = "unicode-nfkc-casefold-whitespace-key-v1"
EXPECTED_EXTRACTION_PROMPT_VERSION = "chain-extraction-prompt-v17"
EXPECTED_EXTRACTION_SCHEMA_VERSION = "chain-extraction-schema-v6"
EXPECTED_RELATION_ADMISSIBILITY_POLICY_VERSION = (
    "chain-final-retry-relation-admissibility-v2"
)
EXPECTED_GROUNDING_POLICY_VERSION = "chain-source-grounding-proposition-omission-v3"
EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION = (
    "chain-final-retry-empty-or-duplicate-canonical-entity-proposition-omission-v2"
)



EXPECTED_EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT = 3
EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION = (
    "chain-extraction-response-format-json-object-v1"
)




_RAW_ONLY_KEYS = frozenset(
    {
        "raw_record_sha256",
        "raw_input_file_sha256",
        "question",
        "questions",
        "options",
        "ground_truth",
        "ground_truth_raw",
        "answer",
        "answers",
        "resolution",
        "prediction",
        "predictions",
        "result",
        "results",
        "label",
        "labels",
        "y_true",
    }
)


class GraphBundleValidationError(ValueError):
    pass



class AtomicPublishError(RuntimeError):
    pass



def _normalise_json(value: Any) -> Any:


    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not canonical JSON values")
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            key_n = unicodedata.normalize("NFC", key)
            if key_n in result:
                raise ValueError(f"duplicate canonical JSON key: {key_n!r}")
            result[key_n] = _normalise_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:


    from chain.config import canonical_json as config_canonical_json

    return config_canonical_json(value)


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    from chain.config import hash_payload

    return hash_payload(value)


def _active_builder_construction_identity() -> Dict[str, str]:


    try:
        from chain.graph import kg_builder
    except Exception as exc:
        raise GraphBundleValidationError(
            f"cannot load active graph builder identity: {exc}"
        ) from exc

    expected_constants = {
        "content_normalization_version": EXPECTED_CONTENT_NORMALIZATION_VERSION,
        "key_normalization_version": EXPECTED_KEY_NORMALIZATION_VERSION,
        "extraction_prompt_version": EXPECTED_EXTRACTION_PROMPT_VERSION,
        "extraction_schema_version": EXPECTED_EXTRACTION_SCHEMA_VERSION,
        "relation_admissibility_policy_version": (
            EXPECTED_RELATION_ADMISSIBILITY_POLICY_VERSION
        ),
        "grounding_policy_version": EXPECTED_GROUNDING_POLICY_VERSION,
        "proposition_schema_projection_policy_version": (
            EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
        ),
        "extraction_transport_attempts_per_schema_attempt": (
            EXPECTED_EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT
        ),
        "extraction_response_format_version": (
            EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
        ),
    }
    actual_names = {
        "content_normalization_version": "CONTENT_NORMALIZATION_VERSION",
        "key_normalization_version": "KEY_NORMALIZATION_VERSION",
        "extraction_prompt_version": "PROMPT_VERSION",
        "extraction_schema_version": "EXTRACTION_SCHEMA_VERSION",
        "relation_admissibility_policy_version": (
            "RELATION_ADMISSIBILITY_POLICY_VERSION"
        ),
        "grounding_policy_version": "GROUNDING_POLICY_VERSION",
        "proposition_schema_projection_policy_version": (
            "PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION"
        ),
        "extraction_transport_attempts_per_schema_attempt": (
            "EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT"
        ),
        "extraction_response_format_version": "EXTRACTION_RESPONSE_FORMAT_VERSION",
    }
    for identity_name, expected in expected_constants.items():
        actual = getattr(kg_builder, actual_names[identity_name], None)
        if actual != expected:
            raise GraphBundleValidationError(
                "active graph builder "
                f"{identity_name} mismatch: expected {expected!r}, found {actual!r}"
            )

    system = getattr(kg_builder, "_EXTRACTION_SYSTEM", None)
    prompt = getattr(kg_builder, "_EXTRACT_PROMPT", None)
    correction = getattr(kg_builder, "_EXTRACTION_SCHEMA_CORRECTION", None)
    if (
        not isinstance(system, str)
        or not isinstance(prompt, str)
        or not isinstance(correction, str)
    ):
        raise GraphBundleValidationError(
            "active graph builder extraction prompt is unavailable"
        )
    prompt_sha256 = hashlib.sha256(
        (system + "\0" + prompt + "\0" + correction).encode("utf-8")
    ).hexdigest()
    return {
        **expected_constants,
        "extraction_prompt_sha256": prompt_sha256,
    }


def _grounded_exact(surface: Any, source_text: Any) -> bool:


    if not isinstance(surface, str) or not isinstance(source_text, str):
        return False
    surface_nfc = unicodedata.normalize("NFC", surface)
    source_nfc = unicodedata.normalize("NFC", source_text)
    return bool(surface_nfc.strip()) and surface_nfc in source_nfc


def _canonical_extraction_key(value: str, *, context: str) -> str:


    if not isinstance(value, str):
        raise GraphBundleValidationError(f"{context} must be a string")
    result = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value).strip(),
    ).casefold()
    if not result:
        raise GraphBundleValidationError(f"{context} must be non-empty")
    return result


def _strict_raw_json_loads(raw: str, *, context: str) -> Any:


    if not isinstance(raw, str) or not raw.strip():
        raise GraphBundleValidationError(f"{context} is empty")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GraphBundleValidationError(f"{context} is not strict JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise GraphBundleValidationError(f"{context} must be a JSON object")
    return value


def _trace_raw_response(attempt: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:


    provider_trace = attempt.get("provider_trace")
    if isinstance(provider_trace, Mapping):
        provider_attempts = provider_trace.get("attempts")
        if isinstance(provider_attempts, list):
            for provider_attempt in reversed(provider_attempts):
                if isinstance(provider_attempt, Mapping) and isinstance(
                    provider_attempt.get("raw_response"), str
                ):
                    raw = str(provider_attempt["raw_response"])
                    expected = provider_attempt.get("response_sha256")
                    return raw, str(expected) if isinstance(expected, str) else None
    if isinstance(attempt.get("raw_response"), str):
        raw = str(attempt["raw_response"])
        expected = attempt.get("raw_response_sha256")
        return raw, str(expected) if isinstance(expected, str) else None
    return None, None


def _validate_json_object_provider_trace(
    trace: Mapping[str, Any],
    *,
    context: str,
    max_transport_attempts: Optional[int] = None,
) -> None:


    scientific = trace.get("scientific")
    attempts = trace.get("attempts")
    if max_transport_attempts is not None and (
        isinstance(max_transport_attempts, bool)
        or not isinstance(max_transport_attempts, int)
        or max_transport_attempts <= 0
    ):
        raise GraphBundleValidationError(
            f"{context} has an invalid transport retry budget"
        )
    if (
        trace.get("response_format") != "json_object"
        or trace.get("response_format_version")
        != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
        or not isinstance(scientific, Mapping)
        or scientific.get("response_format") != "json_object"
        or scientific.get("response_format_version")
        != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
        or not isinstance(attempts, list)
        or not attempts
        or any(
            not isinstance(attempt, Mapping)
            or attempt.get("response_format") != "json_object"
            or attempt.get("response_format_version")
            != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
            for attempt in attempts
        )
    ):
        raise GraphBundleValidationError(
            f"{context} does not attest the sealed json_object response format"
        )
    attempts = trace.get("attempts")
    attempt_count = trace.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count <= 0
        or not isinstance(attempts, list)
        or attempt_count != len(attempts)
    ):
        raise GraphBundleValidationError(
            f"{context} has inconsistent transport attempt accounting"
        )
    if max_transport_attempts is not None and attempt_count > max_transport_attempts:
        raise GraphBundleValidationError(
            f"{context} exceeds the sealed transport retry budget"
        )


def _replay_project_extraction(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    chunk_id: str,
) -> Tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
]:


    if set(payload) != {"propositions", "causal_relations"}:
        raise GraphBundleValidationError(
            f"raw extraction {chunk_id} has invalid top-level schema"
        )
    propositions = payload.get("propositions")
    relations = payload.get("causal_relations")
    if not isinstance(propositions, list) or not isinstance(relations, list):
        raise GraphBundleValidationError(f"raw extraction {chunk_id} arrays are invalid")

    normalised_propositions: list[Dict[str, Any]] = []
    grounding_rejections: list[Dict[str, Any]] = []
    proposition_schema_rejections: list[Dict[str, Any]] = []
    proposition_keys: set[str] = set()
    endpoint_by_raw: Dict[str, str] = {}
    endpoint_name_by_raw: Dict[str, str] = {}

    for proposition_index, proposition in enumerate(propositions):
        if not isinstance(proposition, Mapping) or set(proposition) != {
            "sentence",
            "entities",
        }:
            raise GraphBundleValidationError(
                f"raw extraction {chunk_id} proposition {proposition_index} schema mismatch"
            )
        sentence = proposition.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            raise GraphBundleValidationError(
                f"raw extraction {chunk_id} proposition {proposition_index} sentence invalid"
            )
        sentence_nfc = unicodedata.normalize("NFC", sentence.strip())
        entities = proposition.get("entities")
        if not isinstance(entities, list):
            raise GraphBundleValidationError(
                f"raw extraction {chunk_id} proposition {proposition_index} entities invalid"
            )
        if not entities:
            proposition_schema_rejections.append(
                {
                    "proposition_index": proposition_index,
                    "reason_code": "proposition_entities_empty",
                    "proposition_sha256": canonical_sha256(proposition),
                }
            )
            continue
        unsafe_text_fields: list[str] = []
        if not _xml_char_allowed(sentence_nfc):
            unsafe_text_fields.append("sentence")
        prepared_entities: list[Tuple[str, str, str]] = []
        for entity_index, entity in enumerate(entities):
            if (
                not isinstance(entity, Mapping)
                or "name" not in entity
                or set(entity) - {"name", "name_zh"}
            ):
                raise GraphBundleValidationError(
                    f"raw extraction {chunk_id} proposition {proposition_index} "
                    f"entity {entity_index} schema mismatch"
                )
            name = entity.get("name")
            name_zh = entity.get("name_zh", name)
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(name_zh, str)
            ):
                raise GraphBundleValidationError(
                    f"raw extraction {chunk_id} proposition {proposition_index} "
                    f"entity {entity_index} value invalid"
                )
            name_nfc = unicodedata.normalize("NFC", name.strip())
            name_zh_nfc = unicodedata.normalize("NFC", name_zh.strip()) or name_nfc
            if not _xml_char_allowed(name_nfc):
                unsafe_text_fields.append(f"entities[{entity_index}].name")
            if not _xml_char_allowed(name_zh_nfc):
                unsafe_text_fields.append(f"entities[{entity_index}].name_zh")
            prepared_entities.append((name, name_nfc, name_zh_nfc))
        if unsafe_text_fields:
            grounding_rejections.append(
                {
                    "proposition_index": proposition_index,
                    "reason_code": "proposition_contains_xml_unsafe_text",
                    "proposition_sha256": canonical_sha256(proposition),
                    "unsafe_text_fields": sorted(set(unsafe_text_fields)),
                    "entity_count": len(entities),
                }
            )
            continue
        ungrounded_entity_indices: list[int] = []


        for entity_index, (name, _name_nfc, _name_zh_nfc) in enumerate(
            prepared_entities
        ):
            if not _grounded_exact(name, source_text):
                ungrounded_entity_indices.append(entity_index)

        if ungrounded_entity_indices:
            grounding_rejections.append(
                {
                    "proposition_index": proposition_index,
                    "reason_code": "proposition_contains_ungrounded_entity",
                    "proposition_sha256": canonical_sha256(proposition),
                    "ungrounded_entity_indices": ungrounded_entity_indices,
                    "entity_count": len(entities),
                }
            )
            continue

        normalised_entities: list[Dict[str, Any]] = []
        proposition_endpoints: list[Tuple[str, str, str]] = []
        proposition_entity_keys: set[str] = set()
        has_duplicate_canonical_entity = False
        for name, name_nfc, name_zh_nfc in prepared_entities:
            entity_key = _canonical_extraction_key(
                name_nfc,
                context=f"raw extraction {chunk_id} entity key",
            )
            if entity_key in proposition_entity_keys:
                has_duplicate_canonical_entity = True
            proposition_entity_keys.add(entity_key)
            proposition_endpoints.append((name, entity_key, name_nfc))
            normalised_entities.append(
                {
                    "name": name_nfc,
                    "name_zh": name_zh_nfc,
                    "canonical_key": entity_key,
                }
            )

        if has_duplicate_canonical_entity:
            proposition_schema_rejections.append(
                {
                    "proposition_index": proposition_index,
                    "reason_code": "proposition_duplicate_canonical_entity",
                    "proposition_sha256": canonical_sha256(proposition),
                }
            )
            continue

        proposition_key = _canonical_extraction_key(
            sentence_nfc,
            context=f"raw extraction {chunk_id} proposition key",
        )
        if proposition_key in proposition_keys:
            raise GraphBundleValidationError(
                f"raw extraction {chunk_id} has duplicate proposition key"
            )


        proposition_keys.add(proposition_key)
        for raw_name, entity_key, normalised_name in proposition_endpoints:
            endpoint_by_raw[raw_name] = entity_key
            endpoint_name_by_raw[raw_name] = normalised_name
        normalised_propositions.append(
            {
                "sentence": sentence_nfc,
                "proposition_key": proposition_key,
                "source_proposition_index": proposition_index,
                "entities": normalised_entities,
            }
        )

    class RelationReplayError(ValueError):
        def __init__(self, reason_code: str, **details: Any) -> None:
            self.reason_code = reason_code
            self.details = details
            super().__init__(reason_code)

    normalised_relations: list[Dict[str, Any]] = []
    relation_rejections: list[Dict[str, Any]] = []
    relation_keys: set[Tuple[str, str, str]] = set()
    for relation_index, relation in enumerate(relations):
        try:
            allowed = {"cause", "effect", "type", "strength", "description"}
            required = {"cause", "effect", "type", "strength"}
            if (
                not isinstance(relation, Mapping)
                or set(relation) - allowed
                or not required.issubset(relation)
            ):
                raise RelationReplayError("invalid_relation_fields")
            cause = relation.get("cause")
            effect = relation.get("effect")
            relation_type = relation.get("type")
            if (
                not isinstance(cause, str)
                or not cause.strip()
                or not isinstance(effect, str)
                or not effect.strip()
            ):
                raise RelationReplayError("invalid_endpoint_value")
            missing_roles = [
                role
                for role, endpoint in (("cause", cause), ("effect", effect))
                if endpoint not in endpoint_by_raw
            ]
            if missing_roles:
                raise RelationReplayError(
                    "endpoint_not_in_entity_vocabulary",
                    missing_endpoint_roles=missing_roles,
                )
            ungrounded_roles = [
                role
                for role, endpoint in (("cause", cause), ("effect", effect))
                if not _grounded_exact(endpoint, source_text)
            ]
            if ungrounded_roles:
                raise RelationReplayError(
                    "endpoint_not_grounded_in_chunk",
                    missing_endpoint_roles=ungrounded_roles,
                )
            cause_key = endpoint_by_raw[cause]
            effect_key = endpoint_by_raw[effect]
            if cause_key == effect_key:
                raise RelationReplayError("canonical_self_loop")
            if not isinstance(relation_type, str) or relation_type not in ALLOWED_CAUSAL_TYPES:
                raise RelationReplayError("unsupported_causal_type")
            strength = relation.get("strength")
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise RelationReplayError("invalid_strength_type")
            strength_f = float(strength)
            if not math.isfinite(strength_f) or not 0.0 < strength_f <= 1.0:
                raise RelationReplayError("invalid_strength_range")
            description = relation.get("description", "")
            if not isinstance(description, str):
                raise RelationReplayError("invalid_description")
            description_nfc = unicodedata.normalize("NFC", description.strip())
            if not _xml_char_allowed(description_nfc):
                raise RelationReplayError("invalid_description")
            relation_key = (cause_key, effect_key, relation_type)
            if relation_key in relation_keys:
                raise RelationReplayError("duplicate_typed_relation")
            relation_keys.add(relation_key)
            normalised_relations.append(
                {
                    "cause": endpoint_name_by_raw[cause],
                    "effect": endpoint_name_by_raw[effect],
                    "cause_key": cause_key,
                    "effect_key": effect_key,
                    "type": relation_type,
                    "strength": strength_f,
                    "description": description_nfc,
                    "source_relation_index": relation_index,
                }
            )
        except RelationReplayError as exc:
            rejection = {
                "relation_index": relation_index,
                "reason_code": exc.reason_code,
                "relation_sha256": canonical_sha256(relation),
            }
            rejection.update(exc.details)
            relation_rejections.append(rejection)

    if (
        relation_rejections
        and not normalised_propositions
        and not grounding_rejections
        and not proposition_schema_rejections
    ):
        raise GraphBundleValidationError(
            f"raw extraction {chunk_id} relation projection fabricated an empty extraction"
        )
    return (
        {
            "propositions": normalised_propositions,
            "causal_relations": normalised_relations,
        },
        relation_rejections,
        grounding_rejections,
        proposition_schema_rejections,
    )


def _validate_extraction_grounding_replay(
    telemetry: Mapping[str, Any],
    *,
    chunks: Mapping[str, Mapping[str, Any]],
    schema_retries: Optional[int] = None,
    transport_attempts_per_schema_attempt: Optional[int] = None,
) -> Dict[str, Any]:


    scientific = telemetry.get("scientific")
    operational = telemetry.get("operational")
    if not isinstance(scientific, Mapping) or not isinstance(operational, Mapping):
        raise GraphBundleValidationError("grounding replay requires scientific/operational telemetry")
    traces = operational.get("extraction_traces")
    if not isinstance(traces, list):
        raise GraphBundleValidationError("grounding replay requires extraction traces")
    relation_ledger = operational.get("relation_rejection_ledger", [])
    grounding_ledger = operational.get("grounding_rejection_ledger", [])
    proposition_schema_ledger = operational.get(
        "proposition_schema_rejection_ledger", []
    )
    if not isinstance(relation_ledger, list):
        raise GraphBundleValidationError("grounding replay requires a relation rejection ledger")
    if not isinstance(grounding_ledger, list):
        raise GraphBundleValidationError("grounding replay requires a grounding rejection ledger")
    if not isinstance(proposition_schema_ledger, list):
        raise GraphBundleValidationError(
            "grounding replay requires a proposition schema rejection ledger"
        )
    relation_ledger_by_chunk: Dict[str, list[Mapping[str, Any]]] = {}
    for entry in relation_ledger:
        if not isinstance(entry, Mapping):
            raise GraphBundleValidationError("grounding replay relation ledger entry is invalid")
        chunk_id = entry.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise GraphBundleValidationError("grounding replay relation ledger identity is invalid")
        relation_ledger_by_chunk.setdefault(chunk_id, []).append(
            {key: value for key, value in entry.items() if key != "chunk_id"}
        )
    grounding_ledger_by_chunk: Dict[str, list[Mapping[str, Any]]] = {}
    for entry in grounding_ledger:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("chunk_id"), str):
            raise GraphBundleValidationError("grounding replay grounding ledger entry is invalid")
        chunk_id = str(entry["chunk_id"])
        grounding_ledger_by_chunk.setdefault(chunk_id, []).append(
            {key: value for key, value in entry.items() if key != "chunk_id"}
        )
    proposition_schema_ledger_by_chunk: Dict[str, list[Mapping[str, Any]]] = {}
    for entry in proposition_schema_ledger:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("chunk_id"), str):
            raise GraphBundleValidationError(
                "grounding replay proposition schema ledger entry is invalid"
            )
        chunk_id = str(entry["chunk_id"])
        proposition_schema_ledger_by_chunk.setdefault(chunk_id, []).append(
            {key: value for key, value in entry.items() if key != "chunk_id"}
        )

    proposition_count = 0
    entity_count = 0
    accepted_relation_count = 0
    propositions_rejected = 0
    propositions_rejected_schema = 0
    entities_omitted = 0
    projected_empty_chunks = 0
    observed_chunks: set[str] = set()
    projected_by_chunk: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise GraphBundleValidationError("grounding replay trace entry is invalid")
        chunk_id = trace.get("chunk_id")
        attempts = trace.get("attempts")
        if not isinstance(chunk_id, str) or chunk_id not in chunks or not isinstance(attempts, list) or not attempts:
            raise GraphBundleValidationError("grounding replay trace/chunk closure mismatch")
        if chunk_id in observed_chunks:
            raise GraphBundleValidationError("grounding replay has duplicate chunk trace")
        observed_chunks.add(chunk_id)
        for ordinal, attempt_record in enumerate(attempts, start=1):
            if (
                not isinstance(attempt_record, Mapping)
                or attempt_record.get("attempt") != ordinal
            ):
                raise GraphBundleValidationError(
                    f"grounding replay attempt numbering mismatch for chunk {chunk_id}"
                )
        final_attempt = attempts[-1]
        if not isinstance(final_attempt, Mapping):
            raise GraphBundleValidationError("grounding replay final attempt is invalid")
        final_provider_trace = final_attempt.get("provider_trace")
        if isinstance(final_provider_trace, Mapping):
            _validate_json_object_provider_trace(
                final_provider_trace,
                context=f"grounding replay provider trace for {chunk_id}",
                max_transport_attempts=transport_attempts_per_schema_attempt,
            )
        final_provider_attempts = final_attempt.get("provider_attempts")
        if final_provider_attempts is not None:
            if not isinstance(final_provider_attempts, list) or not final_provider_attempts:
                raise GraphBundleValidationError(
                    f"grounding replay provider attempts are invalid for {chunk_id}"
                )
            if (
                transport_attempts_per_schema_attempt is not None
                and len(final_provider_attempts)
                > transport_attempts_per_schema_attempt
            ):
                raise GraphBundleValidationError(
                    f"grounding replay provider attempts exceed the sealed transport retry budget for {chunk_id}"
                )
        raw, provider_sha = _trace_raw_response(final_attempt)
        if raw is None:
            raise GraphBundleValidationError(
                f"grounding replay unavailable for chunk {chunk_id}"
            )
        actual_raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if not isinstance(provider_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", provider_sha):
            raise GraphBundleValidationError(
                f"grounding replay raw response hash is missing for chunk {chunk_id}"
            )
        if provider_sha != actual_raw_sha:
            raise GraphBundleValidationError(
                f"grounding replay raw response hash mismatch for chunk {chunk_id}"
            )
        outer_raw = final_attempt.get("raw_response")
        outer_raw_sha = final_attempt.get("raw_response_sha256")
        if not isinstance(outer_raw, str) or not outer_raw.strip():
            raise GraphBundleValidationError(
                f"grounding replay outer raw response is missing for chunk {chunk_id}"
            )
        if not isinstance(outer_raw_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", outer_raw_sha
        ):
            raise GraphBundleValidationError(
                f"grounding replay outer raw response hash is missing for chunk {chunk_id}"
            )
        if outer_raw != raw:
            raise GraphBundleValidationError(
                f"grounding replay outer/provider raw response mismatch for chunk {chunk_id}"
            )
        if outer_raw_sha != actual_raw_sha:
            raise GraphBundleValidationError(
                f"grounding replay outer raw response hash mismatch for chunk {chunk_id}"
            )
        payload = _strict_raw_json_loads(raw, context=f"raw extraction {chunk_id}")
        source_text = chunks[chunk_id].get("content")
        if not isinstance(source_text, str) or not source_text.strip():
            raise GraphBundleValidationError(f"grounding replay source text is missing for {chunk_id}")
        (
            projected,
            expected_relation_rejections,
            expected_grounding_rejections,
            expected_proposition_schema_rejections,
        ) = (
            _replay_project_extraction(
                payload,
                source_text=source_text,
                chunk_id=chunk_id,
            )
        )
        expected_projected_response_sha256 = canonical_sha256(projected)




        projected_by_chunk[chunk_id] = projected
        trace_relation_rejections = trace.get("relation_rejections", [])
        trace_grounding_rejections = trace.get("grounding_rejections", [])
        trace_proposition_schema_rejections = trace.get(
            "proposition_schema_rejections", []
        )
        if expected_grounding_rejections and not trace_grounding_rejections:
            raise GraphBundleValidationError(
                f"raw extraction entity is not grounded in chunk {chunk_id}"
            )
        if trace_relation_rejections != expected_relation_rejections:
            raise GraphBundleValidationError(
                f"grounding replay relation projection mismatch for chunk {chunk_id}"
            )
        if trace_grounding_rejections != expected_grounding_rejections:
            raise GraphBundleValidationError(
                f"grounding replay proposition projection mismatch for chunk {chunk_id}"
            )
        if (
            trace_proposition_schema_rejections
            != expected_proposition_schema_rejections
        ):
            raise GraphBundleValidationError(
                f"grounding replay proposition schema projection mismatch for chunk {chunk_id}"
            )

        has_relation_projection = bool(expected_relation_rejections)
        has_grounding_projection = bool(expected_grounding_rejections)
        has_proposition_schema_projection = bool(
            expected_proposition_schema_rejections
        )
        if has_grounding_projection or has_proposition_schema_projection:
            expected_status = (
                "success_projected_empty"
                if not projected["propositions"] and not projected["causal_relations"]
                else (
                    "success_with_proposition_schema_rejections"
                    if has_proposition_schema_projection
                    else "success_with_grounding_rejections"
                )
            )
        elif has_relation_projection:
            expected_status = "success_with_relation_rejections"
        elif projected["propositions"] or projected["causal_relations"]:
            expected_status = "success"
        else:
            expected_status = "success_explicit_empty"
        if trace.get("status") != expected_status:
            raise GraphBundleValidationError(
                f"grounding replay status mismatch for chunk {chunk_id}"
            )

        projection_fields = {
            "relation_admissibility_policy_version",
            "grounding_policy_version",
            "relation_rejections",
            "grounding_rejections",
            "proposition_schema_projection_policy_version",
            "proposition_schema_rejections",
            "projected_response_sha256",
        }
        projection_used = (
            has_relation_projection
            or has_grounding_projection
            or has_proposition_schema_projection
        )
        if projection_used:
            if schema_retries is not None and (
                len(attempts) != schema_retries + 1
                or final_attempt.get("attempt") != schema_retries + 1
            ):
                raise GraphBundleValidationError(
                    f"grounding projection did not use the sealed final retry for chunk {chunk_id}"
                )
            for prior_attempt in attempts[:-1]:
                if projection_fields.intersection(prior_attempt):
                    raise GraphBundleValidationError(
                        f"grounding projection evidence appeared before final retry for chunk {chunk_id}"
                    )
        else:
            if any(
                projection_fields.intersection(attempt_record)
                for attempt_record in attempts
            ) or any(field in trace for field in projection_fields):
                raise GraphBundleValidationError(
                    f"strict extraction trace contains projection evidence for chunk {chunk_id}"
                )
        if relation_ledger_by_chunk.get(chunk_id, []) != expected_relation_rejections:
            raise GraphBundleValidationError(
                f"grounding replay relation ledger mismatch for chunk {chunk_id}"
            )
        if grounding_ledger_by_chunk.get(chunk_id, []) != expected_grounding_rejections:
            raise GraphBundleValidationError(
                f"grounding replay grounding ledger mismatch for chunk {chunk_id}"
            )
        expected_proposition_schema_ledger = [
            {
                **rejection,
                "raw_response_sha256": outer_raw_sha,
                "projected_response_sha256": expected_projected_response_sha256,
                "proposition_schema_projection_policy_version": (
                    EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
                ),
            }
            for rejection in expected_proposition_schema_rejections
        ]
        if proposition_schema_ledger_by_chunk.get(
            chunk_id, []
        ) != expected_proposition_schema_ledger:
            raise GraphBundleValidationError(
                f"grounding replay proposition schema ledger mismatch for chunk {chunk_id}"
            )

        if projection_used:
            projected_hash = final_attempt.get("projected_response_sha256")
            if projected_hash != expected_projected_response_sha256:
                raise GraphBundleValidationError(
                    f"grounding replay projected response hash mismatch for chunk {chunk_id}"
                )
        proposition_count += len(projected["propositions"])
        entity_count += sum(
            len(proposition["entities"])
            for proposition in projected["propositions"]
        )
        accepted_relation_count += len(projected["causal_relations"])
        propositions_rejected += len(expected_grounding_rejections)
        propositions_rejected_schema += len(
            expected_proposition_schema_rejections
        )
        entities_omitted += sum(
            int(entry["entity_count"]) for entry in expected_grounding_rejections
        )
        if (
            (expected_grounding_rejections or expected_proposition_schema_rejections)
            and not projected["propositions"]
            and not projected["causal_relations"]
        ):
            projected_empty_chunks += 1

        outer_response_sha = final_attempt.get("response_sha256")
        if (
            not isinstance(outer_response_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", outer_response_sha)
            or outer_response_sha != canonical_sha256(payload)
        ):
            raise GraphBundleValidationError(
                f"grounding replay normalized response hash mismatch for chunk {chunk_id}"
            )

    if observed_chunks != set(chunks):
        raise GraphBundleValidationError("grounding replay does not close over all chunks")
    summary = {
        "status": "passed",
        "chunks": len(observed_chunks),
        "propositions": proposition_count,
        "entities": entity_count,
        "causal_relations": accepted_relation_count,
        "propositions_admitted": proposition_count,
        "entities_admitted": entity_count,
        "causal_relations_admitted": accepted_relation_count,
        "propositions_rejected_grounding": propositions_rejected,
        "propositions_rejected_schema": propositions_rejected_schema,
        "entities_omitted_grounding": entities_omitted,
        "projected_empty_chunks": projected_empty_chunks,
    }
    declared = scientific.get("grounding_replay")
    if not isinstance(declared, Mapping) or declared.get("status") != "passed":
        raise GraphBundleValidationError("grounding replay summary status is invalid")
    for field in (
        "chunks",
        "propositions",
        "entities",
        "causal_relations",
        "propositions_admitted",
        "entities_admitted",
        "causal_relations_admitted",
        "propositions_rejected_grounding",
        "propositions_rejected_schema",
        "entities_omitted_grounding",
        "projected_empty_chunks",
    ):
        value = declared.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GraphBundleValidationError(
                f"grounding replay summary counter {field} is invalid"
            )
    if declared != summary:
        raise GraphBundleValidationError("grounding replay summary mismatch")
    return {**summary, "projected_by_chunk": projected_by_chunk}


def _validate_graph_source_grounding(
    graph: Any,
    *,
    chunks: Mapping[str, Mapping[str, Any]],
    entity_nodes: set[Any],
    hyperedge_nodes: set[Any],
) -> Tuple[set[Tuple[str, str, str]], set[Tuple[str, str]]]:


    incidence_occurrences: set[Tuple[str, str, str]] = set()
    entity_chunks: set[Tuple[str, str]] = set()
    for node_id in entity_nodes:
        attrs = graph.nodes[node_id]
        aliases = _json_attr(attrs.get("aliases"), field=f"entity {node_id} aliases")
        provenance = _json_attr(
            attrs.get("alias_provenance"),
            field=f"entity {node_id} alias_provenance",
        )
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
            or len(set(aliases)) != len(aliases)
            or not isinstance(provenance, Mapping)
            or set(provenance) != set(aliases)
        ):
            raise GraphBundleValidationError(
                f"entity {node_id!r} has invalid alias grounding metadata"
            )
        for alias in aliases:
            entries = provenance.get(alias)
            if not isinstance(entries, list) or not entries:
                raise GraphBundleValidationError(
                    f"entity {node_id!r} alias {alias!r} lacks provenance"
                )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise GraphBundleValidationError(
                        f"entity {node_id!r} alias provenance entry is invalid"
                    )
                chunk_id = entry.get("chunk_id")
                hyperedge_id = entry.get("hyperedge_id")
                if (
                    not isinstance(chunk_id, str)
                    or chunk_id not in chunks
                    or not isinstance(hyperedge_id, str)
                    or hyperedge_id not in hyperedge_nodes
                    or graph.nodes[hyperedge_id].get("chunk_id") != chunk_id
                ):
                    raise GraphBundleValidationError(
                        f"entity {node_id!r} alias provenance is not graph/chunk closed"
                    )
                if not _grounded_exact(alias, chunks[chunk_id].get("content")):
                    raise GraphBundleValidationError(
                        f"entity {node_id!r} alias {alias!r} is not grounded in chunk {chunk_id}"
                    )
                incidence_occurrences.add(
                    (str(node_id), hyperedge_id, chunk_id)
                )
                entity_chunks.add((str(node_id), chunk_id))
    return incidence_occurrences, entity_chunks


def _stable_edge_id(prefix: str, payload: Mapping[str, Any]) -> str:


    return prefix + canonical_sha256(payload)


def _expected_replay_graph_closure(
    projected_by_chunk: Mapping[str, Mapping[str, Any]],
    *,
    chunks: Mapping[str, Mapping[str, Any]],
    graph_namespace: str,
) -> Dict[str, Any]:


    hyperedges: Dict[str, Dict[str, Any]] = {}
    entity_accumulator: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    hyperedge_text: Dict[str, str] = {}
    hyperedge_source_ids: Dict[str, str] = {}

    if set(projected_by_chunk) != set(chunks):
        raise GraphBundleValidationError(
            "replayed extractions do not close over persisted chunks"
        )

    for chunk_id, chunk in chunks.items():
        extraction = projected_by_chunk.get(chunk_id)
        if not isinstance(extraction, Mapping):
            raise GraphBundleValidationError(
                f"replayed extraction is missing for chunk {chunk_id}"
            )
        propositions = extraction.get("propositions")
        relations = extraction.get("causal_relations")
        if not isinstance(propositions, list) or not isinstance(relations, list):
            raise GraphBundleValidationError(
                f"replayed extraction arrays are invalid for chunk {chunk_id}"
            )

        chunk_entity_ids: Dict[str, str] = {}
        chunk_hyperedge_ids: list[str] = []
        for proposition in propositions:
            if not isinstance(proposition, Mapping):
                raise GraphBundleValidationError(
                    f"replayed proposition is invalid for chunk {chunk_id}"
                )
            proposition_index = proposition.get("source_proposition_index")
            proposition_key = proposition.get("proposition_key")
            sentence = proposition.get("sentence")
            entities = proposition.get("entities")
            if (
                isinstance(proposition_index, bool)
                or not isinstance(proposition_index, int)
                or proposition_index < 0
                or not isinstance(proposition_key, str)
                or not proposition_key
                or not isinstance(sentence, str)
                or not sentence
                or not isinstance(entities, list)
                or not entities
            ):
                raise GraphBundleValidationError(
                    f"replayed proposition identity is invalid for chunk {chunk_id}"
                )
            hyperedge_id = "hyp_" + canonical_sha256(
                {
                    "graph_namespace": graph_namespace,
                    "chunk_id": chunk_id,
                    "proposition_index": proposition_index,
                    "proposition_key": proposition_key,
                }
            )
            if hyperedge_id in hyperedges:
                raise GraphBundleValidationError(
                    f"replayed hyperedge identity is duplicated: {hyperedge_id}"
                )
            chunk_hyperedge_ids.append(hyperedge_id)
            hyperedge_text[hyperedge_id] = sentence
            hyperedge_source_ids[hyperedge_id] = chunk_id
            hyperedges[hyperedge_id] = {
                "role": "hyperedge",
                "name": sentence,
                "proposition_key": proposition_key,
                "source_id": chunk_id,
                "chunk_id": chunk_id,
                "record_id": chunk["record_id"],
                "record_chunk_index": chunk["record_chunk_index"],
                "proposition_index": proposition_index,
                "source": chunk["source"],
                "source_path": chunk["source_path"],
                "timestamp_raw": chunk["timestamp_raw"],
                "timestamp_iso": chunk["timestamp_iso"],
                "date": chunk["date"],
                "availability_upper_bound_iso": chunk[
                    "availability_upper_bound_iso"
                ],
                "availability_upper_bound_date": chunk[
                    "availability_upper_bound_date"
                ],
                "availability_bound_kind": chunk["availability_bound_kind"],
                "availability_bound_source": chunk["availability_bound_source"],
                "availability_bound_source_sha256": chunk[
                    "availability_bound_source_sha256"
                ],
                "semantic_relevance": 0.5,
            }

            proposition_entity_keys: set[str] = set()
            for entity in entities:
                if not isinstance(entity, Mapping):
                    raise GraphBundleValidationError(
                        f"replayed entity is invalid for chunk {chunk_id}"
                    )
                entity_key = entity.get("canonical_key")
                alias = entity.get("name")
                if (
                    not isinstance(entity_key, str)
                    or not entity_key
                    or not isinstance(alias, str)
                    or not alias
                    or entity_key in proposition_entity_keys
                ):
                    raise GraphBundleValidationError(
                        f"replayed entity identity is invalid for chunk {chunk_id}"
                    )
                proposition_entity_keys.add(entity_key)
                entity_id = "ent_" + canonical_sha256(
                    {
                        "graph_namespace": graph_namespace,
                        "canonical_entity_key": entity_key,
                    }
                )
                chunk_entity_ids[entity_key] = entity_id
                accumulator = entity_accumulator.setdefault(
                    entity_id,
                    {"canonical_entity_key": entity_key, "aliases": {}},
                )
                if accumulator["canonical_entity_key"] != entity_key:
                    raise GraphBundleValidationError(
                        f"replayed entity identity collision for {entity_id}"
                    )
                provenance = {
                    "hyperedge_id": hyperedge_id,
                    "chunk_id": chunk_id,
                    "record_id": chunk["record_id"],
                    "source": chunk["source"],
                    "timestamp_iso": chunk["timestamp_iso"],
                    "date": chunk["date"],
                    "availability_upper_bound_iso": chunk[
                        "availability_upper_bound_iso"
                    ],
                }
                accumulator["aliases"].setdefault(alias, []).append(provenance)
                incidence_id = _stable_edge_id(
                    "inc_",
                    {
                        "graph_namespace": graph_namespace,
                        "hyperedge_id": hyperedge_id,
                        "entity_id": entity_id,
                        "incidence_role": "mentions",
                    },
                )
                edge_identity = (hyperedge_id, entity_id, incidence_id)
                if edge_identity in edges:
                    raise GraphBundleValidationError(
                        f"replayed incidence identity is duplicated: {incidence_id}"
                    )
                edges[edge_identity] = {
                    "role": "incidence",
                    "incidence_role": "mentions",
                    "chunk_id": chunk_id,
                    "record_id": chunk["record_id"],
                    "source": chunk["source"],
                    "edge_id": incidence_id,
                }

        for relation in relations:
            if not isinstance(relation, Mapping):
                raise GraphBundleValidationError(
                    f"replayed relation is invalid for chunk {chunk_id}"
                )
            relation_index = relation.get("source_relation_index")
            cause_key = relation.get("cause_key")
            effect_key = relation.get("effect_key")
            relation_type = relation.get("type")
            strength = relation.get("strength")
            description = relation.get("description")
            if (
                isinstance(relation_index, bool)
                or not isinstance(relation_index, int)
                or relation_index < 0
                or not isinstance(cause_key, str)
                or not isinstance(effect_key, str)
                or not isinstance(relation_type, str)
                or relation_type not in ALLOWED_CAUSAL_TYPES
                or isinstance(strength, bool)
                or not isinstance(strength, (int, float))
                or not isinstance(description, str)
            ):
                raise GraphBundleValidationError(
                    f"replayed relation identity is invalid for chunk {chunk_id}"
                )
            cause_id = chunk_entity_ids.get(cause_key)
            effect_id = chunk_entity_ids.get(effect_key)
            if not cause_id or not effect_id or cause_id == effect_id:
                raise GraphBundleValidationError(
                    f"replayed relation lacks entity closure for chunk {chunk_id}"
                )
            causal_id = _stable_edge_id(
                "cedge_",
                {
                    "graph_namespace": graph_namespace,
                    "chunk_id": chunk_id,
                    "relation_index": relation_index,
                    "cause_entity_id": cause_id,
                    "effect_entity_id": effect_id,
                    "causal_type": relation_type,
                },
            )
            edge_identity = (cause_id, effect_id, causal_id)
            if edge_identity in edges:
                raise GraphBundleValidationError(
                    f"replayed causal identity is duplicated: {causal_id}"
                )
            edges[edge_identity] = {
                "role": "causal",
                "causal_type": relation_type,
                "strength": float(strength),
                "description": description,
                "source_id": chunk_id,
                "chunk_id": chunk_id,
                "record_id": chunk["record_id"],
                "source_hyperedge_ids": list(chunk_hyperedge_ids),
                "relation_index": relation_index,
                "source": chunk["source"],
                "source_path": chunk["source_path"],
                "timestamp_raw": chunk["timestamp_raw"],
                "timestamp_iso": chunk["timestamp_iso"],
                "date": chunk["date"],
                "availability_upper_bound_iso": chunk[
                    "availability_upper_bound_iso"
                ],
                "availability_upper_bound_date": chunk[
                    "availability_upper_bound_date"
                ],
                "availability_bound_kind": chunk["availability_bound_kind"],
                "availability_bound_source": chunk["availability_bound_source"],
                "availability_bound_source_sha256": chunk[
                    "availability_bound_source_sha256"
                ],
                "edge_id": causal_id,
            }

    entities: Dict[str, Dict[str, Any]] = {}
    for entity_id, accumulator in entity_accumulator.items():
        aliases = sorted(
            accumulator["aliases"],
            key=lambda value: (_canonical_extraction_key(value, context="alias"), value),
        )
        provenance = {
            alias: sorted(
                accumulator["aliases"][alias],
                key=lambda item: (
                    item["record_id"],
                    item["chunk_id"],
                    item["hyperedge_id"],
                ),
            )
            for alias in aliases
        }
        entities[entity_id] = {
            "role": "entity",
            "name": aliases[0],
            "canonical_entity_key": accumulator["canonical_entity_key"],
            "aliases": aliases,
            "alias_provenance": provenance,
        }

    return {
        "entities": entities,
        "hyperedges": hyperedges,
        "edges": edges,
        "entity_text": {
            node_id: attrs["canonical_entity_key"]
            for node_id, attrs in entities.items()
        },
        "hyperedge_text": hyperedge_text,
        "hyperedge_source_ids": hyperedge_source_ids,
    }


def _compare_replay_graph_attr(
    actual: Any,
    expected: Any,
    *,
    field: str,
) -> None:



    if expected is None:
        expected = ""
    if isinstance(expected, (list, dict)):
        actual = _json_attr(actual, field=field)
    if actual != expected:
        raise GraphBundleValidationError(
            f"GraphML/replay mismatch for {field}"
        )


def _validate_replay_graph_closure(
    graph: Any,
    *,
    expected: Mapping[str, Any],
) -> None:


    expected_nodes = {
        **dict(expected["entities"]),
        **dict(expected["hyperedges"]),
    }
    actual_node_ids = {str(node_id) for node_id in graph.nodes}
    if actual_node_ids != set(expected_nodes):
        raise GraphBundleValidationError(
            "GraphML/replay node closure mismatch"
        )
    for node_id, expected_attrs in expected_nodes.items():
        actual_attrs = graph.nodes[node_id]
        if set(actual_attrs) != set(expected_attrs):
            raise GraphBundleValidationError(
                f"GraphML/replay node fields mismatch for {node_id!r}"
            )
        for field, expected_value in expected_attrs.items():
            _compare_replay_graph_attr(
                actual_attrs.get(field),
                expected_value,
                field=f"node {node_id}.{field}",
            )

    expected_edges = expected["edges"]
    actual_edges = {
        (str(source), str(target), str(key)): dict(attrs)
        for source, target, key, attrs in graph.edges(keys=True, data=True)
    }
    if set(actual_edges) != set(expected_edges):
        raise GraphBundleValidationError(
            "GraphML/replay edge closure mismatch"
        )
    for identity, expected_attrs in expected_edges.items():
        actual_attrs = actual_edges[identity]
        if set(actual_attrs) != set(expected_attrs):
            raise GraphBundleValidationError(
                f"GraphML/replay edge fields mismatch for {identity[2]!r}"
            )
        for field, expected_value in expected_attrs.items():
            _compare_replay_graph_attr(
                actual_attrs.get(field),
                expected_value,
                field=f"edge {identity[2]}.{field}",
            )


def _validate_replay_vdb_closure(
    records: Mapping[str, Mapping[str, Any]],
    *,
    expected_text: Mapping[str, str],
    expected_source_ids: Optional[Mapping[str, str]] = None,
    embedding_signature: Mapping[str, Any],
    role: str,
) -> None:


    if role not in {"entity", "hyperedge_occurrence"}:
        raise GraphBundleValidationError("invalid replay VDB role")
    expected_records: Dict[str, Dict[str, Any]] = {}
    for node_id, content in expected_text.items():
        vector_id = "vec_" + canonical_sha256(
            {
                "role": role,
                "node_id": node_id,
                "text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "embedding_signature": embedding_signature,
            }
        )
        if role == "entity":
            expected_record = {
                "__id__": vector_id,
                "node_id": node_id,
                "entity_name": node_id,
                "canonical_entity_key": content,
                "content": content,
            }
        else:
            if expected_source_ids is None or node_id not in expected_source_ids:
                raise GraphBundleValidationError(
                    "hyperedge VDB replay source closure is unavailable"
                )
            expected_record = {
                "__id__": vector_id,
                "node_id": node_id,
                "hyperedge_name": node_id,
                "content": content,
                "source_id": expected_source_ids[node_id],
            }
        expected_records[vector_id] = expected_record

    if set(records) != set(expected_records):
        raise GraphBundleValidationError(
            f"{role} VDB/replay vector identity closure mismatch"
        )
    for vector_id, expected_record in expected_records.items():
        actual_record = records[vector_id]
        if canonical_json(actual_record) != canonical_json(expected_record):
            raise GraphBundleValidationError(
                f"{role} VDB/replay metadata mismatch for {vector_id!r}"
            )


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_canonical_json(path: os.PathLike[str] | str, value: Any) -> None:


    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    with target.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _strict_object_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraphBundleValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    GraphBundleValidationError(
                        f"non-finite JSON constant {value!r} in {path.name}"
                    )
                ),
            )
    except GraphBundleValidationError:
        raise
    except Exception as exc:
        raise GraphBundleValidationError(f"cannot parse {path.name}: {exc}") from exc


def _json_attr(value: Any, *, field: str) -> Any:
    if not isinstance(value, str) or not value.startswith("json:"):
        raise GraphBundleValidationError(f"GraphML attribute {field!r} is not canonical JSON")
    try:
        parsed = json.loads(
            value[5:],
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                GraphBundleValidationError(
                    f"non-finite GraphML JSON constant {constant!r} in {field}"
                )
            ),
        )
    except Exception as exc:
        raise GraphBundleValidationError(f"invalid canonical GraphML attribute {field!r}: {exc}") from exc
    if "json:" + canonical_json(parsed) != value:
        raise GraphBundleValidationError(f"non-canonical GraphML attribute {field!r}")
    return parsed


def _contains_raw_only_key(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            folded = str(key).casefold()
            if folded in _RAW_ONLY_KEYS or folded.startswith("raw_record") or folded.startswith("raw_input"):
                return str(key)
            found = _contains_raw_only_key(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_raw_only_key(item)
            if found:
                return found
    return None


def _reject_raw_only_keys(value: Any, *, context: str) -> None:
    found = _contains_raw_only_key(value)
    if found:
        raise GraphBundleValidationError(
            f"raw/label field {found!r} leaked into scientific {context}"
        )


def _require_fields(value: Mapping[str, Any], fields: Iterable[str], *, context: str) -> None:
    missing = sorted(set(fields) - set(value))
    if missing:
        raise GraphBundleValidationError(f"{context} missing required fields: {missing}")


def _reject_graph_attr_leakage(attrs: Mapping[str, Any], *, context: str) -> None:
    _reject_raw_only_keys(attrs, context=context)
    for field, value in attrs.items():
        if isinstance(value, str) and value.startswith("json:"):
            parsed = _json_attr(value, field=str(field))
            if field == "alias_provenance" and isinstance(parsed, Mapping):





                for alias, supports in parsed.items():
                    _reject_raw_only_keys(
                        supports,
                        context=f"{context}.{field}[{alias!r}]",
                    )
                continue
            _reject_raw_only_keys(parsed, context=f"{context}.{field}")


def _manifest_self_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "graph_bundle_sha256",
        "manifest_payload_sha256",
        "file_sha256",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "duration_seconds",
        "graph_dir",
        "scratch_dir",
        "source_audit",
        "source_audit_manifest",


        "scientific_config_sha256",



    }
    return {key: value for key, value in manifest.items() if key not in excluded}


def _telemetry_self_payload(telemetry: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "graph_bundle_sha256",
        "telemetry_payload_sha256",
        "file_sha256",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "duration_seconds",
        "graph_dir",
        "scratch_dir",
        "source_audit",
        "source_audit_manifest",
        "scientific_config_sha256",


        "operational",
    }
    return {key: value for key, value in telemetry.items() if key not in excluded}


def graph_bundle_payload(
    manifest: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    artifact_info: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:


    return {
        "schema_version": GRAPH_BUNDLE_SCHEMA,
        "artifacts": {
            name: {
                "sha256": artifact_info[name]["sha256"],
                "bytes": int(artifact_info[name]["bytes"]),
            }
            for name in KG_ARTIFACTS
        },
        "canonical_construction_stream_sha256": manifest[
            "canonical_construction_stream_sha256"
        ],
        "construction_fingerprint_sha256": manifest[
            "construction_fingerprint_sha256"
        ],
        "base_graph_namespace": manifest["base_graph_namespace"],
        "profile_id": manifest["profile_id"],
        "construction_config_sha256": manifest["construction_config_sha256"],
        "resolved_construction_config": manifest["resolved_construction_config"],
        "manifest_payload_sha256": canonical_sha256(_manifest_self_payload(manifest)),
        "telemetry_payload_sha256": canonical_sha256(_telemetry_self_payload(telemetry)),
    }


def _validate_relation_rejection_telemetry(
    telemetry: Mapping[str, Any],
    *,
    chunk_ids: set[str],
    schema_retries: Optional[int] = None,
    transport_attempts_per_schema_attempt: Optional[int] = None,
) -> None:
    scientific = telemetry["scientific"]
    operational = telemetry["operational"]
    traces = operational.get("extraction_traces")
    declared_relation_ledger = operational.get("relation_rejection_ledger")
    declared_grounding_ledger = operational.get("grounding_rejection_ledger")
    declared_proposition_schema_ledger = operational.get(
        "proposition_schema_rejection_ledger"
    )
    if (
        not isinstance(traces, list)
        or not isinstance(declared_relation_ledger, list)
        or not isinstance(declared_grounding_ledger, list)
        or not isinstance(declared_proposition_schema_ledger, list)
    ):
        raise GraphBundleValidationError(
            "operational extraction traces/rejection ledgers must be arrays"
        )
    if len(traces) != len(chunk_ids):
        raise GraphBundleValidationError("extraction trace count does not match chunks")
    if schema_retries is not None and (
        isinstance(schema_retries, bool)
        or not isinstance(schema_retries, int)
        or not 0 <= schema_retries <= 2
    ):
        raise GraphBundleValidationError("invalid extraction schema retry budget")
    if transport_attempts_per_schema_attempt is not None and (
        isinstance(transport_attempts_per_schema_attempt, bool)
        or not isinstance(transport_attempts_per_schema_attempt, int)
        or transport_attempts_per_schema_attempt <= 0
    ):
        raise GraphBundleValidationError(
            "invalid extraction transport retry budget"
        )

    allowed_statuses = {
        "success",
        "success_explicit_empty",
        "success_with_relation_rejections",
        "success_with_grounding_rejections",
        "success_with_proposition_schema_rejections",
        "success_projected_empty",
    }
    allowed_reasons = {
        "invalid_relation_fields",
        "invalid_endpoint_value",
        "endpoint_not_in_entity_vocabulary",
        "canonical_self_loop",
        "unsupported_causal_type",
        "invalid_strength_type",
        "invalid_strength_range",
        "invalid_description",
        "duplicate_typed_relation",
        "endpoint_not_grounded_in_chunk",
    }
    allowed_proposition_schema_reasons = {
        "proposition_entities_empty",
        "proposition_duplicate_canonical_entity",
    }
    relation_policy = EXPECTED_RELATION_ADMISSIBILITY_POLICY_VERSION
    grounding_policy = EXPECTED_GROUNDING_POLICY_VERSION
    proposition_schema_policy = (
        EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
    )
    observed_chunks: set[str] = set()
    reconstructed_relation_ledger: list[Dict[str, Any]] = []
    reconstructed_grounding_ledger: list[Dict[str, Any]] = []
    reconstructed_proposition_schema_ledger: list[Dict[str, Any]] = []
    attempts_total = 0
    nonempty_success = 0
    explicit_empty_success = 0
    projected_empty_success = 0
    success_with_relation_rejections = 0
    success_with_grounding_rejections = 0
    success_with_proposition_schema_rejections = 0

    for trace in traces:
        if not isinstance(trace, Mapping):
            raise GraphBundleValidationError("extraction trace entries must be objects")
        chunk_id = trace.get("chunk_id")
        status = trace.get("status")
        attempts = trace.get("attempts")
        if not isinstance(chunk_id, str) or chunk_id not in chunk_ids or chunk_id in observed_chunks:
            raise GraphBundleValidationError("extraction traces do not close uniquely over chunks")
        observed_chunks.add(chunk_id)
        if status not in allowed_statuses or not isinstance(attempts, list) or not attempts:
            raise GraphBundleValidationError("extraction trace has invalid status/attempts")
        for ordinal, attempt_record in enumerate(attempts, start=1):
            if (
                not isinstance(attempt_record, Mapping)
                or attempt_record.get("attempt") != ordinal
            ):
                raise GraphBundleValidationError(
                    "extraction trace attempts are not canonically numbered"
                )
            if (
                attempt_record.get("response_format") != {"type": "json_object"}
                or attempt_record.get("response_format_version")
                != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
            ):
                raise GraphBundleValidationError(
                    "extraction trace lacks the sealed response-format protocol"
                )
            provider_trace = attempt_record.get("provider_trace")
            if isinstance(provider_trace, Mapping):
                _validate_json_object_provider_trace(
                    provider_trace,
                    context=f"extraction provider trace for {chunk_id} attempt {ordinal}",
                    max_transport_attempts=transport_attempts_per_schema_attempt,
                )
                trace_identity = attempt_record.get("trace_identity")
                if (
                    not isinstance(trace_identity, Mapping)
                    or trace_identity.get("response_format") != "json_object"
                    or trace_identity.get("response_format_version")
                    != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
                ):
                    raise GraphBundleValidationError(
                        "extraction trace identity lacks the sealed response format"
                    )
                if trace_identity.get("attempt_count") != provider_trace.get(
                    "attempt_count"
                ):
                    raise GraphBundleValidationError(
                        "extraction trace identity transport attempt count mismatch"
                    )
            provider_attempts = attempt_record.get("provider_attempts")
            if provider_attempts is not None:
                if (
                    not isinstance(provider_attempts, list)
                    or not provider_attempts
                    or any(
                        not isinstance(item, Mapping)
                        or item.get("response_format") != "json_object"
                        or item.get("response_format_version")
                        != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
                        for item in provider_attempts
                    )
                ):
                    raise GraphBundleValidationError(
                        "extraction provider attempts do not attest the sealed response format"
                    )
                if (
                    transport_attempts_per_schema_attempt is not None
                    and len(provider_attempts)
                    > transport_attempts_per_schema_attempt
                ):
                    raise GraphBundleValidationError(
                        "extraction provider attempts exceed the sealed transport retry budget"
                    )
                if isinstance(provider_trace, Mapping) and provider_attempts != (
                    provider_trace.get("attempts")
                ):
                    raise GraphBundleValidationError(
                        "extraction provider attempt arrays do not close"
                    )
        attempts_total += len(attempts)
        relation_rejections = trace.get("relation_rejections", [])
        grounding_rejections = trace.get("grounding_rejections", [])
        proposition_schema_rejections = trace.get(
            "proposition_schema_rejections", []
        )
        if not isinstance(relation_rejections, list):
            raise GraphBundleValidationError("trace relation_rejections must be an array")
        if not isinstance(grounding_rejections, list):
            raise GraphBundleValidationError("trace grounding_rejections must be an array")
        if not isinstance(proposition_schema_rejections, list):
            raise GraphBundleValidationError(
                "trace proposition_schema_rejections must be an array"
            )
        relation_projected = bool(relation_rejections)
        grounding_projected = bool(grounding_rejections)
        proposition_schema_projected = bool(proposition_schema_rejections)
        if status == "success" and (
            relation_projected or grounding_projected or proposition_schema_projected
        ):
            raise GraphBundleValidationError("strict-success trace contains projection evidence")
        if status == "success_explicit_empty" and (
            relation_projected or grounding_projected or proposition_schema_projected
        ):
            raise GraphBundleValidationError(
                "explicit-empty trace contains projection evidence"
            )
        if status == "success_with_relation_rejections" and (
            not relation_projected or grounding_projected or proposition_schema_projected
        ):
            raise GraphBundleValidationError(
                "trace status/relation rejection mismatch"
            )
        if status == "success_with_grounding_rejections" and (
            not grounding_projected or proposition_schema_projected
        ):
            raise GraphBundleValidationError(
                "trace status/grounding rejection mismatch"
            )
        if status == "success_with_proposition_schema_rejections" and (
            not proposition_schema_projected
        ):
            raise GraphBundleValidationError(
                "trace status/proposition schema rejection mismatch"
            )
        if status == "success_projected_empty" and not (
            grounding_projected or proposition_schema_projected
        ):
            raise GraphBundleValidationError(
                "projected-empty trace lacks proposition deletion evidence"
            )
        if status in {
            "success",
            "success_with_relation_rejections",
            "success_with_grounding_rejections",
            "success_with_proposition_schema_rejections",
        }:
            nonempty_success += 1
        elif status == "success_explicit_empty":
            explicit_empty_success += 1
        else:
            projected_empty_success += 1
        if relation_projected:
            success_with_relation_rejections += 1
        if status == "success_with_grounding_rejections":
            success_with_grounding_rejections += 1
        if status == "success_with_proposition_schema_rejections":
            success_with_proposition_schema_rejections += 1
        projection_used = (
            relation_projected or grounding_projected or proposition_schema_projected
        )
        if projection_used:
            if schema_retries is not None and (
                len(attempts) != schema_retries + 1
                or attempts[-1].get("attempt") != schema_retries + 1
            ):
                raise GraphBundleValidationError(
                    "projected extraction did not consume the sealed final retry"
                )
            final_attempt = attempts[-1]
            if not isinstance(final_attempt, Mapping):
                raise GraphBundleValidationError("final extraction attempt must be an object")
            if final_attempt.get("relation_admissibility_policy_version") != relation_policy:
                raise GraphBundleValidationError("extraction trace relation policy mismatch")
            if final_attempt.get("grounding_policy_version") != grounding_policy:
                raise GraphBundleValidationError("extraction trace grounding policy mismatch")
            if trace.get("grounding_policy_version") != grounding_policy:
                raise GraphBundleValidationError("trace grounding policy mismatch")
            if (
                final_attempt.get(
                    "proposition_schema_projection_policy_version"
                )
                != proposition_schema_policy
                or trace.get("proposition_schema_projection_policy_version")
                != proposition_schema_policy
            ):
                raise GraphBundleValidationError(
                    "extraction trace proposition schema policy mismatch"
                )
            if final_attempt.get("relation_rejections") != relation_rejections:
                raise GraphBundleValidationError("trace/final-attempt relation rejection mismatch")
            if final_attempt.get("grounding_rejections") != grounding_rejections:
                raise GraphBundleValidationError(
                    "trace/final-attempt grounding rejection mismatch"
                )
            if (
                final_attempt.get("proposition_schema_rejections")
                != proposition_schema_rejections
            ):
                raise GraphBundleValidationError(
                    "trace/final-attempt proposition schema rejection mismatch"
                )
            if trace.get("projected_response_sha256") != final_attempt.get(
                "projected_response_sha256"
            ):
                raise GraphBundleValidationError(
                    "trace/final-attempt projected response hash mismatch"
                )
            for hash_field in (
                "response_sha256",
                "raw_response_sha256",
                "projected_response_sha256",
            ):
                value = final_attempt.get(hash_field)
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise GraphBundleValidationError(
                        f"projected extraction lacks valid {hash_field}"
                    )
        for rejection in relation_rejections:
            if not isinstance(rejection, Mapping):
                raise GraphBundleValidationError("relation rejection entries must be objects")
            expected_fields = {
                "relation_index",
                "reason_code",
                "relation_sha256",
            }
            if "missing_endpoint_roles" in rejection:
                expected_fields.add("missing_endpoint_roles")
            if set(rejection) != expected_fields:
                raise GraphBundleValidationError(
                    "relation rejection audit fields do not match sealed schema"
                )
            index = rejection.get("relation_index")
            reason = rejection.get("reason_code")
            relation_hash = rejection.get("relation_sha256")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or reason not in allowed_reasons
                or not isinstance(relation_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", relation_hash)
            ):
                raise GraphBundleValidationError("invalid relation rejection audit entry")
            missing_roles = rejection.get("missing_endpoint_roles")
            if missing_roles is not None and (
                not isinstance(missing_roles, list)
                or not missing_roles
                or any(role not in {"cause", "effect"} for role in missing_roles)
                or len(set(missing_roles)) != len(missing_roles)
            ):
                raise GraphBundleValidationError("invalid missing endpoint role audit")
            reconstructed_relation_ledger.append(
                {"chunk_id": chunk_id, **dict(rejection)}
            )

        for rejection in grounding_rejections:
            if not isinstance(rejection, Mapping):
                raise GraphBundleValidationError(
                    "grounding rejection audit fields do not match sealed schema"
                )
            reason = rejection.get("reason_code")
            expected_fields = (
                {
                    "proposition_index",
                    "reason_code",
                    "proposition_sha256",
                    "ungrounded_entity_indices",
                    "entity_count",
                }
                if reason == "proposition_contains_ungrounded_entity"
                else {
                    "proposition_index",
                    "reason_code",
                    "proposition_sha256",
                    "unsafe_text_fields",
                    "entity_count",
                }
                if reason == "proposition_contains_xml_unsafe_text"
                else set()
            )
            if not expected_fields or set(rejection) != expected_fields:
                raise GraphBundleValidationError(
                    "grounding rejection audit fields do not match sealed schema"
                )
            proposition_index = rejection.get("proposition_index")
            proposition_hash = rejection.get("proposition_sha256")
            entity_count = rejection.get("entity_count")
            if (
                isinstance(proposition_index, bool)
                or not isinstance(proposition_index, int)
                or proposition_index < 0
                or not isinstance(proposition_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", proposition_hash)
                or isinstance(entity_count, bool)
                or not isinstance(entity_count, int)
                or entity_count <= 0
            ):
                raise GraphBundleValidationError(
                    "invalid grounding rejection audit entry"
                )
            if reason == "proposition_contains_ungrounded_entity":
                entity_indices = rejection.get("ungrounded_entity_indices")
                if (
                    not isinstance(entity_indices, list)
                    or not entity_indices
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index < 0
                        or index >= entity_count
                        for index in entity_indices
                    )
                    or entity_indices != sorted(set(entity_indices))
                ):
                    raise GraphBundleValidationError(
                        "invalid grounding rejection audit entry"
                    )
            else:
                unsafe_text_fields = rejection.get("unsafe_text_fields")
                if (
                    not isinstance(unsafe_text_fields, list)
                    or not unsafe_text_fields
                    or any(
                        not isinstance(field, str) or not field.strip()
                        for field in unsafe_text_fields
                    )
                    or unsafe_text_fields != sorted(set(unsafe_text_fields))
                ):
                    raise GraphBundleValidationError(
                        "invalid XML-unsafe proposition rejection audit entry"
                    )
            reconstructed_grounding_ledger.append(
                {"chunk_id": chunk_id, **dict(rejection)}
            )

        for rejection in proposition_schema_rejections:
            if not isinstance(rejection, Mapping) or set(rejection) != {
                "proposition_index",
                "reason_code",
                "proposition_sha256",
            }:
                raise GraphBundleValidationError(
                    "proposition schema rejection audit fields do not match sealed schema"
                )
            proposition_index = rejection.get("proposition_index")
            proposition_hash = rejection.get("proposition_sha256")
            if (
                isinstance(proposition_index, bool)
                or not isinstance(proposition_index, int)
                or proposition_index < 0
                or rejection.get("reason_code")
                not in allowed_proposition_schema_reasons
                or not isinstance(proposition_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", proposition_hash)
            ):
                raise GraphBundleValidationError(
                    "invalid proposition schema rejection audit entry"
                )
            reconstructed_proposition_schema_ledger.append(
                {
                    "chunk_id": chunk_id,
                    **dict(rejection),
                    "raw_response_sha256": final_attempt["raw_response_sha256"],
                    "projected_response_sha256": final_attempt[
                        "projected_response_sha256"
                    ],
                    "proposition_schema_projection_policy_version": (
                        proposition_schema_policy
                    ),
                }
            )

    reconstructed_relation_ledger.sort(
        key=lambda item: (
            str(item["chunk_id"]),
            int(item["relation_index"]),
            str(item["reason_code"]),
            str(item["relation_sha256"]),
        )
    )
    reconstructed_grounding_ledger.sort(
        key=lambda item: (
            str(item["chunk_id"]),
            int(item["proposition_index"]),
            str(item["reason_code"]),
            str(item["proposition_sha256"]),
        )
    )
    reconstructed_proposition_schema_ledger.sort(
        key=lambda item: (
            str(item["chunk_id"]),
            int(item["proposition_index"]),
            str(item["reason_code"]),
            str(item["proposition_sha256"]),
        )
    )

    def _counter(name: str) -> int:
        value = scientific.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GraphBundleValidationError(
                f"telemetry scientific counter {name} must be a non-negative integer"
            )
        return value

    declared_extraction_chunks = _counter("extraction_chunks_total")
    declared_success = _counter("success")
    declared_explicit_empty = _counter("success_explicit_empty")
    declared_relation_rejection_chunks = _counter("success_with_relation_rejections")
    declared_grounding_rejection_chunks = _counter("success_with_grounding_rejections")
    declared_proposition_schema_rejection_chunks = _counter(
        "success_with_proposition_schema_rejections"
    )
    declared_projected_empty = _counter("success_projected_empty")
    declared_rejected_relations = _counter("causal_relations_rejected")
    declared_rejected_propositions = _counter("propositions_rejected_grounding")
    declared_rejected_schema_propositions = _counter(
        "propositions_rejected_schema"
    )
    declared_omitted_entities = _counter("entities_omitted_grounding")
    declared_attempts_total = _counter("attempts_total")
    if canonical_json(declared_relation_ledger) != canonical_json(
        reconstructed_relation_ledger
    ):
        raise GraphBundleValidationError("relation rejection ledger does not match traces")
    if scientific.get("relation_rejection_ledger_sha256") != canonical_sha256(
        reconstructed_relation_ledger
    ):
        raise GraphBundleValidationError("relation rejection ledger hash mismatch")
    if canonical_json(declared_grounding_ledger) != canonical_json(
        reconstructed_grounding_ledger
    ):
        raise GraphBundleValidationError("grounding rejection ledger does not match traces")
    if scientific.get("grounding_rejection_ledger_sha256") != canonical_sha256(
        reconstructed_grounding_ledger
    ):
        raise GraphBundleValidationError("grounding rejection ledger hash mismatch")
    if canonical_json(declared_proposition_schema_ledger) != canonical_json(
        reconstructed_proposition_schema_ledger
    ):
        raise GraphBundleValidationError(
            "proposition schema rejection ledger does not match traces"
        )
    if scientific.get("proposition_schema_rejection_ledger_sha256") != canonical_sha256(
        reconstructed_proposition_schema_ledger
    ):
        raise GraphBundleValidationError(
            "proposition schema rejection ledger hash mismatch"
        )
    if scientific.get("proposition_schema_projection_policy_version") != (
        EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
    ):
        raise GraphBundleValidationError(
            "proposition schema projection policy version mismatch"
        )
    if declared_extraction_chunks != len(chunk_ids):
        raise GraphBundleValidationError("telemetry extraction chunk count mismatch")
    if declared_success != nonempty_success:
        raise GraphBundleValidationError("telemetry successful extraction count mismatch")
    if declared_explicit_empty != explicit_empty_success:
        raise GraphBundleValidationError("telemetry explicit-empty extraction count mismatch")
    if declared_relation_rejection_chunks != success_with_relation_rejections:
        raise GraphBundleValidationError("telemetry relation-rejection chunk count mismatch")
    if declared_grounding_rejection_chunks != success_with_grounding_rejections:
        raise GraphBundleValidationError(
            "telemetry grounding-rejection chunk count mismatch"
        )
    if (
        declared_proposition_schema_rejection_chunks
        != success_with_proposition_schema_rejections
    ):
        raise GraphBundleValidationError(
            "telemetry proposition schema-rejection chunk count mismatch"
        )
    if declared_projected_empty != projected_empty_success:
        raise GraphBundleValidationError(
            "telemetry projected-empty extraction count mismatch"
        )
    if declared_rejected_relations != len(reconstructed_relation_ledger):
        raise GraphBundleValidationError("telemetry rejected relation count mismatch")
    if declared_rejected_propositions != len(reconstructed_grounding_ledger):
        raise GraphBundleValidationError(
            "telemetry rejected grounding proposition count mismatch"
        )
    if declared_rejected_schema_propositions != len(
        reconstructed_proposition_schema_ledger
    ):
        raise GraphBundleValidationError(
            "telemetry rejected proposition schema count mismatch"
        )
    if declared_omitted_entities != sum(
        int(entry["entity_count"]) for entry in reconstructed_grounding_ledger
    ):
        raise GraphBundleValidationError(
            "telemetry omitted grounding entity count mismatch"
        )
    if declared_attempts_total != attempts_total:
        raise GraphBundleValidationError("telemetry extraction attempt count mismatch")
    if scientific.get("checkpoint_required") is not False:
        raise GraphBundleValidationError("canonical graph build cannot require a checkpoint")


def _coerce_expected_construction_hash(expected_config: Any) -> Optional[str]:
    if expected_config is None:
        return None
    if isinstance(expected_config, Mapping):
        value = expected_config.get("construction_config_sha256")
    else:
        value = getattr(expected_config, "construction_config_sha256", None)
    if callable(value):
        value = value()
    return str(value) if value else None


def _validate_chunks(
    value: Any,
    *,
    graph_namespace: str,
    construction_config: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(value, Mapping) or value.get("schema_version") != CHUNKS_SCHEMA:
        raise GraphBundleValidationError(f"chunks.json must use {CHUNKS_SCHEMA}")
    chunks_raw = value.get("chunks")
    if not isinstance(chunks_raw, Mapping):
        raise GraphBundleValidationError("chunks.json 'chunks' must be an object keyed by chunk_id")
    raw_key = _contains_raw_only_key(value)
    if raw_key:
        raise GraphBundleValidationError(f"raw/label field {raw_key!r} leaked into chunks.json")

    for key, chunk in chunks_raw.items():
        if not isinstance(key, str) or not isinstance(chunk, Mapping):
            raise GraphBundleValidationError("chunks.json keys must be strings and values must be objects")
    chunks: Dict[str, Dict[str, Any]] = {}
    ordered = sorted(chunks_raw.items(), key=lambda item: item[1].get("chunk_index", -1))
    record_positions: Dict[str, list[int]] = {}
    record_meta: Dict[str, Tuple[Any, ...]] = {}
    required = {
        "content",
        "tokens",
        "token_start",
        "token_end",
        "chunk_index",
        "record_chunk_index",
        "chunk_id",
        "record_id",
        "record_line_number",
        "source",
        "source_path",
        "timestamp_raw",
        "timestamp_iso",
        "date",
        "availability_upper_bound_iso",
        "availability_upper_bound_date",
        "availability_bound_kind",
        "availability_bound_source",
        "availability_bound_source_sha256",
        "construction_payload_sha256",
        "record_sha256",
        "content_sha256",
        "layout_version",
        "normalization_version",
        "key_normalization_version",
    }
    for expected_index, (key, chunk) in enumerate(ordered):
        missing = sorted(required - set(chunk))
        if missing:
            raise GraphBundleValidationError(f"chunk {key!r} missing fields: {missing}")
        if key != chunk.get("chunk_id"):
            raise GraphBundleValidationError(f"chunk key/id mismatch for {key!r}")
        if not key.startswith("chk_") or len(key) != 68:
            raise GraphBundleValidationError(f"invalid chunk_id: {key!r}")
        if chunk.get("chunk_index") != expected_index:
            raise GraphBundleValidationError("chunk_index must be exactly 0..N-1")
        record_index = chunk.get("record_chunk_index")
        if isinstance(record_index, bool) or not isinstance(record_index, int) or record_index < 0:
            raise GraphBundleValidationError(f"invalid record_chunk_index for {key!r}")
        record_id = chunk.get("record_id")
        if not isinstance(record_id, str) or not record_id.startswith("rec_") or len(record_id) != 68:
            raise GraphBundleValidationError(f"invalid record_id for {key!r}")
        line_number = chunk.get("record_line_number")
        if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number <= 0:
            raise GraphBundleValidationError(f"invalid record_line_number for {key!r}")
        source_path = chunk.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise GraphBundleValidationError(f"invalid source_path for {key!r}")
        construction_hash = chunk.get("construction_payload_sha256")
        if not isinstance(construction_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", construction_hash):
            raise GraphBundleValidationError(f"invalid construction payload hash for {key!r}")
        expected_record_id = "rec_" + canonical_sha256(
            {
                "source_path": source_path,
                "record_line_number": line_number,
                "construction_payload_sha256": construction_hash,
            }
        )
        if record_id != expected_record_id:
            raise GraphBundleValidationError(f"record_id identity mismatch for {key!r}")
        if chunk.get("record_sha256") != chunk.get("construction_payload_sha256"):
            raise GraphBundleValidationError(f"record_sha256 mismatch for {key!r}")
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GraphBundleValidationError(f"invalid chunk content for {key!r}")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash != chunk.get("content_sha256"):
            raise GraphBundleValidationError(f"content hash mismatch for {key!r}")
        expected_chunk_id = "chk_" + canonical_sha256(
            {
                "graph_namespace": graph_namespace,
                "record_id": record_id,
                "record_chunk_index": record_index,
                "content_sha256": content_hash,
            }
        )
        if key != expected_chunk_id:
            raise GraphBundleValidationError(f"chunk_id identity mismatch for {key!r}")
        tokens = chunk.get("tokens")
        token_start = chunk.get("token_start")
        token_end = chunk.get("token_end")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (tokens, token_start, token_end)):
            raise GraphBundleValidationError(f"invalid token offsets for {key!r}")
        if tokens <= 0 or token_start < 0 or token_end <= token_start or token_end - token_start != tokens:
            raise GraphBundleValidationError(f"inconsistent token offsets for {key!r}")
        if chunk.get("layout_version") != construction_config.get("chunk_layout_version"):
            raise GraphBundleValidationError(f"chunk layout version mismatch for {key!r}")
        if chunk.get("normalization_version") != construction_config.get(
            "content_normalization_version"
        ):
            raise GraphBundleValidationError(f"content normalization version mismatch for {key!r}")
        if chunk.get("key_normalization_version") != construction_config.get(
            "key_normalization_version"
        ):
            raise GraphBundleValidationError(f"key normalization version mismatch for {key!r}")
        record_positions.setdefault(record_id, []).append(record_index)
        meta = (
            chunk.get("record_line_number"),
            chunk.get("source"),
            chunk.get("source_path"),
            chunk.get("timestamp_raw"),
            chunk.get("timestamp_iso"),
            chunk.get("date"),
            chunk.get("availability_upper_bound_iso", ""),
            chunk.get("availability_upper_bound_date", ""),
            chunk.get("construction_payload_sha256"),
        )
        previous = record_meta.setdefault(record_id, meta)
        if previous != meta:
            raise GraphBundleValidationError(f"record metadata changed across chunks: {record_id}")
        chunks[key] = dict(chunk)
    for record_id, positions in record_positions.items():
        if sorted(positions) != list(range(len(positions))):
            raise GraphBundleValidationError(
                f"record-local chunk indexes are not continuous for {record_id}"
            )
        record_chunks = sorted(
            (chunk for chunk in chunks.values() if chunk["record_id"] == record_id),
            key=lambda chunk: chunk["record_chunk_index"],
        )
        if record_chunks and record_chunks[0]["token_start"] != 0:
            raise GraphBundleValidationError(f"record token layout does not start at zero: {record_id}")
        overlap = construction_config.get("chunk_overlap_tokens")
        max_tokens = construction_config.get("chunk_max_tokens")
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
            or not 0 <= overlap < max_tokens
        ):
            raise GraphBundleValidationError("manifest has invalid chunk token configuration")
        for previous, current in zip(record_chunks, record_chunks[1:]):
            expected_start = previous["token_start"] + max_tokens - overlap
            if current["token_start"] != expected_start:
                raise GraphBundleValidationError(
                    f"record-local chunk overlap/layout mismatch for {record_id}"
                )
    return chunks, dict(value)


def _audit_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise GraphBundleValidationError(
            f"source audit {field} must be a lower-case SHA-256"
        )
    return value


def _audit_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise GraphBundleValidationError(
            f"source audit {field} must be a non-empty relative path"
        )
    normalized = unicodedata.normalize("NFKC", value.replace("\\", "/"))
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise GraphBundleValidationError(
            f"source audit {field} must be relative and traversal-free"
        )
    canonical = path.as_posix().lstrip("./")
    if not canonical or canonical != value:
        raise GraphBundleValidationError(
            f"source audit {field} is not canonically normalized"
        )
    return canonical


def _validate_source_audit(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    chunks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:


    if not isinstance(value, Mapping) or value.get("schema_version") != SOURCE_AUDIT_SCHEMA:
        raise GraphBundleValidationError(
            f"{SOURCE_AUDIT_FILE} must use {SOURCE_AUDIT_SCHEMA}"
        )
    required = {
        "schema_version",
        "inputs",
        "records",
        "raw_input_set_sha256",
        "raw_record_set_sha256",
        "canonical_construction_stream_sha256",
        "record_count",
        "construction_fingerprint_sha256",
        "base_graph_namespace",
        "graph_bundle_sha256",
    }
    if set(value) != required:
        raise GraphBundleValidationError(
            f"{SOURCE_AUDIT_FILE} fields must be exactly {sorted(required)}"
        )
    inputs = value["inputs"]
    records = value["records"]
    if not isinstance(inputs, list) or not inputs:
        raise GraphBundleValidationError("source audit inputs must be a non-empty array")
    if not isinstance(records, list) or not records:
        raise GraphBundleValidationError("source audit records must be a non-empty array")

    input_rows: list[Dict[str, Any]] = []
    seen_inputs: set[str] = set()
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping) or set(item) != {
            "source_path", "raw_input_file_sha256", "bytes", "records"
        }:
            raise GraphBundleValidationError(
                f"source audit input {index} has invalid fields"
            )
        source_path = _audit_relative_path(item.get("source_path"), field=f"inputs[{index}].source_path")
        if source_path in seen_inputs:
            raise GraphBundleValidationError(f"source audit has duplicate input path: {source_path}")
        seen_inputs.add(source_path)
        _audit_sha256(item.get("raw_input_file_sha256"), field=f"inputs[{index}].raw_input_file_sha256")
        size = item.get("bytes")
        count = item.get("records")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise GraphBundleValidationError(f"source audit inputs[{index}].bytes is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GraphBundleValidationError(f"source audit inputs[{index}].records is invalid")
        input_rows.append({
            "source_path": source_path,
            "raw_input_file_sha256": item["raw_input_file_sha256"],
            "bytes": size,
            "records": count,
        })
    if input_rows != sorted(input_rows, key=lambda item: item["source_path"]):
        raise GraphBundleValidationError("source audit inputs are not canonically ordered")

    record_rows: list[Dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_record_positions: set[Tuple[str, int]] = set()
    for index, item in enumerate(records):
        if not isinstance(item, Mapping) or set(item) != {
            "source_path", "record_line_number", "input_id", "raw_record_sha256",
            "construction_payload_sha256", "record_id",
        }:
            raise GraphBundleValidationError(
                f"source audit record {index} has invalid fields"
            )
        source_path = _audit_relative_path(item.get("source_path"), field=f"records[{index}].source_path")
        if source_path not in seen_inputs:
            raise GraphBundleValidationError(
                f"source audit record refers to unknown input: {source_path}"
            )
        line = item.get("record_line_number")
        if isinstance(line, bool) or not isinstance(line, int) or line <= 0:
            raise GraphBundleValidationError(
                f"source audit records[{index}].record_line_number is invalid"
            )
        if not isinstance(item.get("input_id"), str):
            raise GraphBundleValidationError(f"source audit records[{index}].input_id is invalid")
        position = (source_path, line)
        if position in seen_record_positions:
            raise GraphBundleValidationError(
                f"source audit has duplicate record position: {position}"
            )
        seen_record_positions.add(position)
        construction_hash = _audit_sha256(
            item.get("construction_payload_sha256"),
            field=f"records[{index}].construction_payload_sha256",
        )
        _audit_sha256(item.get("raw_record_sha256"), field=f"records[{index}].raw_record_sha256")
        record_id = item.get("record_id")
        expected_record_id = "rec_" + canonical_sha256({
            "source_path": source_path,
            "record_line_number": line,
            "construction_payload_sha256": construction_hash,
        })
        if record_id != expected_record_id:
            raise GraphBundleValidationError(
                f"source audit record_id identity mismatch at index {index}"
            )
        if record_id in seen_record_ids:
            raise GraphBundleValidationError(f"source audit has duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)
        record_rows.append({
            "source_path": source_path,
            "record_line_number": line,
            "input_id": item["input_id"],
            "raw_record_sha256": item["raw_record_sha256"],
            "construction_payload_sha256": construction_hash,
            "record_id": record_id,
        })
    if record_rows != sorted(
        record_rows, key=lambda item: (item["source_path"], item["record_line_number"])
    ):
        raise GraphBundleValidationError("source audit records are not canonically ordered")
    record_count = value["record_count"]
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise GraphBundleValidationError("source audit record_count must be a positive integer")
    if record_count != len(record_rows) or record_count != sum(
        item["records"] for item in input_rows
    ):
        raise GraphBundleValidationError("source audit record_count does not close over inputs")
    counts_by_path: Dict[str, int] = {}
    for item in record_rows:
        counts_by_path[item["source_path"]] = counts_by_path.get(item["source_path"], 0) + 1
    if any(item["records"] != counts_by_path.get(item["source_path"], 0) for item in input_rows):
        raise GraphBundleValidationError("source audit per-input record counts do not close")
    raw_input_hash = _audit_sha256(value["raw_input_set_sha256"], field="raw_input_set_sha256")
    raw_record_hash = _audit_sha256(value["raw_record_set_sha256"], field="raw_record_set_sha256")
    if canonical_sha256(input_rows) != raw_input_hash:
        raise GraphBundleValidationError("source audit raw input hash mismatch")
    if canonical_sha256(record_rows) != raw_record_hash:
        raise GraphBundleValidationError("source audit raw record hash mismatch")
    _audit_sha256(value["canonical_construction_stream_sha256"], field="canonical_construction_stream_sha256")
    _audit_sha256(value["construction_fingerprint_sha256"], field="construction_fingerprint_sha256")
    _audit_sha256(value["graph_bundle_sha256"], field="graph_bundle_sha256")
    namespace = value["base_graph_namespace"]
    if not isinstance(namespace, str) or not re.fullmatch(r"graph_[0-9a-f]{64}", namespace):
        raise GraphBundleValidationError("source audit base_graph_namespace is invalid")
    for field in (
        "canonical_construction_stream_sha256",
        "construction_fingerprint_sha256",
        "base_graph_namespace",
        "graph_bundle_sha256",
    ):
        if value[field] != manifest.get(field):
            raise GraphBundleValidationError(
                f"source audit/manifest mismatch for {field}"
            )
    expected_fingerprint = canonical_sha256({
        "canonical_construction_stream_sha256": manifest["canonical_construction_stream_sha256"],
        "resolved_construction_config": manifest["resolved_construction_config"],
        "construction_config_sha256": manifest["construction_config_sha256"],
    })
    if value["construction_fingerprint_sha256"] != expected_fingerprint:
        raise GraphBundleValidationError("source audit construction fingerprint mismatch")
    expected_namespace = "graph_" + canonical_sha256({
        "canonical_construction_stream_sha256": manifest["canonical_construction_stream_sha256"],
        "construction_fingerprint_sha256": manifest["construction_fingerprint_sha256"],
    })
    if namespace != expected_namespace:
        raise GraphBundleValidationError("source audit graph namespace mismatch")



    chunk_records: Dict[str, Tuple[str, int, str, str]] = {}
    for chunk in chunks.values():
        record_id = str(chunk["record_id"])
        entry = (
            str(chunk["source_path"]),
            int(chunk["record_line_number"]),
            str(chunk["construction_payload_sha256"]),
            record_id,
        )
        prior = chunk_records.setdefault(record_id, entry)
        if prior != entry:
            raise GraphBundleValidationError(
                f"source audit/chunk record closure mismatch for {record_id}"
            )
    expected_chunk_records = [
        (item["source_path"], item["record_line_number"], item["construction_payload_sha256"], item["record_id"])
        for item in record_rows
    ]
    observed_chunk_records = [
        chunk_records[key]
        for key in sorted(chunk_records, key=lambda key: (chunk_records[key][0], chunk_records[key][1]))
    ]
    if observed_chunk_records != expected_chunk_records:
        raise GraphBundleValidationError("source audit records do not close over chunks")
    return dict(value)


def _validate_vdb(value: Any, *, filename: str) -> Tuple[Dict[str, Dict[str, Any]], Any, Dict[str, Any]]:
    try:
        import numpy as np
    except Exception as exc:
        raise GraphBundleValidationError(f"numpy unavailable while validating {filename}: {exc}") from exc

    if not isinstance(value, Mapping):
        raise GraphBundleValidationError(f"{filename} must contain an object")
    dim = value.get("embedding_dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise GraphBundleValidationError(f"{filename} has invalid embedding_dim")
    data = value.get("data")
    matrix_encoded = value.get("matrix")
    additional = value.get("additional_data")
    if not isinstance(data, list) or not isinstance(matrix_encoded, str) or not isinstance(additional, Mapping):
        raise GraphBundleValidationError(f"{filename} has an invalid vector-store structure")
    if additional.get("schema_version") != VECTOR_SCHEMA:
        raise GraphBundleValidationError(f"{filename} must use {VECTOR_SCHEMA}")
    signature = additional.get("embedding_signature")
    if not isinstance(signature, Mapping):
        raise GraphBundleValidationError(f"{filename} missing embedding_signature")
    required_sig = {
        "provider",
        "endpoint_identity",
        "model",
        "revision",
        "dimension",
        "preprocess_version",
        "dtype",
        "serialization_precision",
        "normalization",
        "distance",
    }
    if set(signature) != required_sig:
        raise GraphBundleValidationError(
            f"{filename} embedding signature fields mismatch: {sorted(set(signature) ^ required_sig)}"
        )
    if (
        signature.get("dimension") != dim
        or signature.get("dtype") != "float32"
        or signature.get("serialization_precision") != "float32-le"
        or signature.get("normalization") != "l2"
        or signature.get("distance") != "cosine"
    ):
        raise GraphBundleValidationError(f"{filename} embedding signature/dimension mismatch")
    try:
        raw = base64.b64decode(matrix_encoded, validate=True)
        matrix = np.frombuffer(raw, dtype=np.dtype("<f4"))
        matrix = matrix.reshape((len(data), dim))
    except Exception as exc:
        raise GraphBundleValidationError(f"{filename} has invalid float32 matrix: {exc}") from exc
    if not np.isfinite(matrix).all():
        raise GraphBundleValidationError(f"{filename} contains non-finite vectors")
    records: Dict[str, Dict[str, Any]] = {}
    node_ids: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise GraphBundleValidationError(f"{filename} record {index} is not an object")
        vector_id = item.get("__id__")
        node_id = item.get("node_id")
        if not isinstance(vector_id, str) or not vector_id or vector_id in records:
            raise GraphBundleValidationError(f"{filename} contains invalid/duplicate vector id")
        if not isinstance(node_id, str) or not node_id:
            raise GraphBundleValidationError(f"{filename} vector {vector_id!r} missing node_id")
        if node_id in node_ids:
            raise GraphBundleValidationError(f"{filename} contains duplicate node_id {node_id!r}")
        if "__vector__" in item or "__metrics__" in item:
            raise GraphBundleValidationError(f"{filename} record contains reserved vector fields")
        node_ids.add(node_id)
        records[vector_id] = dict(item)
    if len(data):
        norms = np.linalg.norm(matrix, axis=1)
        if (norms == 0).any():
            raise GraphBundleValidationError(f"{filename} contains a zero vector")
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise GraphBundleValidationError(f"{filename} vectors are not sealed L2-normalized values")
    return records, matrix, dict(additional)


def _artifact_info(graph_dir: Path) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for name in KG_ARTIFACTS:
        path = graph_dir / name
        if not path.is_file():
            raise GraphBundleValidationError(f"missing canonical KG artifact: {name}")
        size = path.stat().st_size
        if size <= 0:
            raise GraphBundleValidationError(f"empty canonical KG artifact: {name}")
        info[name] = {"sha256": sha256_file(path), "bytes": size}
    return info


def validate_graph_bundle_forensic(
    path: os.PathLike[str] | str,
    *,
    expected_profile_id: Optional[str] = None,
    expected_config: Any = None,
    expected_config_sha256: Optional[str] = None,
    expected_construction_config_sha256: Optional[str] = None,
    require_publishable: bool = False,
) -> Dict[str, Any]:


    graph_dir = Path(path)
    if not graph_dir.is_dir():
        raise GraphBundleValidationError(f"graph directory does not exist: {graph_dir}")
    artifact_info = _artifact_info(graph_dir)
    for name in METADATA_ARTIFACTS:
        metadata_path = graph_dir / name
        if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
            raise GraphBundleValidationError(f"missing/empty graph metadata artifact: {name}")

    manifest = _read_json(graph_dir / "graph_manifest.json")
    telemetry = _read_json(graph_dir / "build_telemetry.json")
    source_audit_path = graph_dir / SOURCE_AUDIT_FILE
    if not source_audit_path.is_file() or source_audit_path.stat().st_size <= 0:
        raise GraphBundleValidationError(
            f"missing/empty required graph metadata artifact: {SOURCE_AUDIT_FILE}"
        )
    source_audit_value = _read_json(source_audit_path)
    chunks_value = _read_json(graph_dir / "chunks.json")
    entity_value = _read_json(graph_dir / "entity_vdb.json")
    hyperedge_value = _read_json(graph_dir / "hyperedge_vdb.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != GRAPH_BUNDLE_SCHEMA:
        raise GraphBundleValidationError(f"graph_manifest.json must use {GRAPH_BUNDLE_SCHEMA}")
    if not isinstance(telemetry, Mapping) or telemetry.get("schema_version") != TELEMETRY_SCHEMA:
        raise GraphBundleValidationError(f"build_telemetry.json must use {TELEMETRY_SCHEMA}")
    _require_fields(
        manifest,
        {
            "profile_id",
            "publishable",
            "out_of_paper_protocol",
            "artifacts",
            "canonical_construction_stream_sha256",
            "construction_fingerprint_sha256",
            "base_graph_namespace",
            "construction_config_sha256",
            "scientific_config_sha256",
            "resolved_config",
            "resolved_construction_config",
            "embedding_signatures",
            "manifest_payload_sha256",
            "graph_bundle_sha256",
        },
        context="graph_manifest.json",
    )
    _require_fields(
        telemetry,
        {
            "status",
            "errors",
            "profile_id",
            "publishable",
            "out_of_paper_protocol",
            "artifacts",
            "counts",
            "scientific",
            "operational",
            "construction_config_sha256",
            "scientific_config_sha256",
            "construction_fingerprint_sha256",
            "telemetry_payload_sha256",
            "graph_bundle_sha256",
        },
        context="build_telemetry.json",
    )
    _reject_raw_only_keys(manifest, context="graph manifest")
    _reject_raw_only_keys(telemetry, context="build telemetry")
    _reject_raw_only_keys(entity_value, context="entity VDB")
    _reject_raw_only_keys(hyperedge_value, context="hyperedge VDB")
    if telemetry.get("status") != "success" or telemetry.get("errors") != []:
        raise GraphBundleValidationError("build telemetry is not a successful, error-free build")
    scientific_telemetry = telemetry.get("scientific")
    operational_telemetry = telemetry.get("operational")
    if not isinstance(scientific_telemetry, Mapping) or not isinstance(
        operational_telemetry, Mapping
    ):
        raise GraphBundleValidationError(
            "build telemetry scientific/operational sections must be objects"
        )
    _require_fields(
        scientific_telemetry,
        {
            "extraction_chunks_total",
            "success",
            "success_explicit_empty",
            "success_with_relation_rejections",
            "success_with_grounding_rejections",
            "success_with_proposition_schema_rejections",
            "success_projected_empty",
            "causal_relations_rejected",
            "relation_rejection_ledger_sha256",
            "relation_admissibility_policy_version",
            "grounding_policy_version",
            "propositions_rejected_grounding",
            "entities_omitted_grounding",
            "grounding_rejection_ledger_sha256",
            "proposition_schema_projection_policy_version",
            "propositions_rejected_schema",
            "proposition_schema_rejection_ledger_sha256",
            "grounding_replay",
            "attempts_total",
            "checkpoint_required",
        },
        context="build telemetry scientific section",
    )

    construction_config = manifest.get("resolved_construction_config")
    if not isinstance(construction_config, Mapping):
        raise GraphBundleValidationError("manifest resolved_construction_config must be an object")
    _require_fields(
        construction_config,
        {
            "schema_version",
            "profile_id",
            "chunk_layout_version",
            "content_normalization_version",
            "key_normalization_version",
            "compat_adapter_version",
            "chunk_max_tokens",
            "chunk_overlap_tokens",
            "requested_tokenizer_model",
            "resolved_tokenizer_encoding",
            "tiktoken_version",
            "extraction_batch_size",
            "extraction_schema_retries",
            "extraction_transport_attempts_per_schema_attempt",
            "extraction_temperature",
            "extraction_max_output_tokens",
            "extractor_provider",
            "extractor_model",
            "extraction_prompt_version",
            "extraction_prompt_sha256",
            "extraction_schema_version",
            "relation_admissibility_policy_version",
            "grounding_policy_version",
            "proposition_schema_projection_policy_version",
            "extraction_response_format_version",
            "extraction_response_format",
            "embedding_signature",
            "timestamp_policy",
            "availability_policy",
            "compat_retrieval_version",
            "compat_rrf_k",
            "compat_rrf_default_top_k",
            "compat_rrf_channel_candidate_multiplier",
        },
        context="resolved construction config",
    )
    active_identity = _active_builder_construction_identity()
    if construction_config.get("schema_version") != EXPECTED_CONSTRUCTION_CONFIG_SCHEMA:
        raise GraphBundleValidationError("resolved construction config schema mismatch")
    if (
        construction_config.get("content_normalization_version")
        != active_identity["content_normalization_version"]
    ):
        raise GraphBundleValidationError("resolved content normalization version mismatch")
    if (
        construction_config.get("key_normalization_version")
        != active_identity["key_normalization_version"]
    ):
        raise GraphBundleValidationError("resolved key normalization version mismatch")
    if (
        construction_config.get("extraction_prompt_version")
        != active_identity["extraction_prompt_version"]
    ):
        raise GraphBundleValidationError("resolved extraction prompt version mismatch")
    prompt_sha256 = construction_config.get("extraction_prompt_sha256")
    if (
        not isinstance(prompt_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", prompt_sha256)
        or prompt_sha256 != active_identity["extraction_prompt_sha256"]
    ):
        raise GraphBundleValidationError("resolved extraction prompt hash mismatch")
    if (
        construction_config.get("extraction_schema_version")
        != active_identity["extraction_schema_version"]
    ):
        raise GraphBundleValidationError("resolved extraction schema mismatch")
    relation_policy = active_identity["relation_admissibility_policy_version"]
    if construction_config.get("relation_admissibility_policy_version") != relation_policy:
        raise GraphBundleValidationError("resolved relation admissibility policy mismatch")
    if scientific_telemetry.get("relation_admissibility_policy_version") != relation_policy:
        raise GraphBundleValidationError("telemetry relation admissibility policy mismatch")
    if construction_config.get("grounding_policy_version") != EXPECTED_GROUNDING_POLICY_VERSION:
        raise GraphBundleValidationError("resolved grounding policy mismatch")
    if scientific_telemetry.get("grounding_policy_version") != EXPECTED_GROUNDING_POLICY_VERSION:
        raise GraphBundleValidationError("telemetry grounding policy mismatch")
    if (
        construction_config.get("proposition_schema_projection_policy_version")
        != EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
    ):
        raise GraphBundleValidationError(
            "resolved proposition schema projection policy mismatch"
        )
    if (
        scientific_telemetry.get("proposition_schema_projection_policy_version")
        != EXPECTED_PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
    ):
        raise GraphBundleValidationError(
            "telemetry proposition schema projection policy mismatch"
        )
    if (
        construction_config.get("extraction_response_format_version")
        != EXPECTED_EXTRACTION_RESPONSE_FORMAT_VERSION
    ):
        raise GraphBundleValidationError(
            "resolved extraction response-format protocol mismatch"
        )
    if construction_config.get("extraction_response_format") != {
        "type": "json_object"
    }:
        raise GraphBundleValidationError(
            "resolved extraction response format must be json_object"
        )
    for name in (
        "construction_config_sha256",
        "scientific_config_sha256",
        "construction_fingerprint_sha256",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise GraphBundleValidationError(f"manifest has invalid {name}")
    for name in (
        "compat_rrf_k",
        "compat_rrf_default_top_k",
        "compat_rrf_channel_candidate_multiplier",
        "extraction_transport_attempts_per_schema_attempt",
    ):
        value = construction_config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GraphBundleValidationError(f"resolved construction config has invalid {name}")
    if (
        construction_config.get("extraction_transport_attempts_per_schema_attempt")
        != EXPECTED_EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT
    ):
        raise GraphBundleValidationError(
            "graph extraction transport retry budget does not match the sealed policy"
        )
    extraction_schema_retries = construction_config.get("extraction_schema_retries")
    if (
        isinstance(extraction_schema_retries, bool)
        or not isinstance(extraction_schema_retries, int)
        or not 0 <= extraction_schema_retries <= 2
    ):
        raise GraphBundleValidationError(
            "graph extraction schema retries must be an integer in [0,2]"
        )

    profile_id = manifest.get("profile_id")
    if profile_id not in {"compat", "paper"}:
        raise GraphBundleValidationError(f"unknown graph profile_id: {profile_id!r}")
    if telemetry.get("profile_id") != profile_id:
        raise GraphBundleValidationError("manifest/telemetry profile mismatch")
    if construction_config.get("profile_id") != profile_id:
        raise GraphBundleValidationError("construction config/profile mismatch")
    if expected_profile_id and expected_profile_id != profile_id:
        raise GraphBundleValidationError(
            f"graph profile mismatch: expected {expected_profile_id!r}, found {profile_id!r}"
        )






    expected_construction_hash = (
        expected_construction_config_sha256
        or _coerce_expected_construction_hash(expected_config)
    )
    if (
        expected_construction_hash
        and manifest.get("construction_config_sha256") != expected_construction_hash
    ):
        raise GraphBundleValidationError("graph construction configuration identity mismatch")
    if (
        expected_config_sha256
        and manifest.get("scientific_config_sha256") != expected_config_sha256
    ):
        raise GraphBundleValidationError("graph build invocation identity mismatch")

    publishable = manifest.get("publishable") is True
    out_of_paper = manifest.get("out_of_paper_protocol") is True
    if profile_id == "compat":
        if publishable or not out_of_paper:
            raise GraphBundleValidationError("compat graphs must be non-publishable/out-of-paper")
    else:
        if not publishable or out_of_paper:
            raise GraphBundleValidationError(
                "paper-profile graphs must be publishable and inside paper protocol"
            )
    if require_publishable and (profile_id != "paper" or not publishable or out_of_paper):
        raise GraphBundleValidationError("a publishable paper-profile graph is required")
    for field in (
        "publishable",
        "out_of_paper_protocol",
        "construction_config_sha256",
        "scientific_config_sha256",
        "construction_fingerprint_sha256",
    ):
        if telemetry.get(field) != manifest.get(field):
            raise GraphBundleValidationError(f"manifest/telemetry mismatch for {field}")

    graph_namespace = manifest.get("base_graph_namespace")
    if not isinstance(graph_namespace, str) or not graph_namespace.startswith("graph_") or len(graph_namespace) != 70:
        raise GraphBundleValidationError("manifest has invalid base_graph_namespace")
    chunks, chunks_document = _validate_chunks(
        chunks_value,
        graph_namespace=graph_namespace,
        construction_config=construction_config,
    )
    source_audit = _validate_source_audit(
        source_audit_value,
        manifest=manifest,
        chunks=chunks,
    )
    _validate_relation_rejection_telemetry(
        telemetry,
        chunk_ids=set(chunks),
        schema_retries=int(construction_config["extraction_schema_retries"]),
        transport_attempts_per_schema_attempt=int(
            construction_config[
                "extraction_transport_attempts_per_schema_attempt"
            ]
        ),
    )
    replay_receipt = _validate_extraction_grounding_replay(
        telemetry,
        chunks=chunks,
        schema_retries=int(construction_config["extraction_schema_retries"]),
        transport_attempts_per_schema_attempt=int(
            construction_config[
                "extraction_transport_attempts_per_schema_attempt"
            ]
        ),
    )
    projected_by_chunk = replay_receipt.get("projected_by_chunk")
    if not isinstance(projected_by_chunk, Mapping):
        raise GraphBundleValidationError(
            "grounding replay did not return projected extraction closure"
        )
    expected_replay_graph = _expected_replay_graph_closure(
        projected_by_chunk,
        chunks=chunks,
        graph_namespace=graph_namespace,
    )
    entity_records, entity_matrix, entity_additional = _validate_vdb(
        entity_value, filename="entity_vdb.json"
    )
    hyperedge_records, hyperedge_matrix, hyperedge_additional = _validate_vdb(
        hyperedge_value, filename="hyperedge_vdb.json"
    )

    try:
        import networkx as nx

        graph = nx.read_graphml(str(graph_dir / "knowledge_graph.graphml"), force_multigraph=True)
    except Exception as exc:
        raise GraphBundleValidationError(f"cannot parse knowledge_graph.graphml: {exc}") from exc
    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    _reject_graph_attr_leakage(graph.graph, context="GraphML graph attributes")
    if graph.graph.get("schema_version") != GRAPH_SCHEMA:
        raise GraphBundleValidationError(f"GraphML must use {GRAPH_SCHEMA}")
    if graph.graph.get("base_graph_namespace") != manifest.get("base_graph_namespace"):
        raise GraphBundleValidationError("GraphML/manifest graph namespace mismatch")
    if graph.graph.get("construction_config_sha256") != manifest.get(
        "construction_config_sha256"
    ):
        raise GraphBundleValidationError(
            "GraphML/manifest construction configuration identity mismatch"
        )
    if graph.graph.get("canonical_construction_stream_sha256") != manifest.get(
        "canonical_construction_stream_sha256"
    ):
        raise GraphBundleValidationError("GraphML/manifest construction stream mismatch")
    if graph.graph.get("construction_fingerprint_sha256") != manifest.get(
        "construction_fingerprint_sha256"
    ):
        raise GraphBundleValidationError("GraphML/manifest construction fingerprint mismatch")

    entity_nodes = {node for node, attrs in graph.nodes(data=True) if attrs.get("role") == "entity"}
    hyperedge_nodes = {node for node, attrs in graph.nodes(data=True) if attrs.get("role") == "hyperedge"}
    unknown_roles = {
        attrs.get("role") for _, attrs in graph.nodes(data=True) if attrs.get("role") not in {"entity", "hyperedge"}
    }
    if unknown_roles:
        raise GraphBundleValidationError(f"unknown/missing graph node roles: {sorted(map(str, unknown_roles))}")
    for node_id, attrs in graph.nodes(data=True):
        _reject_graph_attr_leakage(attrs, context=f"GraphML node {node_id}")
        if attrs.get("role") == "entity":
            entity_key = attrs.get("canonical_entity_key")
            if not isinstance(entity_key, str) or not entity_key:
                raise GraphBundleValidationError(f"entity {node_id!r} lacks canonical_entity_key")
            expected_node_id = "ent_" + canonical_sha256(
                {"graph_namespace": graph_namespace, "canonical_entity_key": entity_key}
            )
            if str(node_id) != expected_node_id:
                raise GraphBundleValidationError(f"entity stable identity mismatch for {node_id!r}")
        else:
            chunk_id = attrs.get("chunk_id")
            proposition_index = attrs.get("proposition_index")
            proposition_key = attrs.get("proposition_key")
            if (
                not isinstance(chunk_id, str)
                or chunk_id not in chunks
                or isinstance(proposition_index, bool)
                or not isinstance(proposition_index, int)
                or proposition_index < 0
                or not isinstance(proposition_key, str)
                or not proposition_key
            ):
                raise GraphBundleValidationError(f"hyperedge identity fields are invalid for {node_id!r}")
            expected_node_id = "hyp_" + canonical_sha256(
                {
                    "graph_namespace": graph_namespace,
                    "chunk_id": chunk_id,
                    "proposition_index": proposition_index,
                    "proposition_key": proposition_key,
                }
            )
            if str(node_id) != expected_node_id:
                raise GraphBundleValidationError(f"hyperedge stable identity mismatch for {node_id!r}")

    grounded_incidence_occurrences, grounded_entity_chunks = (
        _validate_graph_source_grounding(
            graph,
            chunks=chunks,
            entity_nodes=entity_nodes,
            hyperedge_nodes=hyperedge_nodes,
        )
    )

    signatures = manifest.get("embedding_signatures")
    if not isinstance(signatures, Mapping) or set(signatures) != {"entity", "hyperedge", "query", "target"}:
        raise GraphBundleValidationError("manifest must declare entity/hyperedge/query/target signatures")
    sealed_signature = canonical_json(signatures["entity"])
    if any(canonical_json(signatures[role]) != sealed_signature for role in ("hyperedge", "query", "target")):
        raise GraphBundleValidationError("all graph/query/target embeddings must share one sealed space")
    if canonical_json(construction_config.get("embedding_signature")) != sealed_signature:
        raise GraphBundleValidationError("construction config embedding signature does not match manifest")

    entity_refs = {record["node_id"] for record in entity_records.values()}
    hyperedge_refs = {record["node_id"] for record in hyperedge_records.values()}
    if entity_refs != entity_nodes:
        raise GraphBundleValidationError("entity VDB and GraphML entity-node closure mismatch")
    if hyperedge_refs != hyperedge_nodes:
        raise GraphBundleValidationError("hyperedge VDB and GraphML occurrence-node closure mismatch")
    if canonical_json(entity_additional.get("embedding_signature")) != canonical_json(
        manifest.get("embedding_signatures", {}).get("entity")
    ):
        raise GraphBundleValidationError("entity embedding signature does not match manifest")
    if canonical_json(hyperedge_additional.get("embedding_signature")) != canonical_json(
        manifest.get("embedding_signatures", {}).get("hyperedge")
    ):
        raise GraphBundleValidationError("hyperedge embedding signature does not match manifest")
    if entity_additional.get("role") != "entity":
        raise GraphBundleValidationError("entity VDB role does not match replay schema")
    if hyperedge_additional.get("role") != "hyperedge_occurrence":
        raise GraphBundleValidationError("hyperedge VDB role does not match replay schema")
    for node_id in sorted(hyperedge_nodes):
        attrs = graph.nodes[node_id]
        chunk_id = attrs.get("chunk_id") or attrs.get("source_id")
        chunk = chunks.get(str(chunk_id))
        if chunk is None:
            raise GraphBundleValidationError(f"hyperedge {node_id!r} refers to missing chunk {chunk_id!r}")
        for field in (
            "record_id",
            "source",
            "timestamp_raw",
            "timestamp_iso",
            "date",
            "availability_upper_bound_iso",
            "availability_upper_bound_date",
        ):
            if str(attrs.get(field, "")) != str(chunk.get(field, "")):
                raise GraphBundleValidationError(
                    f"hyperedge {node_id!r} disagrees with source chunk on {field}"
                )
        if not isinstance(attrs.get("proposition_key"), str) or not attrs.get("proposition_key"):
            raise GraphBundleValidationError(f"hyperedge {node_id!r} missing proposition_key")

    keyed_edges: list[Tuple[str, str, str]] = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        key_s = str(key)
        keyed_edges.append((str(source), str(target), key_s))
        _reject_graph_attr_leakage(attrs, context=f"GraphML edge {key_s}")
        role = attrs.get("role")
        if attrs.get("edge_id") != key_s:
            raise GraphBundleValidationError(f"edge key/edge_id mismatch for {key_s!r}")
        if role == "incidence":
            if not key_s.startswith("inc_"):
                raise GraphBundleValidationError(f"invalid incidence edge key: {key_s!r}")
            if source not in hyperedge_nodes or target not in entity_nodes:
                raise GraphBundleValidationError(f"invalid incidence endpoints for {key_s!r}")
            source_chunk_id = str(graph.nodes[source].get("chunk_id") or "")
            if (str(target), str(source), source_chunk_id) not in grounded_incidence_occurrences:
                raise GraphBundleValidationError(
                    f"incidence edge {key_s!r} lacks grounded alias provenance"
                )
            expected_key = _stable_edge_id(
                "inc_",
                {
                    "graph_namespace": graph_namespace,
                    "hyperedge_id": str(source),
                    "entity_id": str(target),
                    "incidence_role": attrs.get("incidence_role", ""),
                },
            )
            if key_s != expected_key:
                raise GraphBundleValidationError(f"incidence edge stable identity mismatch for {key_s!r}")
        elif role == "causal":
            if not key_s.startswith("cedge_"):
                raise GraphBundleValidationError(f"invalid causal edge key: {key_s!r}")
            if source not in entity_nodes or target not in entity_nodes or source == target:
                raise GraphBundleValidationError(f"invalid causal endpoints for {key_s!r}")
            if attrs.get("causal_type") not in ALLOWED_CAUSAL_TYPES:
                raise GraphBundleValidationError(f"invalid causal type for {key_s!r}")
            relation_index = attrs.get("relation_index")
            if isinstance(relation_index, bool) or not isinstance(relation_index, int) or relation_index < 0:
                raise GraphBundleValidationError(f"invalid causal relation_index for {key_s!r}")
            strength = attrs.get("strength")
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise GraphBundleValidationError(f"invalid causal strength for {key_s!r}")
            if (
                isinstance(strength, float)
                and not math.isfinite(strength)
            ) or not 0.0 < strength <= 1.0:
                raise GraphBundleValidationError(f"out-of-range causal strength for {key_s!r}")
            chunk_id = str(attrs.get("chunk_id") or attrs.get("source_id") or "")
            if chunk_id not in chunks:
                raise GraphBundleValidationError(f"causal edge {key_s!r} refers to missing chunk")
            if (str(source), chunk_id) not in grounded_entity_chunks or (
                str(target), chunk_id
            ) not in grounded_entity_chunks:
                raise GraphBundleValidationError(
                    f"causal edge {key_s!r} endpoints are not grounded in its source chunk"
                )
            chunk = chunks[chunk_id]
            for field in (
                "record_id",
                "source",
                "timestamp_raw",
                "timestamp_iso",
                "date",
                "availability_upper_bound_iso",
                "availability_upper_bound_date",
            ):
                if str(attrs.get(field, "")) != str(chunk.get(field, "")):
                    raise GraphBundleValidationError(
                        f"causal edge {key_s!r} disagrees with source chunk on {field}"
                    )
            source_hyperedges = _json_attr(
                attrs.get("source_hyperedge_ids"), field="source_hyperedge_ids"
            )
            if not isinstance(source_hyperedges, list) or not source_hyperedges:
                raise GraphBundleValidationError(
                    f"causal edge {key_s!r} lacks source hyperedge provenance"
                )
            if any(
                hyperedge_id not in hyperedge_nodes
                or graph.nodes[hyperedge_id].get("chunk_id") != chunk_id
                for hyperedge_id in source_hyperedges
            ):
                raise GraphBundleValidationError(
                    f"causal edge {key_s!r} source hyperedge provenance is not closed"
                )
            expected_key = _stable_edge_id(
                "cedge_",
                {
                    "graph_namespace": graph_namespace,
                    "chunk_id": chunk_id,
                    "relation_index": relation_index,
                    "cause_entity_id": str(source),
                    "effect_entity_id": str(target),
                    "causal_type": attrs.get("causal_type"),
                },
            )
            if key_s != expected_key:
                raise GraphBundleValidationError(f"causal edge stable identity mismatch for {key_s!r}")
        else:
            raise GraphBundleValidationError(f"edge {key_s!r} has unknown/missing role {role!r}")
    if len(keyed_edges) != len(set(keyed_edges)):
        raise GraphBundleValidationError("duplicate keyed graph edges detected")





    _validate_replay_graph_closure(graph, expected=expected_replay_graph)
    _validate_replay_vdb_closure(
        entity_records,
        expected_text=expected_replay_graph["entity_text"],
        embedding_signature=manifest["embedding_signatures"]["entity"],
        role="entity",
    )
    _validate_replay_vdb_closure(
        hyperedge_records,
        expected_text=expected_replay_graph["hyperedge_text"],
        expected_source_ids=expected_replay_graph["hyperedge_source_ids"],
        embedding_signature=manifest["embedding_signatures"]["hyperedge"],
        role="hyperedge_occurrence",
    )

    ledger = _json_attr(graph.graph.get("support_ledger"), field="support_ledger")
    corpus_counts = _json_attr(
        graph.graph.get("recurrence_corpus_counts"), field="recurrence_corpus_counts"
    )
    if not isinstance(ledger, list) or not isinstance(corpus_counts, Mapping):
        raise GraphBundleValidationError("support ledger/corpus counts have invalid types")
    ledger_order = []
    support_units = set()
    observed_counts: Dict[str, int] = {}
    for entry in ledger:
        if not isinstance(entry, Mapping):
            raise GraphBundleValidationError("support ledger entry is not an object")
        proposition_key = entry.get("proposition_key")
        record_id = entry.get("record_id")
        hyperedge_id = entry.get("hyperedge_id")
        order_key = (proposition_key, record_id, hyperedge_id)
        if not all(isinstance(item, str) and item for item in order_key):
            raise GraphBundleValidationError("support ledger entry has invalid identity")
        ledger_order.append(order_key)
        unit = (proposition_key, record_id)
        if unit in support_units:
            raise GraphBundleValidationError("support ledger double-counts a record/proposition unit")
        support_units.add(unit)
        if hyperedge_id not in hyperedge_nodes:
            raise GraphBundleValidationError("support ledger refers to a missing hyperedge")
        attrs = graph.nodes[hyperedge_id]
        if attrs.get("proposition_key") != proposition_key or attrs.get("record_id") != record_id:
            raise GraphBundleValidationError("support ledger disagrees with hyperedge metadata")
        if entry.get("chunk_id") != attrs.get("chunk_id"):
            raise GraphBundleValidationError("support ledger chunk provenance mismatch")
        observed_counts[proposition_key] = observed_counts.get(proposition_key, 0) + 1
    if ledger_order != sorted(ledger_order):
        raise GraphBundleValidationError("support ledger is not canonically sorted")
    if dict(sorted(observed_counts.items())) != dict(sorted((str(k), int(v)) for k, v in corpus_counts.items())):
        raise GraphBundleValidationError("recurrence corpus counts do not close over support ledger")

    declared_artifacts = manifest.get("artifacts")
    if not isinstance(declared_artifacts, Mapping) or set(declared_artifacts) != set(KG_ARTIFACTS):
        raise GraphBundleValidationError("manifest must declare exactly the four canonical KG artifacts")
    for name in KG_ARTIFACTS:
        if declared_artifacts[name] != artifact_info[name]:
            raise GraphBundleValidationError(f"artifact hash/size mismatch: {name}")
    if telemetry.get("artifacts") != declared_artifacts:
        raise GraphBundleValidationError("telemetry artifact closure mismatch")

    manifest_payload_sha256 = canonical_sha256(_manifest_self_payload(manifest))
    telemetry_payload_sha256 = canonical_sha256(_telemetry_self_payload(telemetry))
    if manifest.get("manifest_payload_sha256") != manifest_payload_sha256:
        raise GraphBundleValidationError("graph manifest self-exclusion payload hash mismatch")
    if telemetry.get("telemetry_payload_sha256") != telemetry_payload_sha256:
        raise GraphBundleValidationError("build telemetry self-exclusion payload hash mismatch")
    bundle_payload = graph_bundle_payload(manifest, telemetry, artifact_info)
    bundle_sha256 = canonical_sha256(bundle_payload)
    if manifest.get("graph_bundle_sha256") != bundle_sha256:
        raise GraphBundleValidationError("graph bundle identity mismatch in manifest")
    if telemetry.get("graph_bundle_sha256") != bundle_sha256:
        raise GraphBundleValidationError("graph bundle identity mismatch in telemetry")

    counts = {
        "chunks": len(chunks),
        "entities": len(entity_nodes),
        "hyperedges": len(hyperedge_nodes),
        "causal_edges": sum(
            1 for _, _, _, attrs in graph.edges(keys=True, data=True) if attrs.get("role") == "causal"
        ),
        "edges": graph.number_of_edges(),
    }
    declared_counts = telemetry.get("counts")
    if not isinstance(declared_counts, Mapping):
        raise GraphBundleValidationError("telemetry missing build counts")
    for key, value in counts.items():
        if declared_counts.get(key) != value:
            raise GraphBundleValidationError(f"telemetry count mismatch for {key}")

    return {
        "graph_dir": str(graph_dir),
        "profile_id": profile_id,
        "publishable": publishable,
        "out_of_paper_protocol": out_of_paper,
        "graph_bundle_sha256": bundle_sha256,
        "manifest": dict(manifest),
        "telemetry": dict(telemetry),
        "artifacts": artifact_info,
        "stats": counts,
        "embedding_signature": dict(entity_additional["embedding_signature"]),
        "embedding_signatures": {
            "entity": dict(entity_additional["embedding_signature"]),
            "hyperedge": dict(hyperedge_additional["embedding_signature"]),
        },
        "chunks": chunks_document,
        "source_audit": source_audit,
        "graph": graph,
        "entity_matrix_shape": tuple(entity_matrix.shape),
        "hyperedge_matrix_shape": tuple(hyperedge_matrix.shape),
    }


def validate_graph_bundle(
    path: os.PathLike[str] | str,
    *,
    expected_profile_id: Optional[str] = None,
    expected_config: Any = None,
    expected_config_sha256: Optional[str] = None,
    expected_construction_config_sha256: Optional[str] = None,
    require_publishable: bool = False,
    **ignored_identity_kwargs: Any,
) -> Dict[str, Any]:


    from chain.graph.completion import load_graph_bundle_portable

    return load_graph_bundle_portable(
        path,
        expected_profile_id=expected_profile_id,
        expected_config=expected_config,
        expected_config_sha256=expected_config_sha256,
        expected_construction_config_sha256=(
            expected_construction_config_sha256
        ),
        require_publishable=require_publishable,
        **ignored_identity_kwargs,
    )


def make_staging_directory(target: os.PathLike[str] | str) -> Path:


    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target_path.name}.staging-", dir=target_path.parent))


def atomic_publish_directory(staging: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:


    staging_path = Path(staging)
    target_path = Path(target)
    if not staging_path.is_dir():
        raise AtomicPublishError(f"staging directory does not exist: {staging_path}")
    if staging_path.parent.resolve() != target_path.parent.resolve():
        raise AtomicPublishError("staging and target must be siblings on the same filesystem")

    backup: Optional[Path] = None
    if target_path.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{target_path.name}.backup-", dir=target_path.parent))
        backup.rmdir()
        try:
            os.replace(target_path, backup)
        except Exception as exc:
            raise AtomicPublishError(f"cannot move old target to backup: {exc}") from exc
    try:
        os.replace(staging_path, target_path)
    except Exception as exc:
        restore_error: Optional[Exception] = None
        if backup is not None and backup.exists():
            try:
                os.replace(backup, target_path)
            except Exception as restore_exc:
                restore_error = restore_exc
        suffix = f"; rollback also failed: {restore_error}" if restore_error else ""
        raise AtomicPublishError(f"cannot publish staged directory: {exc}{suffix}") from exc
    _fsync_directory(target_path.parent)
    if backup is not None and backup.exists():
        try:
            shutil.rmtree(backup)
            _fsync_directory(target_path.parent)
        except OSError as exc:


            warnings.warn(
                f"published graph successfully but retained recoverable backup {backup}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:


    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "ALLOWED_CAUSAL_TYPES",
    "AtomicPublishError",
    "CHUNKS_SCHEMA",
    "GRAPH_BUNDLE_SCHEMA",
    "GRAPH_SCHEMA",
    "GraphBundleValidationError",
    "KG_ARTIFACTS",
    "SOURCE_AUDIT_FILE",
    "SOURCE_AUDIT_SCHEMA",
    "TELEMETRY_SCHEMA",
    "VECTOR_SCHEMA",
    "atomic_publish_directory",
    "atomic_write_json",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "graph_bundle_payload",
    "make_staging_directory",
    "sha256_file",
    "validate_graph_bundle",
    "validate_graph_bundle_forensic",
    "write_canonical_json",
]
