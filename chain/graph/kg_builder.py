from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit, urlunsplit

from chain.config import EXTRACTION_RESPONSE_FORMAT_VERSION
from chain.graph.validation import (
    ALLOWED_CAUSAL_TYPES,
    CHUNKS_SCHEMA,
    GRAPH_BUNDLE_SCHEMA,
    GRAPH_SCHEMA,
    KG_ARTIFACTS,
    SOURCE_AUDIT_FILE,
    SOURCE_AUDIT_SCHEMA,
    TELEMETRY_SCHEMA,
    VECTOR_SCHEMA,
    GraphBundleValidationError,
    atomic_publish_directory,
    canonical_json,
    canonical_sha256,
    graph_bundle_payload,
    make_staging_directory,
    sha256_file,
    write_canonical_json,
)
from chain.graph.completion import EMBEDDING_SPACE_FIELDS, load_graph_bundle_portable



validate_graph_bundle = load_graph_bundle_portable

logger = logging.getLogger(__name__)

GRAPH_FIELD_SEP = "<SEP>"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv"}
COMPAT_PROFILE_ID = "compat"
PAPER_PROFILE_ID = "paper"
CONSTRUCTION_RECORD_SCHEMA = "chain-construction-record-v1"
CONSTRUCTION_CONFIG_SCHEMA = "chain-construction-config-v2"
CHUNK_LAYOUT_VERSION = "chain-record-local-token-chunks-v1"




CONTENT_NORMALIZATION_VERSION = "unicode-nfc-content-answer-safe-v3"
KEY_NORMALIZATION_VERSION = "unicode-nfkc-casefold-whitespace-key-v1"
EXTRACTION_SCHEMA_VERSION = "chain-extraction-schema-v6"
RELATION_ADMISSIBILITY_POLICY_VERSION = (
    "chain-final-retry-relation-admissibility-v2"
)
GROUNDING_POLICY_VERSION = "chain-source-grounding-proposition-omission-v3"




PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION = (
    "chain-final-retry-empty-or-duplicate-canonical-entity-proposition-omission-v2"
)



EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT = 3
PROMPT_VERSION = "chain-extraction-prompt-v17"
COMPAT_RETRIEVAL_VERSION = "chain-compat-rrf-v1"
COMPAT_RRF_K = 4
COMPAT_RRF_DEFAULT_TOP_K = 10
COMPAT_RRF_CHANNEL_CANDIDATE_MULTIPLIER = 2
_RESERVED_VECTOR_META_KEYS = frozenset({"__id__", "__vector__", "__metrics__"})





_COSINE_BOUND_TOLERANCE = 1e-6


_RESERVED_EDGE_ATTR_KEYS = frozenset({"target", "key", "edge_id"})
_ENCODERS: Dict[str, Any] = {}
_SKIP_DIRS = frozenset(
    {".chain", ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode", "expr"}
)
_CONSTRUCTION_METADATA_FIELDS = ("domain", "category", "title", "url", "publication", "author")

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
        raise ExtractionSchemaError(
            f"{context} contains a character that cannot be represented in XML 1.0 GraphML"
        )


class GraphBuildError(RuntimeError):
    pass



class InputRecordError(GraphBuildError):
    pass



class ExtractionSchemaError(GraphBuildError):
    pass



class _RelationSchemaError(ExtractionSchemaError):


    def __init__(
        self, message: str, *, reason_code: str, details: Optional[Mapping[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ArtifactCorruptionError(GraphBuildError):
    pass



class ExtractionAttemptError(GraphBuildError):


    def __init__(self, message: str, *, trace: Any = None, raw_response: Any = None) -> None:
        super().__init__(message)
        self.provider_trace = dict(trace) if isinstance(trace, Mapping) else None
        self.raw_response = raw_response if isinstance(raw_response, str) else None
        self.response_sha256 = (
            _content_sha256(self.raw_response) if self.raw_response is not None else None
        )


class _SyntheticTokenizer:


    name = "chain-test-whitespace-v1"

    def encode(self, text: str) -> List[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(int(token)) for token in tokens)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip())


def canonical_entity_key(value: str) -> str:
    if not isinstance(value, str):
        raise ExtractionSchemaError("entity names must be strings")
    result = _normalise_text(value).casefold()
    if not result:
        raise ExtractionSchemaError("entity names must be non-empty")
    return result


def canonical_proposition_key(value: str) -> str:
    if not isinstance(value, str):
        raise ExtractionSchemaError("proposition sentences must be strings")
    result = _normalise_text(value).casefold()
    if not result:
        raise ExtractionSchemaError("proposition sentences must be non-empty")
    return result


def _typed_sha256(prefix: str, value: Any) -> str:
    return prefix + canonical_sha256(value)


def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_audit_sha256(value: Any) -> str:


    try:
        return canonical_sha256(value)
    except (TypeError, ValueError, OverflowError):
        def seal(item: Any) -> Any:
            if isinstance(item, float) and not math.isfinite(item):
                return {"__chain_nonfinite_float__": repr(item)}
            if isinstance(item, Mapping):
                return {
                    str(key): seal(child)
                    for key, child in item.items()
                }
            if isinstance(item, (list, tuple)):
                return [seal(child) for child in item]
            return item

        return canonical_sha256(
            {
                "schema_version": "chain-audit-nonfinite-value-v1",
                "value": seal(value),
            }
        )


def _normalise_relative_path(value: str) -> str:
    path = PurePosixPath(unicodedata.normalize("NFKC", value.replace("\\", "/")))
    if path.is_absolute() or ".." in path.parts:
        raise InputRecordError(f"corpus path must be relative: {value!r}")
    result = path.as_posix().lstrip("./")
    if not result or result == ".":
        raise InputRecordError("corpus-relative path must be non-empty")
    return result


def _normalise_source(value: str, *, fallback: str) -> str:
    result = _normalise_text(value or fallback).replace("\\", "/")
    path = PurePosixPath(result)
    windows_path = PureWindowsPath(result)
    if (
        "\x00" in result
        or path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or urlsplit(result).scheme.casefold() == "file"
        or ".." in path.parts
    ):
        raise InputRecordError(f"source identity must be relative: {result!r}")
    result = path.as_posix().lstrip("./")
    if not result or result == ".":
        raise InputRecordError("source identity must be non-empty")
    return result


def _source_identity(value: Any, *, fallback: str, field: str = "source") -> str:


    if value in (None, ""):
        value = fallback
    if not isinstance(value, str):
        raise InputRecordError(f"{field} must be a string")
    return _normalise_source(value, fallback=fallback)


def normalize_source_descriptor(value: Any) -> str:


    if not isinstance(value, str) or not value.strip():
        raise InputRecordError("source_descriptor must be a non-empty string")
    return _source_identity(
        value.strip(), fallback="source", field="source_descriptor"
    )


def _parse_timestamp(raw_value: Any) -> Tuple[str, str, str]:
    if raw_value is None:
        return "", "", ""
    if not isinstance(raw_value, str):
        raise InputRecordError("record timestamps must be strings or null")
    raw = raw_value.strip()
    if not raw:
        return "", "", ""
    from chain.skills.temporal_validity import parse_temporal

    parsed = parse_temporal(raw)
    if parsed is None:
        return raw, "", ""
    derived_date = (
        parsed.value.isoformat()
        if parsed.precision == "date"
        else parsed.value.date().isoformat()
    )
    return raw, parsed.canonical, derived_date


def _select_timestamp(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    candidates = [
        (field, record[field])
        for field in ("timestamp_raw", "timestamp", "event_time", "published_at", "date")
        if record.get(field) not in (None, "")
    ]
    if not candidates:
        return "", "", ""
    primary = _parse_timestamp(candidates[0][1])
    for field, value in candidates[1:]:
        other = _parse_timestamp(value)
        if field == "date" and primary[2] and other[2] == primary[2]:
            continue
        if other != primary:
            raise InputRecordError("conflicting timestamp aliases in one record")
    return primary


def _availability_fields(record: Mapping[str, Any]) -> Dict[str, str]:
    iso_raw = record.get("availability_upper_bound_iso") or record.get(
        "availability_upper_bound"
    )
    date_raw = record.get("availability_upper_bound_date")
    if iso_raw not in (None, ""):
        _, bound_iso, bound_date = _parse_timestamp(iso_raw)
        if not bound_iso:
            raise InputRecordError("availability upper bound is not valid ISO date/time")
        if date_raw not in (None, ""):
            _, date_iso, derived_date = _parse_timestamp(date_raw)
            if not date_iso or derived_date != bound_date:
                raise InputRecordError("conflicting availability ISO/date aliases")
    elif date_raw not in (None, ""):
        _, bound_iso, bound_date = _parse_timestamp(date_raw)
        if not bound_iso:
            raise InputRecordError("availability upper-bound date is invalid")
    else:
        bound_iso = bound_date = ""
    kind = record.get("availability_bound_kind", "")
    source = record.get("availability_bound_source", "")
    source_hash = record.get("availability_bound_source_sha256", "")
    if not all(isinstance(value, str) for value in (kind, source, source_hash)):
        raise InputRecordError("availability provenance fields must be strings")
    if bound_iso and (not kind.strip() or not source.strip() or not re.fullmatch(r"[0-9a-f]{64}", source_hash)):
        raise InputRecordError("availability bound requires kind/source/source SHA-256 provenance")
    if not bound_iso and any(value.strip() for value in (kind, source, source_hash)):
        raise InputRecordError("availability provenance cannot exist without an upper bound")
    source_descriptor = _normalise_availability_source(source) if source else ""
    return {
        "availability_upper_bound_iso": bound_iso,
        "availability_upper_bound_date": bound_date,
        "availability_bound_kind": kind.strip(),
        "availability_bound_source": source_descriptor,
        "availability_bound_source_sha256": source_hash,
    }


def _normalise_availability_source(value: str) -> str:


    if not isinstance(value, str):
        raise InputRecordError("availability_bound_source must be a string")
    raw = unicodedata.normalize("NFC", value).strip()
    if not raw or "\x00" in raw:
        raise InputRecordError("availability_bound_source must be non-empty")
    if raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", raw):
        raise InputRecordError(
            "availability_bound_source must not contain an absolute machine-local path"
        )
    parsed = urlsplit(raw)
    if parsed.scheme:
        scheme = parsed.scheme.casefold()
        if scheme == "file":
            raise InputRecordError("availability_bound_source forbids file:// paths")
        if scheme in {"http", "https"}:
            if not parsed.netloc or parsed.username or parsed.password:
                raise InputRecordError("availability_bound_source URL is malformed or contains userinfo")
            return urlunsplit(
                (scheme, parsed.netloc.casefold(), parsed.path, parsed.query, parsed.fragment)
            )
        if scheme not in {"doi", "urn", "isbn", "ledger", "manifest", "dataset", "citation"}:
            raise InputRecordError(
                f"availability_bound_source uses unsupported descriptor scheme {parsed.scheme!r}"
            )
    path_parts = PurePosixPath(raw.replace("\\", "/")).parts
    if ".." in path_parts:
        raise InputRecordError("availability_bound_source must not contain parent traversal")
    return _normalise_text(raw)


def _record_content(record: Mapping[str, Any]) -> str:
    background, content = record.get("background"), record.get("content")
    for name, value in (("background", background), ("content", content)):
        if value not in (None, "") and not isinstance(value, str):
            raise InputRecordError(f"{name} must be a string")
    if isinstance(background, str) and background.strip() and isinstance(content, str) and content.strip():
        if background.strip() != content.strip():
            raise InputRecordError("record has conflicting background/content aliases")
    selected = background if isinstance(background, str) and background.strip() else content
    selected = selected.strip() if isinstance(selected, str) else ""
    if not selected:
        raise InputRecordError("canonical records require non-empty factual background/content")
    return _sanitize_construction_content(selected)


_LEADING_QUESTION_RE = re.compile(
    r"^\s*(?:[-*_]\s*)?QUESTION\s*:\s*",
    flags=re.IGNORECASE,
)
_FACTUAL_SECTION_RE = re.compile(
    r"^\s*(?:CONTEXT|BACKGROUND|EVIDENCE|NEWS)\s*:\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ANSWER_SECTION_RE = re.compile(
    r"^[ \t]*(?:"
    r"\[\s*(?:RESOLUTION|ANSWER|OUTCOME)\s*\]"
    r"|(?:RESOLUTION|ANSWER|OUTCOME)\s*:"
    r"|ANSWER[ \t]+FORMAT\s*:"
    r"|GROUND[ _-]*TRUTH\s*:?"
    r")",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _sanitize_construction_content(value: str) -> str:


    text = unicodedata.normalize("NFC", value)
    leading_question = _LEADING_QUESTION_RE.match(text)
    if leading_question:
        boundary = _FACTUAL_SECTION_RE.search(text, leading_question.end())
        if boundary is None:
            raise InputRecordError(
                "leading QUESTION block has no factual section boundary"
            )
        text = text[boundary.end() :]
    answer_heading = _ANSWER_SECTION_RE.search(text)
    if answer_heading:
        text = text[: answer_heading.start()]
    text = text.strip()
    if not text:
        raise InputRecordError(
            "record contains no factual construction content after safety sanitization"
        )
    return text


def construction_payload_for_record(record: Mapping[str, Any], *, relative_path: str) -> Dict[str, Any]:
    relative = _normalise_relative_path(relative_path)
    timestamp_raw, timestamp_iso, event_date = _select_timestamp(record)
    metadata: Dict[str, Any] = {}
    for field in _CONSTRUCTION_METADATA_FIELDS:
        if record.get(field) in (None, ""):
            continue
        value = record[field]
        if isinstance(value, float) and not math.isfinite(value):
            raise InputRecordError(f"construction metadata {field!r} is non-finite")
        if not isinstance(value, (str, int, float, bool)):
            raise InputRecordError(f"construction metadata {field!r} must be scalar")
        metadata[field] = _normalise_text(value) if isinstance(value, str) else value
    provenance = record.get("construction_provenance", []) or []
    if not isinstance(provenance, list) or any(not isinstance(item, Mapping) for item in provenance):
        raise InputRecordError("construction_provenance must be a list of objects")
    canonical_provenance = []
    for item in provenance:
        if set(item) - {"source_path", "record_line_number", "source"}:
            raise InputRecordError("construction_provenance has unapproved fields")
        source_path = item.get("source_path", "")
        if not isinstance(source_path, str):
            raise InputRecordError("provenance source_path must be a string")
        path = _normalise_relative_path(source_path)
        line = item.get("record_line_number")
        if isinstance(line, bool) or not isinstance(line, int) or line <= 0:
            raise InputRecordError("provenance line numbers must be positive integers")
        canonical_provenance.append(
            {
                "source_path": path,
                "record_line_number": line,
                "source": _source_identity(
                    item.get("source"),
                    fallback=PurePosixPath(path).stem,
                    field="provenance source",
                ),
            }
        )
    canonical_provenance.sort(key=lambda item: (item["source_path"], item["record_line_number"], item["source"]))
    return {
        "schema_version": CONSTRUCTION_RECORD_SCHEMA,
        "content": _record_content(record),
        "source": _source_identity(
            record.get("source"),
            fallback=PurePosixPath(relative).stem,
            field="source",
        ),
        "timestamp_raw": timestamp_raw,
        "timestamp_iso": timestamp_iso,
        "date": event_date,
        **_availability_fields(record),
        "metadata": metadata,
        "construction_provenance": canonical_provenance,
    }


def parse_source_record(record: Mapping[str, Any], *, relative_path: str, line_number: int) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise InputRecordError("each JSONL line must contain one object")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number <= 0:
        raise InputRecordError("record line number must be a positive integer")
    relative = _normalise_relative_path(relative_path)
    raw_payload = dict(record)
    construction = construction_payload_for_record(record, relative_path=relative)
    construction_hash = canonical_sha256(construction)
    return {
        "raw_record_payload": raw_payload,
        "raw_record_sha256": canonical_sha256(raw_payload),
        "construction_payload": construction,
        "construction_payload_sha256": construction_hash,
        "record_sha256": construction_hash,
        "record_id": _typed_sha256(
            "rec_",
            {
                "source_path": relative,
                "record_line_number": line_number,
                "construction_payload_sha256": construction_hash,
            },
        ),
        "record_line_number": line_number,
        "source_path": relative,
        "input_id": str(record.get("id", "")),
    }


def _strict_json_loads(raw: str, *, context: str) -> Any:
    def pairs_hook(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InputRecordError(f"duplicate JSON key {key!r} in {context}")
            result[key] = value
        return result

    def strict_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise InputRecordError(
                f"non-finite JSON number {value!r} in {context}"
            )
        return parsed

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_float=strict_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InputRecordError(f"non-finite JSON constant {value!r} in {context}")
            ),
        )
    except InputRecordError:
        raise
    except Exception as exc:
        raise InputRecordError(f"invalid JSON in {context}: {exc}") from exc


def _strict_artifact_json_loads(raw: str, *, context: str) -> Any:


    try:
        return _strict_json_loads(raw, context=context)
    except InputRecordError as exc:
        raise ArtifactCorruptionError(str(exc)) from exc


def _source_files(source: Path, *, allow_compat_adapters: bool) -> Tuple[Path, List[Path]]:
    if not source.exists():
        raise FileNotFoundError(f"knowledge source does not exist: {source}")
    allowed = SUPPORTED_EXTENSIONS if allow_compat_adapters else {".jsonl"}
    if source.is_file():
        if source.suffix.lower() not in allowed:
            raise InputRecordError(f"unsupported source format: {source.suffix or '<none>'}")
        return source.parent, [source]
    files: List[Path] = []
    for root, dirs, names in os.walk(source):
        dirs[:] = sorted(item for item in dirs if item not in _SKIP_DIRS)
        for name in sorted(names):
            path = Path(root) / name
            if path.suffix.lower() in allowed:
                files.append(path)
    files.sort(key=lambda path: path.relative_to(source).as_posix())
    if not files:
        raise InputRecordError(f"no supported knowledge records found in {source}")
    return source, files


def _compat_file_records(path: Path, relative: str) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        yield 1, {"content": path.read_text(encoding="utf-8"), "source": PurePosixPath(relative).stem}
        return
    if suffix == ".json":
        value = _strict_json_loads(path.read_text(encoding="utf-8"), context=relative)
        for index, item in enumerate(value if isinstance(value, list) else [value], 1):
            if isinstance(item, Mapping) and any(key in item for key in ("background", "content")):
                yield index, dict(item)
            else:
                yield index, {"content": canonical_json(item), "source": PurePosixPath(relative).stem}
        return
    raise InputRecordError(f"compat adapter does not support {suffix}")


def load_source_records(
    source: os.PathLike[str] | str,
    *,
    allow_compat_adapters: bool = False,
    source_descriptor: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_path = Path(source)
    descriptor = None
    if source_descriptor is not None:
        descriptor = normalize_source_descriptor(source_descriptor)
    root, files = _source_files(source_path, allow_compat_adapters=allow_compat_adapters)
    records: List[Dict[str, Any]] = []
    input_entries: List[Dict[str, Any]] = []
    audit_records: List[Dict[str, Any]] = []
    for file_path in files:
        relative = _normalise_relative_path(file_path.relative_to(root).as_posix())
        if file_path.suffix.lower() == ".jsonl":
            parsed_lines = []
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    if not raw_line.strip():
                        continue
                    value = _strict_json_loads(raw_line, context=f"{relative}:{line_number}")
                    if not isinstance(value, Mapping):
                        raise InputRecordError(f"{relative}:{line_number} is not a JSON object")
                    parsed_lines.append((line_number, value))
            iterable: Iterable[Tuple[int, Mapping[str, Any]]] = parsed_lines
        else:
            if not allow_compat_adapters:
                raise InputRecordError("canonical/paper graph builds accept JSONL only")
            iterable = _compat_file_records(file_path, relative)
        file_count = 0
        for line_number, raw_record in iterable:
            if descriptor is not None and not raw_record.get("source"):
                raw_record = {**dict(raw_record), "source": descriptor}
            parsed = parse_source_record(raw_record, relative_path=relative, line_number=line_number)
            records.append(parsed)
            file_count += 1
            audit_records.append(
                {
                    "source_path": relative,
                    "record_line_number": line_number,
                    "input_id": parsed["input_id"],
                    "raw_record_sha256": parsed["raw_record_sha256"],
                    "construction_payload_sha256": parsed["construction_payload_sha256"],
                    "record_id": parsed["record_id"],
                }
            )
        input_entries.append(
            {
                "source_path": relative,
                "raw_input_file_sha256": sha256_file(file_path),
                "bytes": file_path.stat().st_size,
                "records": file_count,
            }
        )
    if not records:
        raise InputRecordError("knowledge source contains no canonical records")
    construction_stream = [
        {
            "source_path": record["source_path"],
            "record_line_number": record["record_line_number"],
            "construction_payload": record["construction_payload"],
            "construction_payload_sha256": record["construction_payload_sha256"],
            "record_id": record["record_id"],
        }
        for record in records
    ]
    audit = {
        "schema_version": SOURCE_AUDIT_SCHEMA,
        "inputs": input_entries,
        "records": audit_records,
        "raw_input_set_sha256": canonical_sha256(input_entries),
        "raw_record_set_sha256": canonical_sha256(audit_records),
        "canonical_construction_stream_sha256": canonical_sha256(construction_stream),
        "record_count": len(records),
    }
    return records, audit


def _get_encoder(model: str) -> Tuple[Any, str, str]:
    if not isinstance(model, str) or not model.strip():
        raise GraphBuildError("tokenizer model must be non-empty")
    if model in _ENCODERS:
        return _ENCODERS[model]
    if model == "chain-test-whitespace-v1":
        result = (_SyntheticTokenizer(), model, "chain-test-tokenizer-v1")
        _ENCODERS[model] = result
        return result
    try:
        import tiktoken

        encoder = tiktoken.encoding_for_model(model)
        result = (encoder, encoder.name, getattr(tiktoken, "__version__", "unknown"))
    except Exception as exc:
        raise GraphBuildError(f"cannot resolve tokenizer for {model!r}: {exc}") from exc
    _ENCODERS[model] = result
    return result


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64, model: str = "gpt-4o") -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        raise TypeError("chunk text must be a string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_tokens:
        raise ValueError("overlap must be in [0,max_tokens)")
    encoder, _, _ = _get_encoder(model)
    tokens = encoder.encode(text)
    if not tokens:
        return []
    step = max_tokens - overlap
    result = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        content = encoder.decode(tokens[start:end])
        if content.strip():
            result.append(
                {
                    "content": content,
                    "tokens": end - start,
                    "chunk_index": len(result),
                    "token_start": start,
                    "token_end": end,
                }
            )
        if end >= len(tokens):
            break
        start += step
    return result


def record_to_chunks(
    record: Mapping[str, Any],
    *,
    graph_namespace: str,
    chunk_index_start: int,
    max_tokens: int = 512,
    overlap: int = 64,
    tokenizer_model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    construction = record["construction_payload"]
    result = []
    for local_index, raw_chunk in enumerate(
        chunk_text(str(construction["content"]), max_tokens, overlap, tokenizer_model)
    ):
        content_hash = _content_sha256(raw_chunk["content"])
        chunk_id = _typed_sha256(
            "chk_",
            {
                "graph_namespace": graph_namespace,
                "record_id": record["record_id"],
                "record_chunk_index": local_index,
                "content_sha256": content_hash,
            },
        )
        result.append(
            {
                **raw_chunk,
                "chunk_index": chunk_index_start + local_index,
                "record_chunk_index": local_index,
                "chunk_id": chunk_id,
                "record_id": record["record_id"],
                "record_line_number": record["record_line_number"],
                "source": construction["source"],
                "source_path": record["source_path"],
                "timestamp_raw": construction["timestamp_raw"],
                "timestamp_iso": construction["timestamp_iso"],
                "date": construction["date"],
                "availability_upper_bound_iso": construction["availability_upper_bound_iso"],
                "availability_upper_bound_date": construction["availability_upper_bound_date"],
                "availability_bound_kind": construction["availability_bound_kind"],
                "availability_bound_source": construction["availability_bound_source"],
                "availability_bound_source_sha256": construction["availability_bound_source_sha256"],
                "construction_payload_sha256": record["construction_payload_sha256"],
                "record_sha256": record["construction_payload_sha256"],
                "content_sha256": content_hash,
                "layout_version": CHUNK_LAYOUT_VERSION,
                "normalization_version": CONTENT_NORMALIZATION_VERSION,
                "key_normalization_version": KEY_NORMALIZATION_VERSION,
            }
        )
    return result


def records_to_chunks(
    records: Sequence[Mapping[str, Any]],
    *,
    graph_namespace: str,
    max_tokens: int = 512,
    overlap: int = 64,
    tokenizer_model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for record in records:
        result.extend(
            record_to_chunks(
                record,
                graph_namespace=graph_namespace,
                chunk_index_start=len(result),
                max_tokens=max_tokens,
                overlap=overlap,
                tokenizer_model=tokenizer_model,
            )
        )
    return result


def is_supported(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def ingest_file(file_path: str, max_tokens: int = 512, overlap: int = 64, **kwargs: Any) -> Tuple[List[Dict], List[Dict]]:
    records, audit = load_source_records(file_path, allow_compat_adapters=True)
    tokenizer_model = str(kwargs.get("tokenizer_model") or kwargs.get("model") or "gpt-4o")
    namespace = kwargs.get("graph_namespace") or _typed_sha256(
        "graph_",
        {
            "canonical_construction_stream_sha256": audit["canonical_construction_stream_sha256"],
            "chunk_max_tokens": max_tokens,
            "chunk_overlap_tokens": overlap,
            "tokenizer_model": tokenizer_model,
        },
    )
    return (
        records_to_chunks(
            records,
            graph_namespace=str(namespace),
            max_tokens=max_tokens,
            overlap=overlap,
            tokenizer_model=tokenizer_model,
        ),
        list(audit["records"]),
    )


_ALLOWED_CAUSAL_TYPES_PROMPT = ", ".join(
    f'"{relation_type}"' for relation_type in sorted(ALLOWED_CAUSAL_TYPES)
)

_EXTRACT_PROMPT = """\
Extract factual propositions and causal relationships from the supplied text.

Return exactly one JSON object with exactly "propositions" and
"causal_relations" arrays. Do not output any other top-level field.

The response MUST be one RFC 8259 JSON value. Every string value MUST use
valid JSON escaping. A literal backslash from the supplied Text MUST be
encoded as two consecutive backslash characters in the JSON encoding; never
emit an unsupported escape such as a single backslash followed by a letter,
underscore, or dollar sign. Do not emit raw LaTeX commands inside JSON
strings. Use plain Unicode/text when possible; when a verbatim source phrase
contains a backslash, preserve it only with valid JSON escaping. After JSON
decoding, the resulting string value must still satisfy every verbatim
grounding rule below. Never use a single JSON escape such as \\b, \\f, \\n,
\\r, or \\t to represent the beginning of a source LaTeX command: those
escapes decode to control characters rather than a literal backslash. Every
decoded string must contain no XML 1.0-forbidden control character. Check the
complete response with a strict JSON parser before returning it.

Every proposition object MUST contain exactly:
{{"sentence":"non-empty factual sentence","entities":[...]}}
Do not output any other proposition field. Every entity object MUST contain
either exactly {{"name":"non-empty entity name"}} or exactly
{{"name":"non-empty entity name","name_zh":"optional Chinese name"}}.
Do not output entity types, IDs, descriptions, aliases, confidence, or any
other entity field. Every emitted entity "name" MUST copy, byte-for-byte, one
complete contiguous substring from the supplied Text, including its exact
capitalization, punctuation, Unicode form, and internal whitespace. The
"entities" array MUST be non-empty (at least one entity). If ANY emitted entity
in a proposition cannot be grounded this way, OMIT THAT ENTIRE PROPOSITION;
never return an empty "entities" array, delete only the bad entity, or invent,
repair, normalize, paraphrase, translate, or complete an entity from a
truncated or incomplete sentence.

ENTITY SURFACE-FORM RULE: when the same entity appears in multiple
propositions, reuse one consistent spelling, capitalization, Unicode form,
and whitespace pattern for its name. Do not emit separate entity names that
differ only under case-folding, Unicode compatibility normalization, or
whitespace normalization. The builder retains any such surface forms as
aliases under one canonical entity identity; this does not merge names with
different canonical keys. Within one proposition, list each canonical entity
only once. If two distinct referents could otherwise collapse under this
normalization, use explicit disambiguating names such as "Apple Inc." and
"apple fruit" rather than case alone.

Every causal relation object MUST contain exactly the required fields
"cause", "effect", "type", and "strength", and MAY also contain the single
optional string field "description". Do not output any other relation field.
The "strength" value is a finite JSON number strictly in (0,1]. Estimate it
independently for each relation from the supplied text; do not copy a fixed
number from these instructions or reuse one default value across relations.

CRITICAL ENDPOINT IDENTITY RULE: treat the emitted
propositions[*].entities[*].name strings as a closed vocabulary. Every
cause/effect MUST copy, byte-for-byte, one complete name string from that
vocabulary in this same response. Never use a complete sentence, pronoun,
summary, description, alias, normalized spelling, or paraphrase as an
endpoint. If you cannot confirm an exact byte-for-byte match, OMIT the causal
relation; do not invent, repair, shorten, or rewrite an endpoint. If a compound
event is needed, first emit it as an entity name and then copy that exact name.
The supplied Text may end in the middle of a word or sentence. Never complete,
repair, or infer missing text after a truncation; omit any fact or causal
relation that cannot be supported without the missing continuation.
Before returning JSON, self-check every causal relation by locating both
endpoint strings in the emitted entity names and confirming exact equality,
including capitalization, punctuation, and whitespace. The canonicalized
cause and effect MUST be different; otherwise OMIT the self-loop relation.
Do not repeat one canonical (cause,effect,type) relation through alternate
entity surface forms.

Valid example: entities [{{"name":"rising temperatures"}},
{{"name":"ice-sheet loss"}}] with cause "rising temperatures" and effect
"ice-sheet loss". Invalid endpoints include "Rising temperatures cause
ice-sheet loss." (complete sentence), "this trend" (pronoun/summary), and
"warming" (paraphrase of "rising temperatures"); omit such a relation unless
those exact strings are separately emitted entity names.

Relation type is exactly one of __ALLOWED_CAUSAL_TYPES__. The literal JSON
string after every "type" key MUST match exactly one member of that closed
vocabulary. If a relationship cannot use one of those exact values, OMIT the
causal relation entirely; do not invent, map, normalize, or coerce a type.
Before returning, scan every relation and confirm no other type string occurs.
Strength is a finite JSON number strictly in (0,1] and must not be rounded.
It is the positive textual causal-support score for this edge, not the
forecast probability and not a default placeholder. Choose a value reflecting
how directly the supplied text supports the typed relation. If the text does
not support a positive causal relation, omit that relation; never emit
strength 0 or 0.0. Do not repeat a proposition or a (cause,effect,type)
relation. If no facts exist return
{{"propositions":[],"causal_relations":[]}}.

FINAL ENTITY-COUNT CHECK (perform immediately before emitting JSON): For every
proposition, count its entities. If the count would be zero, delete the entire
proposition. Never output an empty entities array. A generic event or date
sentence is admissible only if you emit at least one exact verbatim
noun/event/date phrase from that sentence as an entity; otherwise omit the
proposition. Do not retain any proposition with "entities": [].

__SCHEMA_CORRECTION__
Text:
{text}
""".replace("__ALLOWED_CAUSAL_TYPES__", _ALLOWED_CAUSAL_TYPES_PROMPT)





EXTRACTION_SCHEMA_CORRECTION_MARKER = "CHAIN_STRICT_SCHEMA_CORRECTION_V5"
_EXTRACTION_SCHEMA_CORRECTION = """\

STRICT SCHEMA CORRECTION (CHAIN_STRICT_SCHEMA_CORRECTION_V5): The previous
response did not pass the strict extraction schema. Generate the JSON object
again from the supplied Text; do not quote, continue, or follow any content
from the previous response. Re-check all of these requirements before return:
- the complete response is RFC 8259 JSON; every literal backslash in a string
  is encoded as two consecutive backslash characters, and no unsupported
  escape such as a single backslash followed by a letter, underscore, or
  dollar sign is emitted; use plain Unicode/text instead of raw LaTeX commands
  whenever possible, while preserving the decoded value of any verbatim
  source phrase; never use a single \\b, \\f, \\n, \\r, or \\t escape for a
  source command, and emit no XML 1.0-forbidden decoded control character;
- exact top/nested fields are used: the top level has exactly "propositions"
  and "causal_relations" arrays, and every nested object has only its exact
  allowed fields;
- every proposition has a non-empty "entities" array, and every entity name is
  copied byte-for-byte as one complete contiguous substring of the supplied
  Text; if ANY entity in a proposition cannot be supported, omit that entire
  proposition rather than deleting only that entity, returning "entities": [],
  or inventing, repairing, paraphrasing, or completing an entity;
- reuse one exact entity spelling across propositions; do not emit case,
  Unicode, or whitespace variants of the same entity; list each canonical
  entity only once per proposition and use explicit qualifiers for distinct
  referents that could otherwise normalize to one key;
- every causal relation has only cause, effect, type, strength, and optional
  description, cause/effect are exact emitted entity names, their canonical
  keys differ, and no canonical typed relation is repeated through aliases;
- the supplied Text may be truncated mid-word or mid-sentence; never complete
  or infer the missing continuation, and omit facts/relations that need it;
- the exact value after every "type" key matches
  ^(causes|enables|prevents)$; otherwise omit that entire relation, and use an
  empty causal_relations array when none qualify;
- strength is a finite JSON number strictly in (0,1], never a boolean or
  string; zero means no supported edge and must be omitted.
FINAL CHECK: count every proposition's entities immediately before return. If
the count is zero, delete that proposition. Never output "entities": []. A
generic event/date sentence is valid only when at least one exact verbatim
noun/event/date phrase from it is emitted as an entity; otherwise omit it.
Return exactly one newly generated JSON object and no surrounding text.
"""

_EXTRACTION_SYSTEM = (
    "You extract one strict causal-temporal JSON object and must perform the "
    "final RFC 8259 escaping check before returning it. Every literal "
    "backslash in a JSON string must be encoded as two consecutive backslash "
    "characters; never emit an unsupported escape or raw LaTeX command. Also "
    "perform the prompt's final entity-count check immediately before "
    "returning it. Never "
    "return a proposition whose entities array is empty; if any entity in a "
    "proposition is not an exact substring of the supplied text, omit the "
    "entire proposition rather than deleting or repairing only that entity. "
    "Every admitted causal relation must have a strictly positive textual-"
    "support strength; never emit zero. The only permitted causal relation "
    "type values are causes, enables, and prevents. Omit any relation that "
    "cannot use one of those exact values; never invent another type string."
)


def extraction_prompt(text: str, *, schema_correction: bool = False) -> str:


    if not isinstance(schema_correction, bool):
        raise TypeError("schema_correction must be bool")
    correction = _EXTRACTION_SCHEMA_CORRECTION if schema_correction else ""
    return _EXTRACT_PROMPT.replace(
        "__SCHEMA_CORRECTION__", correction
    ).format(text=text)


def _extraction_request_hash(text: str, *, schema_correction: bool) -> str:


    return canonical_sha256(
        {
            "system": _EXTRACTION_SYSTEM,
            "prompt": extraction_prompt(text, schema_correction=schema_correction),
            "response_format": {"type": "json_object"},
            "response_format_version": EXTRACTION_RESPONSE_FORMAT_VERSION,
        }
    )


def _successful_trace(trace: Any) -> bool:
    if not isinstance(trace, Mapping):
        return False
    if trace.get("status") == "ok":
        attempts = trace.get("attempt_count")
        return (
            not isinstance(attempts, bool)
            and isinstance(attempts, int)
            and attempts >= 1
            and trace.get("stage") == "graph_extraction"
        )


    return (
        trace.get("status") == "success"
        and trace.get("provider_call_succeeded") is True
        and trace.get("stage", "graph_extraction") == "graph_extraction"
    )


def _grounded_exact(surface: str, source_text: Optional[str]) -> bool:


    if source_text is None:
        return True
    if not isinstance(surface, str) or not isinstance(source_text, str):
        return False
    surface_nfc = unicodedata.normalize("NFC", surface)
    source_nfc = unicodedata.normalize("NFC", source_text)
    return bool(surface_nfc.strip()) and surface_nfc in source_nfc


def _unwrap_extraction(value: Any) -> Tuple[Any, Optional[Mapping[str, Any]]]:
    if isinstance(value, Mapping) and set(value) == {"payload", "trace"}:
        trace = value.get("trace")
        return value.get("payload"), trace if isinstance(trace, Mapping) else None
    if isinstance(value, Mapping):
        trace = value.get("_trace")
        if "_trace" in value:



            return (
                {key: item for key, item in value.items() if key != "_trace"},
                trace if isinstance(trace, Mapping) else None,
            )
        return value, None
    return value, None


def _raw_response_from_trace(trace: Optional[Mapping[str, Any]]) -> Optional[str]:


    if not isinstance(trace, Mapping):
        return None
    attempts = trace.get("attempts")
    if not isinstance(attempts, list):
        return None
    for attempt in reversed(attempts):
        if isinstance(attempt, Mapping) and isinstance(attempt.get("raw_response"), str):
            return str(attempt["raw_response"])
    return None


def _normalise_extraction_payload(
    value: Any,
    *,
    trace: Optional[Mapping[str, Any]],
    require_trace: bool,
    reject_inadmissible_relations: bool,
    omit_ungrounded_propositions: bool = False,
    omit_schema_invalid_propositions: bool = False,
    schema_rejections: Optional[List[Dict[str, Any]]] = None,
    source_text: Optional[str] = None,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not isinstance(value, Mapping) or set(value) != {"propositions", "causal_relations"}:
        raise ExtractionSchemaError("extractor output must be exactly propositions/causal_relations")
    propositions, relations = value["propositions"], value["causal_relations"]
    if not isinstance(propositions, list) or not isinstance(relations, list):
        raise ExtractionSchemaError("propositions and causal_relations must be arrays")
    if require_trace and not _successful_trace(trace):
        raise ExtractionSchemaError("published extraction lacks a successful provider trace")
    if not propositions and not relations and not _successful_trace(trace):
        raise ExtractionSchemaError("explicit-empty extraction requires a successful provider trace")

    norm_props, proposition_keys = [], set()
    grounding_rejections: List[Dict[str, Any]] = []
    if schema_rejections is None:
        schema_rejections = []
    endpoint_by_raw: Dict[str, str] = {}
    endpoint_name_by_raw: Dict[str, str] = {}
    for prop_index, proposition in enumerate(propositions):
        if not isinstance(proposition, Mapping) or set(proposition) != {"sentence", "entities"}:
            raise ExtractionSchemaError(f"proposition {prop_index} has unknown/missing fields")
        sentence = proposition["sentence"]
        if not isinstance(sentence, str) or not sentence.strip():
            raise ExtractionSchemaError("proposition sentence must be non-empty")
        sentence = unicodedata.normalize("NFC", sentence.strip())
        if not isinstance(proposition["entities"], list):
            raise ExtractionSchemaError("proposition entities must be a non-empty array")
        if not proposition["entities"]:
            if omit_schema_invalid_propositions:
                schema_rejections.append(
                    {
                        "proposition_index": prop_index,
                        "reason_code": "proposition_entities_empty",
                        "proposition_sha256": _safe_audit_sha256(proposition),
                    }
                )
                continue
            raise ExtractionSchemaError("proposition entities must be a non-empty array")
        prepared_entities: List[Tuple[str, str, str]] = []
        unsafe_text_fields: List[str] = []
        if not _xml_char_allowed(sentence):
            unsafe_text_fields.append("sentence")
        for entity_index, entity in enumerate(proposition["entities"]):
            if not isinstance(entity, Mapping) or "name" not in entity or set(entity) - {"name", "name_zh"}:
                raise ExtractionSchemaError("entity has unknown/missing fields")
            raw_name = entity["name"]
            raw_name_zh = entity.get("name_zh", raw_name)
            if (
                not isinstance(raw_name, str)
                or not raw_name.strip()
                or not isinstance(raw_name_zh, str)
            ):
                raise ExtractionSchemaError("entity names must be non-empty strings")
            name = unicodedata.normalize("NFC", raw_name.strip())
            name_zh = unicodedata.normalize("NFC", raw_name_zh.strip()) or name
            if not _xml_char_allowed(name):
                unsafe_text_fields.append(f"entities[{entity_index}].name")
            if not _xml_char_allowed(name_zh):
                unsafe_text_fields.append(f"entities[{entity_index}].name_zh")
            prepared_entities.append((raw_name, name, name_zh))
        if unsafe_text_fields:
            if not omit_ungrounded_propositions:
                first = unsafe_text_fields[0]
                context = (
                    f"proposition {prop_index} sentence"
                    if first == "sentence"
                    else f"proposition {prop_index} {first.replace('entities[', 'entity ').replace('].', ' ')}"
                )
                raise ExtractionSchemaError(
                    f"{context} contains a character that cannot be represented in XML 1.0 GraphML"
                )
            grounding_rejections.append(
                {
                    "proposition_index": prop_index,
                    "reason_code": "proposition_contains_xml_unsafe_text",
                    "proposition_sha256": _safe_audit_sha256(proposition),
                    "unsafe_text_fields": sorted(set(unsafe_text_fields)),
                    "entity_count": len(proposition["entities"]),
                }
            )
            continue
        ungrounded_entity_indices: List[int] = []



        for entity_index, (raw_name, _name, _name_zh) in enumerate(prepared_entities):
            if source_text is not None and not _grounded_exact(raw_name, source_text):
                ungrounded_entity_indices.append(entity_index)

        if ungrounded_entity_indices:
            if not omit_ungrounded_propositions:
                first_index = ungrounded_entity_indices[0]
                raise ExtractionSchemaError(
                    f"proposition {prop_index} entity {first_index} is not verbatim "
                    "grounded in supplied chunk text"
                )
            grounding_rejections.append(
                {
                    "proposition_index": prop_index,
                    "reason_code": "proposition_contains_ungrounded_entity",
                    "proposition_sha256": _safe_audit_sha256(proposition),
                    "ungrounded_entity_indices": list(ungrounded_entity_indices),
                    "entity_count": len(proposition["entities"]),
                }
            )
            continue

        prop_keys: set[str] = set()
        duplicate_entity_indices: List[int] = []
        prepared_canonical_entities: List[Tuple[str, str, str, str]] = []
        for entity_index, (raw_name, name, name_zh) in enumerate(prepared_entities):
            entity_key = canonical_entity_key(name)
            if entity_key in prop_keys:
                duplicate_entity_indices.append(entity_index)
            else:
                prop_keys.add(entity_key)
            prepared_canonical_entities.append((raw_name, name, name_zh, entity_key))

        if duplicate_entity_indices:
            if omit_schema_invalid_propositions:
                schema_rejections.append(
                    {
                        "proposition_index": prop_index,
                        "reason_code": "proposition_duplicate_canonical_entity",
                        "proposition_sha256": _safe_audit_sha256(proposition),
                    }
                )
                continue
            raise ExtractionSchemaError(
                "duplicate canonical entity inside a proposition"
            )

        proposition_key = canonical_proposition_key(sentence)
        if proposition_key in proposition_keys:
            raise ExtractionSchemaError("duplicate proposition key in one chunk")

        norm_entities = []
        proposition_endpoints: List[Tuple[str, str, str]] = []
        for raw_name, name, name_zh, entity_key in prepared_canonical_entities:






            proposition_endpoints.append((raw_name, entity_key, name))
            norm_entities.append({"name": name, "name_zh": name_zh, "canonical_key": entity_key})



        proposition_keys.add(proposition_key)
        for raw_name, entity_key, name in proposition_endpoints:
            endpoint_by_raw[raw_name] = entity_key
            endpoint_name_by_raw[raw_name] = name
        norm_props.append(
            {
                "sentence": sentence,
                "proposition_key": proposition_key,
                "source_proposition_index": prop_index,
                "entities": norm_entities,
            }
        )

    norm_relations, relation_keys, rejections = [], set(), []
    for relation_index, relation in enumerate(relations):
        try:
            allowed = {"cause", "effect", "type", "strength", "description"}
            required = {"cause", "effect", "type", "strength"}
            if (
                not isinstance(relation, Mapping)
                or set(relation) - allowed
                or not required.issubset(relation)
            ):
                raise _RelationSchemaError(
                    f"causal relation {relation_index} has unknown/missing fields",
                    reason_code="invalid_relation_fields",
                )
            cause, effect, relation_type = (
                relation["cause"],
                relation["effect"],
                relation["type"],
            )
            if (
                not isinstance(cause, str)
                or not cause.strip()
                or not isinstance(effect, str)
                or not effect.strip()
            ):
                raise _RelationSchemaError(
                    "causal endpoints must be non-empty strings",
                    reason_code="invalid_endpoint_value",
                )
            raw_cause, raw_effect = cause, effect
            missing_roles = [
                role
                for role, endpoint in (("cause", raw_cause), ("effect", raw_effect))
                if endpoint not in endpoint_by_raw
            ]
            if missing_roles:
                raise _RelationSchemaError(
                    "causal endpoints must be verbatim entities from the response",
                    reason_code="endpoint_not_in_entity_vocabulary",
                    details={"missing_endpoint_roles": missing_roles},
                )
            if source_text is not None:
                ungrounded_roles = [
                    role
                    for role, endpoint in (("cause", raw_cause), ("effect", raw_effect))
                    if not _grounded_exact(endpoint, source_text)
                ]
                if ungrounded_roles:
                    raise _RelationSchemaError(
                        "causal endpoints must be grounded in the supplied chunk text",
                        reason_code="endpoint_not_grounded_in_chunk",
                        details={"missing_endpoint_roles": ungrounded_roles},
                    )
            cause_key, effect_key = endpoint_by_raw[raw_cause], endpoint_by_raw[raw_effect]
            cause, effect = endpoint_name_by_raw[raw_cause], endpoint_name_by_raw[raw_effect]
            if cause_key == effect_key:
                raise _RelationSchemaError(
                    "causal self-loops are not allowed",
                    reason_code="canonical_self_loop",
                )
            if not isinstance(relation_type, str) or relation_type not in ALLOWED_CAUSAL_TYPES:
                raise _RelationSchemaError(
                    f"unsupported causal type: {relation_type!r}",
                    reason_code="unsupported_causal_type",
                )
            strength = relation["strength"]
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise _RelationSchemaError(
                    "strength must be a real number, not bool/string",
                    reason_code="invalid_strength_type",
                )
            strength = float(strength)
            if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
                raise _RelationSchemaError(
                    "strength must be finite and strictly in (0,1]",
                    reason_code="invalid_strength_range",
                )
            description = relation.get("description", "")
            if not isinstance(description, str):
                raise _RelationSchemaError(
                    "description must be a string",
                    reason_code="invalid_description",
                )
            description = unicodedata.normalize("NFC", description.strip())
            if not _xml_char_allowed(description):
                raise _RelationSchemaError(
                    "description contains a character that cannot be represented in XML 1.0 GraphML",
                    reason_code="invalid_description",
                )
            relation_key = (cause_key, effect_key, relation_type)
            if relation_key in relation_keys:
                raise _RelationSchemaError(
                    "duplicate typed relation in one chunk",
                    reason_code="duplicate_typed_relation",
                )
            relation_keys.add(relation_key)
            norm_relations.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "cause_key": cause_key,
                    "effect_key": effect_key,
                    "type": relation_type,
                    "strength": strength,
                    "description": description,
                    "source_relation_index": relation_index,
                }
            )
        except _RelationSchemaError as exc:
            if not reject_inadmissible_relations:
                raise
            rejection = {
                "relation_index": relation_index,
                "reason_code": exc.reason_code,
                "relation_sha256": _safe_audit_sha256(relation),
            }
            rejection.update(exc.details)
            rejections.append(rejection)

    if rejections and not norm_props and not grounding_rejections and not schema_rejections:
        raise ExtractionSchemaError(
            "relation projection cannot turn a non-empty relation response into explicit empty extraction"
        )
    if (
        not norm_props
        and not norm_relations
        and not grounding_rejections
        and not schema_rejections
        and not _successful_trace(trace)
    ):
        raise ExtractionSchemaError("explicit-empty extraction requires a successful provider trace")
    return (
        {"propositions": norm_props, "causal_relations": norm_relations},
        rejections,
        grounding_rejections,
    )


def validate_extraction_payload(
    value: Any,
    *,
    trace: Optional[Mapping[str, Any]] = None,
    require_trace: bool = False,
    source_text: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:


    normalised, _, grounding_rejections = _normalise_extraction_payload(
        value,
        trace=trace,
        require_trace=require_trace,
        reject_inadmissible_relations=False,
        source_text=source_text,
    )
    if grounding_rejections:
        raise ExtractionSchemaError("grounding projection is not allowed in strict validation")
    return normalised


def _project_inadmissible_relations(
    value: Any,
    *,
    trace: Optional[Mapping[str, Any]],
    require_trace: bool,
    source_text: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:


    normalised, relation_rejections, _ = _normalise_extraction_payload(
        value,
        trace=trace,
        require_trace=require_trace,
        reject_inadmissible_relations=True,
        source_text=source_text,
    )
    return normalised, relation_rejections


def _project_final_extraction(
    value: Any,
    *,
    trace: Optional[Mapping[str, Any]],
    require_trace: bool,
    source_text: Optional[str] = None,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:


    schema_rejections: List[Dict[str, Any]] = []
    normalised, relation_rejections, grounding_rejections = _normalise_extraction_payload(
        value,
        trace=trace,
        require_trace=require_trace,
        reject_inadmissible_relations=True,
        omit_ungrounded_propositions=True,
        omit_schema_invalid_propositions=True,
        schema_rejections=schema_rejections,
        source_text=source_text,
    )
    return normalised, relation_rejections, grounding_rejections, schema_rejections


class LLMExtractor:


    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4o-mini",
        *,
        resolved_config: Any = None,
        workers: int = 256,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        del api_key
        self._base_url = base_url
        self._model = model
        self._resolved_config = resolved_config
        self._workers = max(1, int(workers))
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)

        configured_revision = _config_value(
            resolved_config, ("build_llm_revision",), None
        )
        self._model_revision = (
            str(configured_revision).strip()
            if isinstance(configured_revision, str) and configured_revision.strip()
            else None
        )

    def _extract_one(
        self, text: str, *, schema_correction: bool = False
    ) -> Dict[str, Any]:
        from chain import llm

        prompt = extraction_prompt(text, schema_correction=schema_correction)
        raw, transport_trace = llm.resolved_build_chat(
            prompt,
            system=_EXTRACTION_SYSTEM,
            config=self._resolved_config,
            stage="graph_extraction",
            model=self._model,
            model_revision=self._model_revision,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format="json_object",




            max_attempts=EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT,
        )
        if not isinstance(raw, str):
            raise GraphBuildError("shared LLM boundary returned non-text extraction content")
        try:
            payload = _strict_json_loads(raw, context="graph extraction provider response")
        except InputRecordError as exc:
            raise ExtractionAttemptError(
                f"provider response is not one direct JSON object: {exc}",
                trace=transport_trace,
                raw_response=raw,
            ) from exc
        if not isinstance(transport_trace, Mapping) or not _successful_trace(transport_trace):
            raise ExtractionAttemptError(
                "shared build LLM boundary returned an unverifiable success trace",
                trace=transport_trace,
                raw_response=raw,
            )
        scientific_trace = transport_trace.get("scientific")
        provider_attempts = transport_trace.get("attempts")
        if (
            transport_trace.get("response_format") != "json_object"
            or transport_trace.get("response_format_version")
            != EXTRACTION_RESPONSE_FORMAT_VERSION
            or not isinstance(scientific_trace, Mapping)
            or scientific_trace.get("response_format") != "json_object"
            or scientific_trace.get("response_format_version")
            != EXTRACTION_RESPONSE_FORMAT_VERSION
            or not isinstance(provider_attempts, list)
            or not provider_attempts
            or any(
                not isinstance(item, Mapping)
                or item.get("response_format") != "json_object"
                or item.get("response_format_version")
                != EXTRACTION_RESPONSE_FORMAT_VERSION
                for item in provider_attempts
            )
        ):
            raise ExtractionAttemptError(
                "shared build LLM boundary did not attest the sealed json_object "
                "response format",
                trace=transport_trace,
                raw_response=raw,
            )
        return {"payload": payload, "trace": dict(transport_trace)}

    def _extract_batch(
        self, texts: List[str], *, schema_correction: bool
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []
        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=min(self._workers, len(texts))) as pool:
            futures = {
                pool.submit(
                    self._extract_one,
                    text,
                    schema_correction=schema_correction,
                ): index
                for index, text in enumerate(texts)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = {
                        "_extract_error": str(exc),
                        "_extract_error_type": type(exc).__name__,
                        "_extract_trace": getattr(exc, "provider_trace", None)
                        or getattr(exc, "trace", None),
                        "_extract_attempts": getattr(exc, "attempts", None),
                        "_raw_response": getattr(exc, "raw_response", None),
                        "_response_sha256": getattr(exc, "response_sha256", None),
                    }
        if any(item is None for item in results):
            raise GraphBuildError("extractor worker returned an incomplete batch")
        return [item for item in results if item is not None]

    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:


        return self._extract_batch(texts, schema_correction=False)

    def extract_batch_with_feedback(
        self,
        texts: List[str],
        *,
        correction: str,
    ) -> List[Dict[str, Any]]:


        if correction != EXTRACTION_SCHEMA_CORRECTION_MARKER:
            raise ValueError("unsupported extraction correction marker")
        return self._extract_batch(texts, schema_correction=True)


def _endpoint_identity(base_url: str) -> Any:
    if not base_url:
        return {
            "scheme": "synthetic",
            "host": "local-fixture",
            "port": "",
            "path_sha256": _content_sha256(""),
        }
    from chain.config import endpoint_identity



    return endpoint_identity(base_url)


def _signature_from_values(
    *, provider: str, endpoint: Any, model: str, revision: str, dimension: int,
    preprocess_version: str = "chain-canonical-entity-key-v1",
) -> Dict[str, Any]:
    return {
        "provider": provider,
        "endpoint_identity": endpoint,
        "model": model,
        "revision": revision,
        "dimension": dimension,
        "dtype": "float32",
        "serialization_precision": "float32-le",
        "preprocess_version": preprocess_version,
        "normalization": "l2",
        "distance": "cosine",
    }


class Embedder:


    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        resolved_config: Any = None,
        provider: str = "openai-compatible",
        revision: str = "unspecified",
        batch_size: int = 100,
        workers: int = 256,
    ):
        del api_key
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("embedding dimension must be a positive integer")
        self._base_url = base_url
        self._model = model
        self._dim = dim
        self._resolved_config = resolved_config
        self._batch_size = max(1, int(batch_size))
        self._workers = max(1, int(workers))
        self.signature = _signature_from_values(
            provider=provider,
            endpoint=_endpoint_identity(base_url),
            model=model,
            revision=revision,
            dimension=dim,
        )

    def _embed_batch(
        self,
        texts: List[str],
        *,
        trace_sink: Optional[List[Dict[str, Any]]] = None,
        trace_lock: Optional[threading.Lock] = None,
        trace_role: str = "embedding",
    ) -> List[List[float]]:
        from chain import llm

        vectors, trace = llm.embed_texts_strict(
            texts,
            config=self._resolved_config,
            model=self._model,
            model_revision=str(self.signature["revision"]),
            expected_dim=self._dim,
            stage="graph_embedding",
        )
        if (
            not isinstance(trace, Mapping)
            or trace.get("status") != "ok"
            or not isinstance(trace.get("attempt_count"), int)
            or trace.get("attempt_count", 0) < 1
        ):
            raise GraphBuildError("shared embedding boundary returned an unverifiable success trace")
        if trace_sink is not None:
            if trace_lock is None:
                trace_sink.append({"role": str(trace_role), "trace": dict(trace)})
            else:
                with trace_lock:
                    trace_sink.append({"role": str(trace_role), "trace": dict(trace)})
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise GraphBuildError("embedding provider returned the wrong number of vectors")
        result = []
        for vector in vectors:
            if not isinstance(vector, (list, tuple)) or len(vector) != self._dim:
                raise GraphBuildError("embedding vector dimension mismatch")
            parsed = [float(item) for item in vector]
            if any(not math.isfinite(item) for item in parsed):
                raise GraphBuildError("embedding provider returned a non-finite vector")
            if math.sqrt(sum(item * item for item in parsed)) == 0.0:
                raise GraphBuildError("embedding provider returned a zero vector")
            result.append(parsed)
        return result

    def embed(
        self,
        texts: List[str],
        *,
        trace_sink: Optional[List[Dict[str, Any]]] = None,
        trace_lock: Optional[threading.Lock] = None,
        trace_role: str = "embedding",
    ) -> List[List[float]]:
        if not texts:
            return []
        batches = [texts[index : index + self._batch_size] for index in range(0, len(texts), self._batch_size)]
        outputs: List[Optional[List[List[float]]]] = [None] * len(batches)
        completed_batches = 0
        completed_texts = 0
        with ThreadPoolExecutor(max_workers=min(self._workers, len(batches))) as pool:
            futures = {
                pool.submit(
                    self._embed_batch,
                    batch,
                    trace_sink=trace_sink,
                    trace_lock=trace_lock,
                    trace_role=trace_role,
                ): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index = futures[future]
                outputs[index] = future.result()
                completed_batches += 1
                completed_texts += len(batches[index])
                logger.info(
                    "Embedding %s: batch %d/%d complete; embedded %d/%d texts (%.1f%%)",
                    trace_role,
                    completed_batches,
                    len(batches),
                    completed_texts,
                    len(texts),
                    100.0 * completed_texts / len(texts),
                )
        result: List[List[float]] = []
        for output in outputs:
            if output is None:
                raise GraphBuildError("embedding worker returned an incomplete batch")
            result.extend(output)
        return result

    def embed_one(self, text: str) -> List[float]:
        value = self.embed([text])
        if len(value) != 1:
            raise GraphBuildError("embedding provider did not return one query vector")
        return value[0]


class JsonKVStorage:
    def __init__(self, path: str, *, create: bool = False):
        self._path = Path(path)
        self._data: Dict[str, Any] = {}
        if self._path.exists():
            try:
                value = _strict_artifact_json_loads(
                    self._path.read_text(encoding="utf-8"), context=str(self._path)
                )
            except ArtifactCorruptionError as exc:
                raise ArtifactCorruptionError(f"cannot parse {self._path}: {exc}") from exc
            if isinstance(value, Mapping) and value.get("schema_version") == CHUNKS_SCHEMA:
                value = value.get("chunks")
            if not isinstance(value, Mapping):
                raise ArtifactCorruptionError(f"{self._path} is not an object KV store")
            self._data = dict(value)
        elif not create:
            raise FileNotFoundError(f"KV storage does not exist: {self._path}")

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def upsert(self, data: Dict[str, Any]) -> None:
        self._data.update(data)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def filter_new(self, keys: Iterable[str]) -> List[str]:
        return [key for key in keys if key not in self._data]

    def save(self) -> None:
        write_canonical_json(
            self._path,
            {"schema_version": CHUNKS_SCHEMA, "chunks": dict(sorted(self._data.items()))},
        )


class VectorStorage:
    def __init__(
        self,
        path: str,
        dim: int,
        *,
        create: bool = False,
        signature: Optional[Mapping[str, Any]] = None,
        role: str = "unknown",
    ):
        try:
            import numpy as np
        except Exception as exc:
            raise GraphBuildError(f"numpy is required for vector storage: {exc}") from exc
        self._np = np
        self._path = Path(path)
        self._dim = int(dim)
        self._role = role
        self._signature = dict(signature or {})
        self._records: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, Any] = {}
        if self._path.exists():
            try:
                value = _strict_artifact_json_loads(
                    self._path.read_text(encoding="utf-8"), context=str(self._path)
                )
                if not isinstance(value, Mapping):
                    raise ArtifactCorruptionError("vector storage root must be an object")
                stored_dim, data = value["embedding_dim"], value["data"]
                if isinstance(stored_dim, bool) or not isinstance(stored_dim, int) or stored_dim <= 0:
                    raise ArtifactCorruptionError("vector storage has invalid embedding_dim")
                if not isinstance(data, list):
                    raise ArtifactCorruptionError("vector storage data must be an array")
                matrix = np.frombuffer(
                    base64.b64decode(value["matrix"], validate=True), dtype=np.dtype("<f4")
                ).reshape((len(data), stored_dim))
                additional = value["additional_data"]
            except ArtifactCorruptionError:
                raise
            except Exception as exc:
                raise ArtifactCorruptionError(f"cannot parse vector storage {self._path}: {exc}") from exc
            if stored_dim != self._dim:
                raise ArtifactCorruptionError("vector storage dimension mismatch")
            if not isinstance(additional, Mapping) or additional.get("schema_version") != VECTOR_SCHEMA:
                raise ArtifactCorruptionError("vector storage lacks canonical metadata")
            self._signature = dict(additional.get("embedding_signature", {}))
            self._role = str(additional.get("role", role))
            node_ids: Set[str] = set()
            if not np.isfinite(matrix).all():
                raise ArtifactCorruptionError("vector storage contains non-finite vectors")
            if len(data) and (np.linalg.norm(matrix, axis=1) == 0).any():
                raise ArtifactCorruptionError("vector storage contains a zero vector")
            for index, record in enumerate(data):
                if not isinstance(record, Mapping):
                    raise ArtifactCorruptionError("vector storage record is not an object")
                vector_id = record.get("__id__")
                if not isinstance(vector_id, str) or vector_id in self._records:
                    raise ArtifactCorruptionError("invalid/duplicate vector ID")
                node_id = record.get("node_id")
                if not isinstance(node_id, str) or not node_id:
                    raise ArtifactCorruptionError("vector storage record lacks node_id")
                if node_id in node_ids:
                    raise ArtifactCorruptionError("vector storage contains duplicate node_id")
                node_ids.add(node_id)
                self._records[vector_id] = dict(record)
                vector = matrix[index].copy()



                norm = float(np.linalg.norm(vector.astype(np.float64)))
                if not math.isfinite(norm) or norm <= 0.0:
                    raise ArtifactCorruptionError(
                        "vector storage contains a vector with invalid norm"
                    )
                normalised = vector / norm
                if not np.isfinite(normalised).all():
                    raise ArtifactCorruptionError(
                        "vector storage contains an invalid normalised vector"
                    )
                self._vectors[vector_id] = normalised
        elif not create:
            raise FileNotFoundError(f"vector storage does not exist: {self._path}")

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(self._signature)

    def upsert(self, vid: str, vector: List[float], meta: Dict[str, Any]) -> None:
        self.upsert_batch([{"id": vid, "vector": vector, "meta": meta}])

    def upsert_batch(self, batch: List[Dict[str, Any]]) -> None:
        np = self._np
        if not isinstance(batch, list):
            raise GraphBuildError("vector upsert batch must be a list")
        pending_node_ids: Dict[str, str] = {}
        for entry in batch:
            if not isinstance(entry, Mapping):
                raise GraphBuildError("vector upsert entry must be an object")
            vector_id = entry.get("id")
            if not isinstance(vector_id, str) or not vector_id:
                raise GraphBuildError("vector IDs must be non-empty strings")
            vector = np.asarray(entry.get("vector"), dtype=np.dtype("<f4"))
            if vector.shape != (self._dim,) or not np.isfinite(vector).all():
                raise GraphBuildError(f"invalid vector for {vector_id}")
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if not math.isfinite(norm) or norm <= 0.0:
                raise GraphBuildError(f"zero vector for {vector_id}")
            meta = entry.get("meta", {})
            if not isinstance(meta, Mapping):
                raise GraphBuildError(f"vector metadata for {vector_id} must be an object")
            reserved = _RESERVED_VECTOR_META_KEYS.intersection(meta)
            if reserved:
                raise GraphBuildError(
                    f"vector metadata for {vector_id} contains reserved fields: {sorted(reserved)}"
                )
            node_id = meta.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise GraphBuildError(f"vector metadata for {vector_id} requires node_id")
            prior_id = pending_node_ids.get(node_id)
            if prior_id is not None and prior_id != vector_id:
                raise GraphBuildError(f"node_id {node_id!r} is assigned to multiple vectors in one batch")
            pending_node_ids[node_id] = vector_id
            existing_id = next(
                (candidate_id for candidate_id, record in self._records.items() if record.get("node_id") == node_id),
                None,
            )
            if existing_id is not None and existing_id != vector_id:
                raise GraphBuildError(f"node_id {node_id!r} is already assigned to vector {existing_id!r}")
            normalised = vector / norm
            if not np.isfinite(normalised).all():
                raise GraphBuildError(f"invalid normalised vector for {vector_id}")
            if existing_id == vector_id and not np.array_equal(self._vectors[vector_id], normalised):
                raise GraphBuildError(f"vector identity for node_id {node_id!r} changed during upsert")
            self._records[vector_id] = {"__id__": vector_id, **dict(meta)}
            self._vectors[vector_id] = normalised

    def search(
        self,
        query_vec: List[float],
        top_k: int = 10,
        *,
        allowed_node_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        np = self._np
        if top_k <= 0:
            return []



        query = np.asarray(query_vec, dtype=np.float64)
        if query.shape != (self._dim,) or not np.isfinite(query).all():
            raise GraphBuildError("query vector has incompatible dimension/non-finite values")
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm <= 0.0:
            raise GraphBuildError("query vector is zero")
        query = query / norm
        scored = []
        for vector_id in sorted(self._records):
            record = self._records[vector_id]
            if allowed_node_ids is not None and record.get("node_id") not in allowed_node_ids:
                continue
            raw_score = float(
                np.dot(
                    np.asarray(self._vectors[vector_id], dtype=np.float64),
                    query,
                )
            )
            if not math.isfinite(raw_score):
                raise GraphBuildError("entity similarity is non-finite")
            if (
                raw_score < -1.0 - _COSINE_BOUND_TOLERANCE
                or raw_score > 1.0 + _COSINE_BOUND_TOLERANCE
            ):
                raise GraphBuildError(
                    "entity similarity is materially outside [-1,1]"
                )
            score = min(1.0, max(-1.0, raw_score))
            scored.append((score, vector_id, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**record, "__metrics__": score} for score, _, record in scored[:top_k]]

    def records(self) -> List[Dict[str, Any]]:
        return [dict(self._records[key]) for key in sorted(self._records)]

    def save(self) -> None:
        np = self._np
        ordered = sorted(self._records)
        matrix = (
            np.vstack([self._vectors[key] for key in ordered]).astype(np.dtype("<f4"))
            if ordered
            else np.empty((0, self._dim), dtype=np.dtype("<f4"))
        )
        write_canonical_json(
            self._path,
            {
                "embedding_dim": self._dim,
                "data": [self._records[key] for key in ordered],
                "matrix": base64.b64encode(matrix.tobytes(order="C")).decode("ascii"),
                "additional_data": {
                    "schema_version": VECTOR_SCHEMA,
                    "role": self._role,
                    "embedding_signature": self._signature,
                },
            },
        )


def _xml_safe(value: str) -> str:
    if not _xml_char_allowed(value):
        raise GraphBuildError(
            "GraphML text contains a character that is not legal in XML 1.0"
        )
    return value


def _graphml_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return "json:" + canonical_json(value)
    if isinstance(value, str):
        return _xml_safe(value)
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphBuildError("GraphML cannot store non-finite values")
        return value
    return "json:" + canonical_json(value)


def _decode_graphml_json_values(graph: Any) -> Any:


    def decode(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("json:"):
            parsed = _strict_artifact_json_loads(
                value[5:], context="tagged GraphML JSON attribute"
            )
            if "json:" + canonical_json(parsed) != value:
                raise ArtifactCorruptionError("non-canonical tagged GraphML JSON attribute")
            return parsed
        return value

    graph.graph.update({key: decode(value) for key, value in graph.graph.items()})
    for _, attrs in graph.nodes(data=True):
        attrs.update({key: decode(value) for key, value in attrs.items()})
    for _, _, _, attrs in graph.edges(keys=True, data=True):
        attrs.update({key: decode(value) for key, value in attrs.items()})
    return graph


class GraphStorage:
    def __init__(self, path: str, *, create: bool = False, graph: Any = None):
        import networkx as nx

        self._path = Path(path)
        if graph is not None:
            self._graph = nx.MultiDiGraph(graph)
        elif self._path.exists():
            try:
                self._graph = nx.MultiDiGraph(nx.read_graphml(str(self._path), force_multigraph=True))
            except Exception as exc:
                raise ArtifactCorruptionError(f"cannot parse GraphML {self._path}: {exc}") from exc
        elif create:
            self._graph = nx.MultiDiGraph()
        else:
            raise FileNotFoundError(f"graph storage does not exist: {self._path}")

    def upsert_node(self, node_id: str, attrs: Dict[str, Any]) -> None:
        if node_id in self._graph:
            self._graph.nodes[node_id].update(attrs)
        else:
            self._graph.add_node(node_id, **attrs)

    def upsert_edge(
        self, src: str, dst: str, attrs: Optional[Dict[str, Any]] = None, *, key: Optional[str] = None
    ) -> str:
        values = dict(attrs or {})
        reserved = _RESERVED_EDGE_ATTR_KEYS.intersection(values)
        if reserved:
            raise GraphBuildError(f"edge attributes contain reserved fields: {sorted(reserved)}")
        if key is None:
            key = _typed_sha256(
                "edge_",
                {"source": src, "target": dst, "attrs": values, "ordinal": self._graph.number_of_edges(src, dst)},
            )
        values["edge_id"] = key
        self._graph.add_edge(src, dst, key=key, **values)
        return key

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return dict(self._graph.nodes[node_id]) if node_id in self._graph else None

    def neighbors(self, node_id: str) -> List[str]:
        return sorted(self._graph.successors(node_id)) if node_id in self._graph else []

    def predecessors(self, node_id: str) -> List[str]:
        return sorted(self._graph.predecessors(node_id)) if node_id in self._graph else []

    def nodes_by_role(self, role: str) -> List[Tuple[str, Dict]]:
        return sorted(
            ((str(node), dict(attrs)) for node, attrs in self._graph.nodes(data=True) if attrs.get("role") == role),
            key=lambda item: item[0],
        )

    def all_nodes(self) -> List[Tuple[str, Dict]]:
        return sorted(((str(node), dict(attrs)) for node, attrs in self._graph.nodes(data=True)), key=lambda item: item[0])

    def all_edges(self) -> List[Tuple[str, str]]:
        return sorted((str(source), str(target)) for source, target in self._graph.edges())

    def all_keyed_edges(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        return sorted(
            ((str(s), str(t), str(k), dict(a)) for s, t, k, a in self._graph.edges(keys=True, data=True)),
            key=lambda item: (item[0], item[1], item[2]),
        )

    def causal_successors(self, node_id: str) -> List[Tuple[str, Dict]]:
        result = []
        if node_id in self._graph:
            for _, target, key, attrs in self._graph.out_edges(node_id, keys=True, data=True):
                if attrs.get("role") == "causal":
                    result.append((str(target), {**dict(attrs), "key": str(key)}))
        result.sort(
            key=lambda item: (
                -float(item[1].get("strength", 0.0)),
                item[0],
                str(item[1].get("causal_type", "")),
                str(item[1].get("key", "")),
            )
        )
        return result

    def causal_predecessors(self, node_id: str) -> List[Tuple[str, Dict]]:
        result = []
        if node_id in self._graph:
            for source, _, key, attrs in self._graph.in_edges(node_id, keys=True, data=True):
                if attrs.get("role") == "causal":
                    result.append((str(source), {**dict(attrs), "key": str(key)}))
        result.sort(key=lambda item: (item[0], str(item[1].get("causal_type", "")), str(item[1].get("key", ""))))
        return result

    def causal_edge_count(self) -> int:
        return sum(1 for _, _, _, attrs in self._graph.edges(keys=True, data=True) if attrs.get("role") == "causal")

    def save(self) -> None:
        import networkx as nx

        canonical_graph = nx.MultiDiGraph()
        for key, value in sorted(self._graph.graph.items(), key=lambda item: str(item[0])):
            canonical_graph.graph[str(key)] = _graphml_scalar(value)
        for node_id, attrs in sorted(self._graph.nodes(data=True), key=lambda item: str(item[0])):
            canonical_graph.add_node(
                _xml_safe(str(node_id)),
                **{str(key): _graphml_scalar(value) for key, value in sorted(attrs.items())},
            )
        for source, target, key, attrs in self.all_keyed_edges():
            if attrs.get("edge_id") != key:
                raise GraphBuildError(f"edge {key!r} has inconsistent edge_id metadata")
            reserved = {"target", "key"}.intersection(attrs)
            if reserved:
                raise GraphBuildError(
                    f"edge {key!r} contains reserved identity attributes: {sorted(reserved)}"
                )
            values = {
                str(name): _graphml_scalar(value)
                for name, value in sorted(attrs.items())
                if name != "edge_id"
            }
            values["edge_id"] = key
            canonical_graph.add_edge(_xml_safe(source), _xml_safe(target), key=key, **values)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(canonical_graph, str(self._path), encoding="utf-8", prettyprint=False)
        with self._path.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            descriptor = os.open(self._path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)


def entity_node_id(graph_namespace: str, entity_key: str) -> str:
    return _typed_sha256("ent_", {"graph_namespace": graph_namespace, "canonical_entity_key": entity_key})


def hyperedge_node_id(graph_namespace: str, chunk_id: str, proposition_index: int, proposition_key: str) -> str:
    return _typed_sha256(
        "hyp_",
        {
            "graph_namespace": graph_namespace,
            "chunk_id": chunk_id,
            "proposition_index": proposition_index,
            "proposition_key": proposition_key,
        },
    )


def incidence_edge_id(graph_namespace: str, hyperedge_id: str, entity_id: str, role: str = "mentions") -> str:
    return _typed_sha256(
        "inc_",
        {"graph_namespace": graph_namespace, "hyperedge_id": hyperedge_id, "entity_id": entity_id, "incidence_role": role},
    )


def causal_edge_id(
    graph_namespace: str,
    chunk_id: str,
    relation_index: int,
    cause_entity_id: str,
    effect_entity_id: str,
    causal_type: str,
) -> str:
    return _typed_sha256(
        "cedge_",
        {
            "graph_namespace": graph_namespace,
            "chunk_id": chunk_id,
            "relation_index": relation_index,
            "cause_entity_id": cause_entity_id,
            "effect_entity_id": effect_entity_id,
            "causal_type": causal_type,
        },
    )


def _config_value(config: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            return value() if callable(value) and name.endswith("sha256") else value
    return default


def _profile_id(config: Any, profile: Optional[str]) -> str:
    value = _config_value(config, ("profile_id",), None)
    if value:
        if value not in {COMPAT_PROFILE_ID, PAPER_PROFILE_ID}:
            raise GraphBuildError(f"unknown profile identity: {value!r}")
        return str(value)
    profile_name = profile or str(_config_value(config, ("profile",), "compat"))
    if profile_name in {"compat", COMPAT_PROFILE_ID}:
        return COMPAT_PROFILE_ID
    if profile_name in {"paper", PAPER_PROFILE_ID}:
        return PAPER_PROFILE_ID
    raise GraphBuildError(f"unknown profile: {profile_name!r}")


def _object_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    for method in ("to_dict", "as_dict", "model_dump"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            result = candidate()
            if isinstance(result, Mapping):
                return dict(result)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return {}


def _scrub_config(value: Any) -> Any:
    secret_tokens = ("api_key", "token", "secret", "password", "credential")
    path_names = {"graph_dir", "scratch_dir", "output", "results_dir", "event_dir", "knowledge_dir"}
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_config(item)
            for key, item in value.items()
            if not any(token in str(key).casefold() for token in secret_tokens)
            and str(key).casefold() not in path_names
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_config(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _embedding_signature(config: Any, embedder: Any) -> Dict[str, Any]:
    signature = getattr(embedder, "signature", None)
    if callable(signature):
        signature = signature()
    if not isinstance(signature, Mapping):
        signature = _config_value(config, ("embedding_signature",), None)
        if callable(signature):
            signature = signature()
    if isinstance(signature, Mapping):
        result = dict(signature)
    else:
        model = str(_config_value(config, ("embed_model", "embedding_model", "EMBED_MODEL"), "text-embedding-3-small"))
        dimension = int(_config_value(config, ("embed_dim", "embedding_dimension", "EMBED_DIM"), 1536))
        base_url = str(_config_value(config, ("embed_base_url", "embedding_base_url", "EMBED_BASE_URL"), ""))
        result = _signature_from_values(
            provider=str(_config_value(config, ("api_provider", "embedding_provider"), "openai-compatible")),
            endpoint=_endpoint_identity(base_url),
            model=model,
            revision=str(_config_value(config, ("embed_revision", "embedding_revision"), "unspecified")),
            dimension=dimension,
            preprocess_version=str(
                _config_value(config, ("embedding_preprocess_version",), "chain-canonical-entity-key-v1")
            ),
        )
    required = {
        "provider",
        "endpoint_identity",
        "model",
        "revision",
        "dimension",
        "dtype",
        "serialization_precision",
        "preprocess_version",
        "normalization",
        "distance",
    }
    if set(result) != required:
        raise GraphBuildError(f"embedding signature field mismatch: {sorted(set(result) ^ required)}")
    if result["dtype"] != "float32" or result["serialization_precision"] != "float32-le":
        raise GraphBuildError("embedding identity must use float32/float32-le")
    if result["normalization"] != "l2" or result["distance"] != "cosine":
        raise GraphBuildError("embedding identity must use l2/cosine")
    if isinstance(result["dimension"], bool) or not isinstance(result["dimension"], int) or result["dimension"] <= 0:
        raise GraphBuildError("embedding dimension must be a positive integer")
    configured = _config_value(config, ("embedding_signature",), None)
    if callable(configured):
        configured = configured()
    if isinstance(configured, Mapping):
        configured_space = tuple(
            configured.get(field) for field in EMBEDDING_SPACE_FIELDS
        )
        observed_space = tuple(result.get(field) for field in EMBEDDING_SPACE_FIELDS)
        if configured_space != observed_space:
            raise GraphBuildError(
                "embedder vector space differs from resolved configuration"
            )
    return result


def _construction_config(config: Any, profile_id: str, signature: Mapping[str, Any]) -> Dict[str, Any]:
    tokenizer_model = str(_config_value(config, ("tokenizer_model",), "gpt-4o"))
    _, resolved_encoding, tokenizer_version = _get_encoder(tokenizer_model)
    max_tokens = int(_config_value(config, ("chunk_max_tokens",), 512))
    overlap = int(_config_value(config, ("chunk_overlap_tokens",), 64))
    retries = int(_config_value(config, ("extractor_schema_retries", "extraction_schema_retries"), 2))
    batch_size = int(_config_value(config, ("extraction_batch_size",), 5))
    resolved_transport_attempts = _config_value(config, ("llm_retry_attempts",), 5)
    if max_tokens <= 0 or not 0 <= overlap < max_tokens or not 0 <= retries <= 2 or batch_size <= 0:
        raise GraphBuildError("invalid chunk/extraction construction configuration")
    if (
        isinstance(resolved_transport_attempts, bool)
        or not isinstance(resolved_transport_attempts, int)
        or resolved_transport_attempts
        < EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT
    ):
        raise GraphBuildError(
            "resolved llm_retry_attempts must be at least "
            f"{EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT} to satisfy the "
            "sealed extraction transport retry policy"
        )
    configured_prompt_version = str(
        _config_value(config, ("extraction_prompt_version",), PROMPT_VERSION)
    ).strip()
    if configured_prompt_version != PROMPT_VERSION:
        raise GraphBuildError(
            "resolved extraction_prompt_version does not match the active extraction prompt"
        )
    configured_schema_version = str(
        _config_value(config, ("extraction_schema_version",), EXTRACTION_SCHEMA_VERSION)
    ).strip()
    if configured_schema_version != EXTRACTION_SCHEMA_VERSION:
        raise GraphBuildError(
            "resolved extraction_schema_version does not match the active extraction schema"
        )
    configured_relation_policy = str(
        _config_value(
            config,
            ("relation_admissibility_policy_version",),
            RELATION_ADMISSIBILITY_POLICY_VERSION,
        )
    ).strip()
    if configured_relation_policy != RELATION_ADMISSIBILITY_POLICY_VERSION:
        raise GraphBuildError(
            "resolved relation_admissibility_policy_version does not match the active policy"
        )
    configured_grounding_policy = str(
        _config_value(
            config,
            ("grounding_policy_version",),
            GROUNDING_POLICY_VERSION,
        )
    ).strip()
    if configured_grounding_policy != GROUNDING_POLICY_VERSION:
        raise GraphBuildError(
            "resolved grounding_policy_version does not match the active policy"
        )
    configured_proposition_schema_policy = str(
        _config_value(
            config,
            ("proposition_schema_projection_policy_version",),
            PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION,
        )
    ).strip()
    if (
        configured_proposition_schema_policy
        != PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
    ):
        raise GraphBuildError(
            "resolved proposition_schema_projection_policy_version does not match the active policy"
        )
    configured_response_format_version = str(
        _config_value(
            config,
            ("extraction_response_format_version",),
            EXTRACTION_RESPONSE_FORMAT_VERSION,
        )
    ).strip()
    if configured_response_format_version != EXTRACTION_RESPONSE_FORMAT_VERSION:
        raise GraphBuildError(
            "resolved extraction_response_format_version does not match the active "
            "structured-output protocol"
        )
    return {
        "schema_version": CONSTRUCTION_CONFIG_SCHEMA,
        "profile_id": profile_id,
        "chunk_layout_version": CHUNK_LAYOUT_VERSION,
        "content_normalization_version": CONTENT_NORMALIZATION_VERSION,
        "key_normalization_version": KEY_NORMALIZATION_VERSION,
        "compat_adapter_version": "chain-compat-record-adapters-v1",
        "chunk_max_tokens": max_tokens,
        "chunk_overlap_tokens": overlap,
        "requested_tokenizer_model": tokenizer_model,
        "resolved_tokenizer_encoding": resolved_encoding,
        "tiktoken_version": tokenizer_version,
        "extraction_batch_size": batch_size,
        "extraction_schema_retries": retries,
        "extraction_transport_attempts_per_schema_attempt": (
            EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT
        ),
        "extraction_temperature": float(_config_value(config, ("extractor_temperature",), 0.0)),
        "extraction_max_output_tokens": int(_config_value(config, ("extractor_max_tokens",), 4096)),
        "extractor_provider": str(_config_value(config, ("api_provider", "extractor_provider"), "openai-compatible")),
        "extractor_model": str(_config_value(config, ("build_llm_model", "extractor_model"), "gpt-4o-mini")),
        "extraction_prompt_version": PROMPT_VERSION,


        "extraction_prompt_sha256": _content_sha256(
            _EXTRACTION_SYSTEM
            + "\0"
            + _EXTRACT_PROMPT
            + "\0"
            + _EXTRACTION_SCHEMA_CORRECTION
        ),
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        "relation_admissibility_policy_version": RELATION_ADMISSIBILITY_POLICY_VERSION,
        "grounding_policy_version": GROUNDING_POLICY_VERSION,
        "proposition_schema_projection_policy_version": (
            PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
        ),
        "extraction_response_format_version": EXTRACTION_RESPONSE_FORMAT_VERSION,
        "extraction_response_format": {"type": "json_object"},
        "embedding_signature": dict(signature),
        "timestamp_policy": "precision-preserving-no-current-time-v1",
        "availability_policy": "audited-upper-bound-only-v1",
        "compat_retrieval_version": COMPAT_RETRIEVAL_VERSION,
        "compat_rrf_k": COMPAT_RRF_K,
        "compat_rrf_default_top_k": COMPAT_RRF_DEFAULT_TOP_K,
        "compat_rrf_channel_candidate_multiplier": COMPAT_RRF_CHANNEL_CANDIDATE_MULTIPLIER,
    }


def _scientific_config_hash(config: Any, construction: Mapping[str, Any]) -> str:
    value = _config_value(config, ("scientific_config_sha256",), None)
    if value:
        value = str(value)
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise GraphBuildError("scientific_config_sha256 must be full lower-case SHA-256")
        return value
    payload = _scrub_config(_object_dict(config)) or {"construction": construction}
    return canonical_sha256(payload)


def _construction_config_hash(config: Any, construction: Mapping[str, Any]) -> str:


    value = _config_value(config, ("construction_config_sha256",), None)
    if value:
        value = str(value)
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise GraphBuildError(
                "construction_config_sha256 must be full lower-case SHA-256"
            )
        return value
    payload = _config_value(config, ("construction_payload",), None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, Mapping):
        return canonical_sha256(payload)


    return canonical_sha256(construction)


def _default_extractor(config: Any, construction: Mapping[str, Any]) -> LLMExtractor:
    return LLMExtractor(
        base_url=str(_config_value(config, ("build_llm_base_url",), "")),
        model=str(construction["extractor_model"]),
        resolved_config=config,
        workers=int(_config_value(config, ("extraction_workers",), 256)),
        temperature=float(construction["extraction_temperature"]),
        max_tokens=int(construction["extraction_max_output_tokens"]),
    )


def _default_embedder(config: Any, signature: Mapping[str, Any]) -> Embedder:
    return Embedder(
        base_url=str(_config_value(config, ("embed_base_url",), "")),
        model=str(signature["model"]),
        dim=int(signature["dimension"]),
        resolved_config=config,
        provider=str(
            signature.get("provider")
            or _config_value(
                config, ("api_provider", "embedding_provider"), "openai-compatible"
            )
        ),
        revision=str(signature["revision"]),
        batch_size=int(_config_value(config, ("embedding_batch_size",), 100)),
        workers=int(_config_value(config, ("embedding_workers",), 256)),
    )


def _invoke_extractor(
    extractor: Any, texts: List[str], *, schema_correction: bool = False
) -> List[Any]:
    feedback_method = getattr(extractor, "extract_batch_with_feedback", None)
    if schema_correction and callable(feedback_method):


        value = feedback_method(
            texts, correction=EXTRACTION_SCHEMA_CORRECTION_MARKER
        )
    elif hasattr(extractor, "extract_batch"):
        value = extractor.extract_batch(texts)
    elif hasattr(extractor, "extract"):
        value = [extractor.extract(text) for text in texts]
    elif callable(extractor):
        value = [extractor(text) for text in texts]
    else:
        raise GraphBuildError("extractor must define extract_batch/extract or be callable")
    if not isinstance(value, list) or len(value) != len(texts):
        raise GraphBuildError("extractor returned the wrong number of results")
    return value


def extract_chunks(
    chunks: Sequence[Mapping[str, Any]],
    extractor: Any,
    *,
    batch_size: int,
    schema_retries: int,
    require_trace: bool,
) -> Tuple[List[Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    if (
        isinstance(schema_retries, bool)
        or not isinstance(schema_retries, int)
        or not 0 <= schema_retries <= 2
    ):
        raise GraphBuildError("schema_retries must be an integer in [0,2]")
    results: List[Optional[Dict[str, List[Dict[str, Any]]]]] = [None] * len(chunks)
    traces = [{"chunk_id": chunk["chunk_id"], "attempts": [], "status": "pending"} for chunk in chunks]
    pending = list(range(len(chunks)))
    accepted_total = 0
    logger.info(
        "Extraction started: %d chunks, batch size %d, at most %d schema attempt(s) "
        "per chunk; up to %d transport attempt(s) per schema attempt",
        len(chunks),
        batch_size,
        schema_retries + 1,
        EXTRACTION_TRANSPORT_ATTEMPTS_PER_SCHEMA_ATTEMPT,
    )
    for attempt in range(schema_retries + 1):
        schema_correction = attempt > 0
        next_pending: List[int] = []
        attempt_size = len(pending)
        attempt_batches = math.ceil(attempt_size / batch_size) if attempt_size else 0
        for start in range(0, len(pending), batch_size):
            indices = pending[start : start + batch_size]
            batch_number = start // batch_size + 1
            texts = [str(chunks[index]["content"]) for index in indices]
            uses_schema_feedback = schema_correction and callable(
                getattr(extractor, "extract_batch_with_feedback", None)
            )
            try:
                raw_values = _invoke_extractor(
                    extractor, texts, schema_correction=schema_correction
                )
            except Exception as exc:
                if attempt >= schema_retries:
                    raise GraphBuildError(f"extractor transport/worker failure after retries: {exc}") from exc
                for index in indices:
                    traces[index]["attempts"].append(
                        {
                            "attempt": attempt + 1,
                            "response_format": {"type": "json_object"},
                            "response_format_version": EXTRACTION_RESPONSE_FORMAT_VERSION,
                            "request_sha256": _extraction_request_hash(
                                str(chunks[index]["content"]),
                                schema_correction=uses_schema_feedback,
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    next_pending.append(index)
                logger.info(
                    "Extraction attempt %d/%d: batch %d/%d complete; accepted %d/%d (%.1f%%), awaiting retry %d",
                    attempt + 1,
                    schema_retries + 1,
                    batch_number,
                    attempt_batches,
                    accepted_total,
                    len(chunks),
                    100.0 * accepted_total / len(chunks) if chunks else 100.0,
                    len(next_pending),
                )
                continue
            for index, raw_value in zip(indices, raw_values):
                if isinstance(raw_value, Mapping) and "_extract_error" in raw_value:
                    attempt_record = {
                        "attempt": attempt + 1,
                        "response_format": {"type": "json_object"},
                        "response_format_version": EXTRACTION_RESPONSE_FORMAT_VERSION,
                        "request_sha256": _extraction_request_hash(
                            str(chunks[index]["content"]),
                            schema_correction=uses_schema_feedback,
                        ),
                        "error_type": str(raw_value.get("_extract_error_type", "ExtractionError")),
                        "error": str(raw_value.get("_extract_error", "unknown extraction error")),
                    }
                    if raw_value.get("_response_sha256"):
                        attempt_record["response_sha256"] = str(raw_value["_response_sha256"])
                    if raw_value.get("_raw_response") is not None:
                        attempt_record["raw_response"] = str(raw_value["_raw_response"])
                    provider_trace = raw_value.get("_extract_trace")
                    if isinstance(provider_trace, Mapping):
                        attempt_record["provider_trace"] = dict(provider_trace)
                        attempt_record["trace_identity"] = {
                            key: provider_trace[key]
                            for key in (
                                "status",
                                "provider_call_succeeded",
                                "stage",
                                "provider",
                                "model",
                                "model_revision",
                                "response_format",
                                "response_format_version",
                                "request_sha256",
                                "response_sha256",
                                "attempt_count",
                                "usage",
                            )
                            if key in provider_trace
                        }
                    provider_attempts = raw_value.get("_extract_attempts")
                    if isinstance(provider_attempts, list):
                        attempt_record["provider_attempts"] = [
                            dict(item) for item in provider_attempts if isinstance(item, Mapping)
                        ]
                    traces[index]["attempts"].append(attempt_record)
                    if attempt >= schema_retries:
                        raise GraphBuildError(
                            f"chunk {chunks[index]['chunk_id']} failed extraction after "
                            f"{schema_retries + 1} attempts: {attempt_record['error']}"
                        )
                    next_pending.append(index)
                    continue
                payload, trace = _unwrap_extraction(raw_value)
                attempt_record: Dict[str, Any] = {
                    "attempt": attempt + 1,
                    "response_format": {"type": "json_object"},
                    "response_format_version": EXTRACTION_RESPONSE_FORMAT_VERSION,
                    "request_sha256": _extraction_request_hash(
                        str(chunks[index]["content"]),
                        schema_correction=uses_schema_feedback,
                    ),
                }
                raw_response = _raw_response_from_trace(trace)
                if raw_response is not None:
                    attempt_record["raw_response"] = raw_response
                    attempt_record["raw_response_sha256"] = _content_sha256(
                        raw_response
                    )
                if trace is not None:
                    attempt_record["provider_trace"] = dict(trace)
                    attempt_record["trace_identity"] = {
                        key: trace[key]
                        for key in (
                            "status",
                            "provider_call_succeeded",
                            "stage",
                            "provider",
                            "model",
                            "model_revision",
                            "response_format",
                            "response_format_version",
                            "request_sha256",
                            "response_sha256",
                            "attempt_count",
                            "usage",
                        )
                        if key in trace
                    }
                try:
                    normalised = validate_extraction_payload(
                        payload,
                        trace=trace,
                        require_trace=require_trace,
                        source_text=str(chunks[index]["content"]),
                    )
                except ExtractionSchemaError as exc:
                    attempt_record["error"] = str(exc)
                    if attempt >= schema_retries:
                        try:
                            (
                                normalised,
                                relation_rejections,
                                grounding_rejections,
                                proposition_schema_rejections,
                            ) = _project_final_extraction(
                                payload,
                                trace=trace,
                                require_trace=require_trace,
                                source_text=str(chunks[index]["content"]),
                            )
                        except ExtractionSchemaError as projection_exc:
                            traces[index]["attempts"].append(attempt_record)
                            raise GraphBuildError(
                                f"chunk {chunks[index]['chunk_id']} failed extraction schema after "
                                f"{schema_retries + 1} attempts: strict validation error: {exc}; "
                                f"final projection error: {projection_exc}"
                            ) from projection_exc
                        attempt_record["response_sha256"] = _safe_audit_sha256(payload)
                        attempt_record["relation_admissibility_policy_version"] = (
                            RELATION_ADMISSIBILITY_POLICY_VERSION
                        )
                        attempt_record["relation_rejections"] = relation_rejections
                        attempt_record["grounding_policy_version"] = GROUNDING_POLICY_VERSION
                        attempt_record["grounding_rejections"] = grounding_rejections
                        attempt_record["proposition_schema_projection_policy_version"] = (
                            PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
                        )
                        attempt_record["proposition_schema_rejections"] = (
                            proposition_schema_rejections
                        )
                        attempt_record["projected_response_sha256"] = canonical_sha256(
                            normalised
                        )
                        if "raw_response" not in attempt_record:
                            try:
                                synthetic_raw = canonical_json(payload)
                            except (TypeError, ValueError, OverflowError):
                                synthetic_raw = ""
                            if synthetic_raw:
                                attempt_record["raw_response"] = synthetic_raw
                                attempt_record["raw_response_sha256"] = _content_sha256(
                                    synthetic_raw
                                )
                        traces[index]["relation_rejections"] = relation_rejections
                        traces[index]["grounding_rejections"] = grounding_rejections
                        traces[index]["grounding_policy_version"] = GROUNDING_POLICY_VERSION
                        traces[index]["proposition_schema_rejections"] = (
                            proposition_schema_rejections
                        )
                        traces[index]["proposition_schema_projection_policy_version"] = (
                            PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
                        )
                        traces[index]["projected_response_sha256"] = attempt_record[
                            "projected_response_sha256"
                        ]
                        if (
                            (grounding_rejections or proposition_schema_rejections)
                            and not normalised["propositions"]
                            and not normalised["causal_relations"]
                        ):
                            traces[index]["status"] = "success_projected_empty"
                        elif proposition_schema_rejections:
                            traces[index]["status"] = "success_with_proposition_schema_rejections"
                        elif grounding_rejections:
                            traces[index]["status"] = "success_with_grounding_rejections"
                        else:
                            traces[index]["status"] = "success_with_relation_rejections"
                        traces[index]["attempts"].append(attempt_record)
                        results[index] = normalised
                        accepted_total += 1
                        logger.warning(
                            "Chunk %s admitted after omitting %d schema-invalid proposition(s), %d grounding proposition(s), and %d inadmissible causal relation(s)",
                            chunks[index]["chunk_id"],
                            len(proposition_schema_rejections),
                            len(grounding_rejections),
                            len(relation_rejections),
                        )
                        continue
                    traces[index]["attempts"].append(attempt_record)
                    next_pending.append(index)
                else:
                    attempt_record["response_sha256"] = _safe_audit_sha256(payload)
                    if "raw_response" not in attempt_record:
                        synthetic_raw = canonical_json(payload)
                        attempt_record["raw_response"] = synthetic_raw
                        attempt_record["raw_response_sha256"] = _content_sha256(
                            synthetic_raw
                        )
                    traces[index]["attempts"].append(attempt_record)
                    traces[index]["status"] = (
                        "success_explicit_empty"
                        if not normalised["propositions"] and not normalised["causal_relations"]
                        else "success"
                    )
                    results[index] = normalised
                    accepted_total += 1
            logger.info(
                "Extraction attempt %d/%d: batch %d/%d complete; accepted %d/%d (%.1f%%), awaiting retry %d",
                attempt + 1,
                schema_retries + 1,
                batch_number,
                attempt_batches,
                accepted_total,
                len(chunks),
                100.0 * accepted_total / len(chunks) if chunks else 100.0,
                len(next_pending),
            )
        pending = next_pending
        if not pending:
            break
    if pending or any(result is None for result in results):
        raise GraphBuildError("extraction did not close over every chunk")
    logger.info("Extraction complete: accepted %d/%d chunks", accepted_total, len(chunks))
    return [result for result in results if result is not None], traces


def _build_occurrence_graph(
    chunks: Sequence[Mapping[str, Any]],
    extractions: Sequence[Mapping[str, Any]],
    *,
    graph_namespace: str,
    graph_path: Path,
) -> Tuple[GraphStorage, Dict[str, Dict[str, Any]], Dict[str, str]]:
    graph = GraphStorage(str(graph_path), create=True)
    graph._graph.graph.update({"schema_version": GRAPH_SCHEMA, "base_graph_namespace": graph_namespace})
    entity_accumulator: Dict[str, Dict[str, Any]] = {}
    hyperedge_text: Dict[str, str] = {}
    support_candidates: List[Dict[str, Any]] = []
    for chunk, extraction in zip(chunks, extractions):
        chunk_entity_ids: Dict[str, str] = {}
        chunk_hyperedge_ids: List[str] = []
        for proposition_index, proposition in enumerate(extraction["propositions"]):
            source_proposition_index = proposition.get(
                "source_proposition_index", proposition_index
            )
            if (
                isinstance(source_proposition_index, bool)
                or not isinstance(source_proposition_index, int)
                or source_proposition_index < 0
            ):
                raise GraphBuildError("validated proposition source index is invalid")
            proposition_key = proposition["proposition_key"]
            hyperedge_id = hyperedge_node_id(
                graph_namespace,
                str(chunk["chunk_id"]),
                source_proposition_index,
                proposition_key,
            )
            chunk_hyperedge_ids.append(hyperedge_id)
            hyperedge_text[hyperedge_id] = proposition["sentence"]
            graph.upsert_node(
                hyperedge_id,
                {
                    "role": "hyperedge",
                    "name": proposition["sentence"],
                    "proposition_key": proposition_key,
                    "source_id": chunk["chunk_id"],
                    "chunk_id": chunk["chunk_id"],
                    "record_id": chunk["record_id"],
                    "record_chunk_index": chunk["record_chunk_index"],
                    "proposition_index": source_proposition_index,
                    "source": chunk["source"],
                    "source_path": chunk["source_path"],
                    "timestamp_raw": chunk["timestamp_raw"],
                    "timestamp_iso": chunk["timestamp_iso"],
                    "date": chunk["date"],
                    "availability_upper_bound_iso": chunk["availability_upper_bound_iso"],
                    "availability_upper_bound_date": chunk["availability_upper_bound_date"],
                    "availability_bound_kind": chunk["availability_bound_kind"],
                    "availability_bound_source": chunk["availability_bound_source"],
                    "availability_bound_source_sha256": chunk["availability_bound_source_sha256"],
                    "semantic_relevance": 0.5,
                },
            )
            support_candidates.append(
                {
                    "proposition_key": proposition_key,
                    "record_id": chunk["record_id"],
                    "hyperedge_id": hyperedge_id,
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "timestamp_raw": chunk["timestamp_raw"],
                    "timestamp_iso": chunk["timestamp_iso"],
                    "date": chunk["date"],
                    "availability_upper_bound_iso": chunk["availability_upper_bound_iso"],
                    "availability_upper_bound_date": chunk["availability_upper_bound_date"],
                    "availability_bound_kind": chunk["availability_bound_kind"],
                    "availability_bound_source": chunk["availability_bound_source"],
                    "availability_bound_source_sha256": chunk["availability_bound_source_sha256"],
                }
            )
            for entity in proposition["entities"]:
                entity_key = entity["canonical_key"]
                entity_id = entity_node_id(graph_namespace, entity_key)
                chunk_entity_ids[entity_key] = entity_id
                accumulator = entity_accumulator.setdefault(
                    entity_id, {"canonical_entity_key": entity_key, "aliases": {}}
                )
                provenance = {
                    "hyperedge_id": hyperedge_id,
                    "chunk_id": chunk["chunk_id"],
                    "record_id": chunk["record_id"],
                    "source": chunk["source"],
                    "timestamp_iso": chunk["timestamp_iso"],
                    "date": chunk["date"],
                    "availability_upper_bound_iso": chunk["availability_upper_bound_iso"],
                }
                accumulator["aliases"].setdefault(entity["name"], []).append(provenance)
                graph.upsert_edge(
                    hyperedge_id,
                    entity_id,
                    {
                        "role": "incidence",
                        "incidence_role": "mentions",
                        "chunk_id": chunk["chunk_id"],
                        "record_id": chunk["record_id"],
                        "source": chunk["source"],
                    },
                    key=incidence_edge_id(graph_namespace, hyperedge_id, entity_id),
                )
        for relation in extraction["causal_relations"]:
            relation_index = int(relation["source_relation_index"])
            cause_id, effect_id = chunk_entity_ids.get(relation["cause_key"]), chunk_entity_ids.get(relation["effect_key"])
            if not cause_id or not effect_id:
                raise GraphBuildError("validated relation endpoint absent from chunk entity closure")
            edge_id = causal_edge_id(
                graph_namespace, str(chunk["chunk_id"]), relation_index, cause_id, effect_id, relation["type"]
            )
            graph.upsert_edge(
                cause_id,
                effect_id,
                {
                    "role": "causal",
                    "causal_type": relation["type"],
                    "strength": relation["strength"],
                    "description": relation["description"],
                    "source_id": chunk["chunk_id"],
                    "chunk_id": chunk["chunk_id"],
                    "record_id": chunk["record_id"],
                    "source_hyperedge_ids": chunk_hyperedge_ids,
                    "relation_index": relation_index,
                    "source": chunk["source"],
                    "source_path": chunk["source_path"],
                    "timestamp_raw": chunk["timestamp_raw"],
                    "timestamp_iso": chunk["timestamp_iso"],
                    "date": chunk["date"],
                    "availability_upper_bound_iso": chunk["availability_upper_bound_iso"],
                    "availability_upper_bound_date": chunk["availability_upper_bound_date"],
                    "availability_bound_kind": chunk["availability_bound_kind"],
                    "availability_bound_source": chunk["availability_bound_source"],
                    "availability_bound_source_sha256": chunk["availability_bound_source_sha256"],
                },
                key=edge_id,
            )
    for entity_id in sorted(entity_accumulator):
        accumulator = entity_accumulator[entity_id]
        aliases = sorted(accumulator["aliases"], key=lambda value: (canonical_entity_key(value), value))
        provenance = {
            alias: sorted(
                accumulator["aliases"][alias],
                key=lambda item: (item["record_id"], item["chunk_id"], item["hyperedge_id"]),
            )
            for alias in aliases
        }
        graph.upsert_node(
            entity_id,
            {
                "role": "entity",
                "name": aliases[0],
                "canonical_entity_key": accumulator["canonical_entity_key"],
                "aliases": aliases,
                "alias_provenance": provenance,
            },
        )
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in sorted(support_candidates, key=lambda item: (item["proposition_key"], item["record_id"], item["hyperedge_id"])):
        selected.setdefault((entry["proposition_key"], entry["record_id"]), entry)
    ledger = sorted(selected.values(), key=lambda item: (item["proposition_key"], item["record_id"], item["hyperedge_id"]))
    counts: Dict[str, int] = {}
    for entry in ledger:
        counts[entry["proposition_key"]] = counts.get(entry["proposition_key"], 0) + 1
    graph._graph.graph["support_ledger"] = "json:" + canonical_json(ledger)
    graph._graph.graph["recurrence_corpus_counts"] = "json:" + canonical_json(dict(sorted(counts.items())))
    return graph, entity_accumulator, hyperedge_text


def _embed_graph_nodes(
    graph: GraphStorage,
    entity_accumulator: Mapping[str, Mapping[str, Any]],
    hyperedge_text: Mapping[str, str],
    *,
    embedder: Any,
    signature: Mapping[str, Any],
    entity_path: Path,
    hyperedge_path: Path,
) -> Tuple[VectorStorage, VectorStorage, List[Dict[str, Any]]]:
    dimension = int(signature["dimension"])
    entity_store = VectorStorage(str(entity_path), dimension, create=True, signature=signature, role="entity")
    hyperedge_store = VectorStorage(
        str(hyperedge_path), dimension, create=True, signature=signature, role="hyperedge_occurrence"
    )
    entity_ids, hyperedge_ids = sorted(entity_accumulator), sorted(hyperedge_text)
    entity_texts = [str(entity_accumulator[node_id]["canonical_entity_key"]) for node_id in entity_ids]
    hyperedge_texts = [hyperedge_text[node_id] for node_id in hyperedge_ids]
    logger.info(
        "Embedding started: %d entities and %d hyperedges",
        len(entity_texts),
        len(hyperedge_texts),
    )
    embedding_traces: List[Dict[str, Any]] = []
    embedding_trace_lock = threading.Lock()

    def embed_with_trace(texts: List[str], *, role: str) -> List[List[float]]:
        if not texts:
            return []



        if isinstance(embedder, Embedder):
            return embedder.embed(
                texts,
                trace_sink=embedding_traces,
                trace_lock=embedding_trace_lock,
                trace_role=role,
            )
        return embedder.embed(texts)

    entity_vectors = embed_with_trace(entity_texts, role="entity_embedding")
    logger.info("Entity embedding complete: %d/%d", len(entity_vectors), len(entity_ids))
    hyperedge_vectors = embed_with_trace(hyperedge_texts, role="hyperedge_embedding")
    logger.info(
        "Hyperedge embedding complete: %d/%d",
        len(hyperedge_vectors),
        len(hyperedge_ids),
    )
    if len(entity_vectors) != len(entity_ids) or len(hyperedge_vectors) != len(hyperedge_ids):
        raise GraphBuildError("embedding output does not close over graph nodes")
    entity_store.upsert_batch(
        [
            {
                "id": _typed_sha256(
                    "vec_",
                    {"role": "entity", "node_id": node_id, "text_sha256": _content_sha256(text), "embedding_signature": signature},
                ),
                "vector": vector,
                "meta": {
                    "node_id": node_id,
                    "entity_name": node_id,
                    "canonical_entity_key": text,
                    "content": text,
                },
            }
            for node_id, text, vector in zip(entity_ids, entity_texts, entity_vectors)
        ]
    )
    hyperedge_store.upsert_batch(
        [
            {
                "id": _typed_sha256(
                    "vec_",
                    {"role": "hyperedge_occurrence", "node_id": node_id, "text_sha256": _content_sha256(text), "embedding_signature": signature},
                ),
                "vector": vector,
                "meta": {
                    "node_id": node_id,
                    "hyperedge_name": node_id,
                    "content": text,
                    "source_id": (graph.get_node(node_id) or {}).get("chunk_id", ""),
                },
            }
            for node_id, text, vector in zip(hyperedge_ids, hyperedge_texts, hyperedge_vectors)
        ]
    )
    embedding_traces.sort(key=canonical_json)
    return entity_store, hyperedge_store, embedding_traces


def _provider_usage(
    extraction_traces: Sequence[Mapping[str, Any]] = (),
    embedding_traces: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, int]:


    result = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }

    def add_usage(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            raw = value.get(key, 0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            if math.isfinite(float(raw)):
                result[key] += int(raw)

    def consume_transport(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        attempts = value.get("attempts")
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, Mapping):
                    result["api_calls"] += 1
                    add_usage(attempt.get("usage"))
            return
        count = value.get("attempt_count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            result["api_calls"] += count



            add_usage(value.get("usage"))
            return
        if value.get("provider_call_succeeded") is True or value.get("status") == "ok":
            result["api_calls"] += 1
            add_usage(value.get("usage"))

    for chunk_trace in extraction_traces:
        if not isinstance(chunk_trace, Mapping):
            continue
        outer_attempts = chunk_trace.get("attempts")
        if not isinstance(outer_attempts, list):
            continue
        for outer_attempt in outer_attempts:
            if not isinstance(outer_attempt, Mapping):
                continue
            provider_trace = outer_attempt.get("provider_trace")
            if isinstance(provider_trace, Mapping):
                consume_transport(provider_trace)
                continue
            provider_attempts = outer_attempt.get("provider_attempts")
            if isinstance(provider_attempts, list):
                for attempt in provider_attempts:
                    if isinstance(attempt, Mapping):
                        result["api_calls"] += 1
                        add_usage(attempt.get("usage"))

    for embedding_trace in embedding_traces:
        if isinstance(embedding_trace, Mapping):
            consume_transport(embedding_trace.get("trace", embedding_trace))

    return result


def _relation_rejection_ledger(
    extraction_traces: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for trace in extraction_traces:
        chunk_id = trace.get("chunk_id")
        rejections = trace.get("relation_rejections", [])
        if not isinstance(chunk_id, str) or not isinstance(rejections, list):
            continue
        for rejection in rejections:
            if isinstance(rejection, Mapping):
                ledger.append({"chunk_id": chunk_id, **dict(rejection)})
    ledger.sort(
        key=lambda item: (
            str(item["chunk_id"]),
            int(item["relation_index"]),
            str(item["reason_code"]),
            str(item["relation_sha256"]),
        )
    )
    return ledger


def _grounding_rejection_ledger(
    extraction_traces: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for trace in extraction_traces:
        chunk_id = trace.get("chunk_id")
        rejections = trace.get("grounding_rejections", [])
        if not isinstance(chunk_id, str) or not isinstance(rejections, list):
            continue
        for rejection in rejections:
            if isinstance(rejection, Mapping):
                ledger.append({"chunk_id": chunk_id, **dict(rejection)})
    ledger.sort(
        key=lambda item: (
            str(item["chunk_id"]),
            int(item["proposition_index"]),
            str(item["reason_code"]),
            str(item["proposition_sha256"]),
        )
    )
    return ledger


def _proposition_schema_rejection_ledger(
    extraction_traces: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for trace in extraction_traces:
        chunk_id = trace.get("chunk_id")
        rejections = trace.get("proposition_schema_rejections", [])
        if not isinstance(chunk_id, str) or not isinstance(rejections, list):
            raise GraphBuildError(
                "proposition schema rejection trace is not auditable"
            )
        if not rejections:
            continue
        attempts = trace.get("attempts")
        if not isinstance(attempts, list) or not attempts or not isinstance(
            attempts[-1], Mapping
        ):
            raise GraphBuildError(
                "proposition schema rejection lacks a final extraction attempt"
            )
        final_attempt = attempts[-1]
        raw_response_sha256 = final_attempt.get("raw_response_sha256")
        projected_response_sha256 = trace.get("projected_response_sha256")
        policy_version = trace.get(
            "proposition_schema_projection_policy_version"
        )
        if (
            not isinstance(raw_response_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", raw_response_sha256)
            or not isinstance(projected_response_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", projected_response_sha256)
            or policy_version != PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
        ):
            raise GraphBuildError(
                "proposition schema rejection lacks sealed raw/projected evidence"
            )
        for rejection in rejections:
            if not isinstance(rejection, Mapping):
                raise GraphBuildError(
                    "proposition schema rejection entry is not auditable"
                )
            ledger.append(
                {
                    "chunk_id": chunk_id,
                    **dict(rejection),
                    "raw_response_sha256": raw_response_sha256,
                    "projected_response_sha256": projected_response_sha256,
                    "proposition_schema_projection_policy_version": policy_version,
                }
            )
    ledger.sort(
        key=lambda item: (
            str(item["chunk_id"]),
            int(item["proposition_index"]),
            str(item["reason_code"]),
            str(item["proposition_sha256"]),
        )
    )
    return ledger


def _grounding_summary(
    extractions: Sequence[Mapping[str, Any]],
    extraction_traces: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:


    propositions_admitted = sum(
        len(extraction.get("propositions", [])) for extraction in extractions
    )
    entities_admitted = sum(
        len(proposition.get("entities", []))
        for extraction in extractions
        for proposition in extraction.get("propositions", [])
    )
    causal_relations_admitted = sum(
        len(extraction.get("causal_relations", [])) for extraction in extractions
    )
    grounding_rejections = [
        rejection
        for trace in extraction_traces
        for rejection in trace.get("grounding_rejections", [])
        if isinstance(rejection, Mapping)
    ]
    proposition_schema_rejections = [
        rejection
        for trace in extraction_traces
        for rejection in trace.get("proposition_schema_rejections", [])
        if isinstance(rejection, Mapping)
    ]
    return {
        "status": "passed",
        "chunks": len(extractions),
        "propositions": propositions_admitted,
        "entities": entities_admitted,
        "causal_relations": causal_relations_admitted,
        "propositions_admitted": propositions_admitted,
        "entities_admitted": entities_admitted,
        "causal_relations_admitted": causal_relations_admitted,
        "propositions_rejected_grounding": len(grounding_rejections),
        "propositions_rejected_schema": len(proposition_schema_rejections),
        "entities_omitted_grounding": sum(
            int(rejection.get("entity_count", 0))
            for rejection in grounding_rejections
        ),
        "projected_empty_chunks": sum(
            trace.get("status") == "success_projected_empty"
            for trace in extraction_traces
        ),
    }


def build_graph_bundle(
    source: os.PathLike[str] | str,
    graph_dir: os.PathLike[str] | str,
    *,
    resolved_config: Any = None,
    profile: Optional[str] = None,
    extractor: Any = None,
    embedder: Any = None,
    force: bool = False,
    allow_compat_adapters: bool = False,
    require_extraction_trace: bool = False,
    scratch_dir: Optional[os.PathLike[str] | str] = None,
    source_descriptor: Optional[str] = None,
) -> Dict[str, Any]:


    target = Path(graph_dir)
    logger.info("Build started: source=%s target=%s", source, target)







    if target.exists() and not force:
        logger.info("Existing target found; checking portable completion")
        try:
            existing = load_graph_bundle_portable(target)
        except Exception as exc:
            raise GraphBuildError(
                "existing target is incomplete or structurally invalid; "
                "use --force to replace it"
            ) from exc
        existing_audit = existing.get("source_audit")
        input_count = (
            len(existing_audit.get("inputs", []))
            if isinstance(existing_audit, Mapping)
            and isinstance(existing_audit.get("inputs"), list)
            else 0
        )
        manifest = existing.get("manifest")
        base_namespace = (
            str(manifest.get("base_graph_namespace", ""))
            if isinstance(manifest, Mapping)
            else ""
        )
        logger.info("Existing graph is complete; reusing it without provider calls")
        return {
            **existing["stats"],
            "files_found": input_count,
            "files_processed": 0,
            "files_skipped": input_count,
            "chunks_created": 0,
            "entities_new": 0,
            "hyperedges_new": 0,
            "causal_edges_new": 0,
            "graph_bundle_sha256": existing["graph_bundle_sha256"],
            "base_graph_namespace": base_namespace,
            "reused": True,
            "audit_refreshed": False,
        }

    if scratch_dir is not None:
        scratch = Path(scratch_dir)


        if scratch.resolve() == target.resolve():
            raise GraphBuildError("scratch_dir must not alias graph_dir")

    profile_id = _profile_id(resolved_config, profile)
    if profile_id == PAPER_PROFILE_ID and allow_compat_adapters:
        raise GraphBuildError("paper builds cannot enable general-file adapters")
    if profile_id == PAPER_PROFILE_ID:
        from chain.config import ResolvedConfig

        if not isinstance(resolved_config, ResolvedConfig):
            raise GraphBuildError(
                "paper graph construction requires an immutable ResolvedConfig"
            )
        if extractor is not None or embedder is not None:
            raise GraphBuildError(
                "paper graph construction does not accept injected extractor/embedder fixtures"
            )


    records, source_audit = load_source_records(
        source,
        allow_compat_adapters=allow_compat_adapters,
        source_descriptor=source_descriptor,
    )
    logger.info(
        "Input loaded: %d records from %d JSONL file(s)",
        len(records),
        len(source_audit["inputs"]),
    )
    provisional_signature = _embedding_signature(resolved_config, embedder or object())
    construction = _construction_config(resolved_config, profile_id, provisional_signature)
    if embedder is None:
        embedder = _default_embedder(resolved_config, provisional_signature)
    signature = _embedding_signature(resolved_config, embedder)
    if canonical_json(signature) != canonical_json(provisional_signature):
        raise GraphBuildError("resolved embedder signature changed after preflight")
    construction["embedding_signature"] = signature



    scientific_hash = _scientific_config_hash(resolved_config, construction)
    construction_hash = _construction_config_hash(resolved_config, construction)
    construction_fingerprint = canonical_sha256(
        {
            "canonical_construction_stream_sha256": source_audit["canonical_construction_stream_sha256"],
            "resolved_construction_config": construction,
            "construction_config_sha256": construction_hash,
        }
    )
    graph_namespace = _typed_sha256(
        "graph_",
        {
            "canonical_construction_stream_sha256": source_audit["canonical_construction_stream_sha256"],
            "construction_fingerprint_sha256": construction_fingerprint,
        },
    )
    source_audit = {
        **source_audit,
        "construction_fingerprint_sha256": construction_fingerprint,
        "base_graph_namespace": graph_namespace,
    }

    chunks = records_to_chunks(
        records,
        graph_namespace=graph_namespace,
        max_tokens=int(construction["chunk_max_tokens"]),
        overlap=int(construction["chunk_overlap_tokens"]),
        tokenizer_model=str(construction["requested_tokenizer_model"]),
    )
    if not chunks:
        raise GraphBuildError("record parser produced no chunks")
    logger.info(
        "Chunking complete: %d records produced %d chunks",
        len(records),
        len(chunks),
    )
    if extractor is None:
        extractor = _default_extractor(resolved_config, construction)
        require_extraction_trace = True
    extractions, extraction_traces = extract_chunks(
        chunks,
        extractor,
        batch_size=int(construction["extraction_batch_size"]),
        schema_retries=int(construction["extraction_schema_retries"]),
        require_trace=require_extraction_trace,
    )

    logger.info("Creating isolated staging bundle")
    staging = make_staging_directory(target)
    try:
        logger.info("Graph assembly started")
        graph, entity_accumulator, hyperedge_text = _build_occurrence_graph(
            chunks,
            extractions,
            graph_namespace=graph_namespace,
            graph_path=staging / "knowledge_graph.graphml",
        )
        logger.info(
            "Graph assembly complete: %d entities, %d hyperedges, %d causal edges",
            len(entity_accumulator),
            len(hyperedge_text),
            graph.causal_edge_count(),
        )
        graph._graph.graph.update(
            {
                "construction_config_sha256": construction_hash,
                "canonical_construction_stream_sha256": source_audit["canonical_construction_stream_sha256"],
                "construction_fingerprint_sha256": construction_fingerprint,
            }
        )
        entity_store, hyperedge_store, embedding_traces = _embed_graph_nodes(
            graph,
            entity_accumulator,
            hyperedge_text,
            embedder=embedder,
            signature=signature,
            entity_path=staging / "entity_vdb.json",
            hyperedge_path=staging / "hyperedge_vdb.json",
        )
        logger.info("Writing graph artifacts and audit metadata")
        write_canonical_json(
            staging / "chunks.json",
            {"schema_version": CHUNKS_SCHEMA, "chunks": {chunk["chunk_id"]: chunk for chunk in chunks}},
        )
        entity_store.save()
        hyperedge_store.save()
        graph.save()
        artifacts = {
            name: {"sha256": sha256_file(staging / name), "bytes": (staging / name).stat().st_size}
            for name in KG_ARTIFACTS
        }
        publishable = profile_id == PAPER_PROFILE_ID
        counts = {
            "chunks": len(chunks),
            "entities": len(entity_accumulator),
            "hyperedges": len(hyperedge_text),
            "causal_edges": graph.causal_edge_count(),
            "edges": graph._graph.number_of_edges(),
        }
        manifest: Dict[str, Any] = {
            "schema_version": GRAPH_BUNDLE_SCHEMA,
            "profile_id": profile_id,
            "publishable": publishable,
            "out_of_paper_protocol": not publishable,
            "artifacts": artifacts,
            "canonical_construction_stream_sha256": source_audit["canonical_construction_stream_sha256"],
            "construction_fingerprint_sha256": construction_fingerprint,
            "base_graph_namespace": graph_namespace,
            "construction_config_sha256": construction_hash,
            "scientific_config_sha256": scientific_hash,
            "resolved_config": {
                "construction_config_sha256": construction_hash,
                "construction": construction,
            },
            "resolved_construction_config": construction,
            "embedding_signatures": {
                "entity": signature,
                "hyperedge": signature,
                "query": signature,
                "target": signature,
            },
        }
        relation_rejection_ledger = _relation_rejection_ledger(extraction_traces)
        grounding_rejection_ledger = _grounding_rejection_ledger(extraction_traces)
        proposition_schema_rejection_ledger = _proposition_schema_rejection_ledger(
            extraction_traces
        )
        telemetry: Dict[str, Any] = {
            "schema_version": TELEMETRY_SCHEMA,
            "status": "success",
            "errors": [],
            "profile_id": profile_id,
            "publishable": publishable,
            "out_of_paper_protocol": not publishable,
            "artifacts": artifacts,
            "counts": counts,
            "scientific": {
                "extraction_chunks_total": len(chunks),
                "success": sum(
                    trace["status"]
                    in {
                        "success",
                        "success_with_relation_rejections",
                        "success_with_grounding_rejections",
                        "success_with_proposition_schema_rejections",
                    }
                    for trace in extraction_traces
                ),
                "success_explicit_empty": sum(
                    trace["status"] == "success_explicit_empty"
                    for trace in extraction_traces
                ),
                "success_with_relation_rejections": sum(
                    bool(trace.get("relation_rejections", []))
                    for trace in extraction_traces
                ),
                "success_with_grounding_rejections": sum(
                    trace["status"] == "success_with_grounding_rejections"
                    for trace in extraction_traces
                ),
                "success_with_proposition_schema_rejections": sum(
                    trace["status"]
                    == "success_with_proposition_schema_rejections"
                    for trace in extraction_traces
                ),
                "success_projected_empty": sum(
                    trace["status"] == "success_projected_empty"
                    for trace in extraction_traces
                ),
                "causal_relations_rejected": sum(
                    len(trace.get("relation_rejections", []))
                    for trace in extraction_traces
                ),
                "relation_rejection_ledger_sha256": canonical_sha256(
                    relation_rejection_ledger
                ),
                "relation_admissibility_policy_version": (
                    RELATION_ADMISSIBILITY_POLICY_VERSION
                ),
                "grounding_policy_version": GROUNDING_POLICY_VERSION,
                "propositions_rejected_grounding": len(
                    grounding_rejection_ledger
                ),
                "entities_omitted_grounding": sum(
                    int(entry["entity_count"])
                    for entry in grounding_rejection_ledger
                ),
                "grounding_rejection_ledger_sha256": canonical_sha256(
                    grounding_rejection_ledger
                ),
                "proposition_schema_projection_policy_version": (
                    PROPOSITION_SCHEMA_PROJECTION_POLICY_VERSION
                ),
                "propositions_rejected_schema": len(
                    proposition_schema_rejection_ledger
                ),
                "proposition_schema_rejection_ledger_sha256": canonical_sha256(
                    proposition_schema_rejection_ledger
                ),
                "grounding_replay": _grounding_summary(
                    extractions, extraction_traces
                ),
                "attempts_total": sum(len(trace["attempts"]) for trace in extraction_traces),
                "checkpoint_required": False,
            },
            "operational": {
                "provider_usage": _provider_usage(extraction_traces, embedding_traces),




                "extraction_traces": extraction_traces,
                "relation_rejection_ledger": relation_rejection_ledger,
                "grounding_rejection_ledger": grounding_rejection_ledger,
                "proposition_schema_rejection_ledger": (
                    proposition_schema_rejection_ledger
                ),
                "embedding_traces": embedding_traces,
            },
            "construction_config_sha256": construction_hash,
            "scientific_config_sha256": scientific_hash,
            "construction_fingerprint_sha256": construction_fingerprint,
        }
        preliminary = graph_bundle_payload(manifest, telemetry, artifacts)
        manifest["manifest_payload_sha256"] = preliminary["manifest_payload_sha256"]
        telemetry["telemetry_payload_sha256"] = preliminary["telemetry_payload_sha256"]
        graph_hash = canonical_sha256(graph_bundle_payload(manifest, telemetry, artifacts))
        manifest["graph_bundle_sha256"] = graph_hash
        telemetry["graph_bundle_sha256"] = graph_hash
        source_audit["graph_bundle_sha256"] = graph_hash
        write_canonical_json(staging / "graph_manifest.json", manifest)
        write_canonical_json(staging / "build_telemetry.json", telemetry)
        write_canonical_json(staging / SOURCE_AUDIT_FILE, source_audit)
        logger.info("Portable completion validation of the staged graph started")
        load_graph_bundle_portable(staging)
        logger.info("Staged graph completed; publishing atomically")
        atomic_publish_directory(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    final = load_graph_bundle_portable(target, expected_config=resolved_config)
    logger.info(
        "Build complete: %d chunks, %d entities, %d hyperedges, %d causal edges",
        final["stats"]["chunks"],
        final["stats"]["entities"],
        final["stats"]["hyperedges"],
        final["stats"]["causal_edges"],
    )
    return {
        **final["stats"],
        "files_found": len(source_audit["inputs"]),
        "files_processed": len(source_audit["inputs"]),
        "files_skipped": 0,
        "chunks_created": counts["chunks"],
        "entities_new": counts["entities"],
        "hyperedges_new": counts["hyperedges"],
        "causal_edges_new": counts["causal_edges"],
        "graph_bundle_sha256": final["graph_bundle_sha256"],
        "base_graph_namespace": graph_namespace,
        "reused": False,
        "audit_refreshed": False,
    }


def _legacy_build_config(kwargs: Mapping[str, Any], embedder: Any) -> Dict[str, Any]:
    config = dict(kwargs.get("resolved_config") or {})
    config.setdefault("chunk_max_tokens", int(kwargs.get("max_tokens", 512)))
    config.setdefault("chunk_overlap_tokens", int(kwargs.get("overlap", 64)))
    config.setdefault("extraction_batch_size", int(kwargs.get("extract_batch_size", 5)))
    config.setdefault("profile_id", COMPAT_PROFILE_ID)
    signature = getattr(embedder, "signature", None)
    if isinstance(signature, Mapping):
        config.setdefault("embedding_signature", dict(signature))
    return config


def build_from_folder(
    folder_path: str,
    kb_dir: str,
    extractor: Any,
    embedder: Any,
    kv: Any = None,
    vdb_entity: Any = None,
    vdb_hyperedge: Any = None,
    graph: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    del kv, vdb_entity, vdb_hyperedge, graph
    return build_graph_bundle(
        folder_path,
        kb_dir,
        resolved_config=_legacy_build_config(kwargs, embedder),
        profile="compat",
        extractor=extractor,
        embedder=embedder,
        allow_compat_adapters=True,
    )


def build_from_file(
    file_path: str,
    kb_dir: str,
    extractor: Any,
    embedder: Any,
    kv: Any = None,
    vdb_entity: Any = None,
    vdb_hyperedge: Any = None,
    graph: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    del kv, vdb_entity, vdb_hyperedge, graph
    return build_graph_bundle(
        file_path,
        kb_dir,
        resolved_config=_legacy_build_config(kwargs, embedder),
        profile="compat",
        extractor=extractor,
        embedder=embedder,
        allow_compat_adapters=True,
    )


def retrieve(
    query: str,
    embedder: Any,
    kv: JsonKVStorage,
    vdb_entity: VectorStorage,
    vdb_hyperedge: VectorStorage,
    graph: GraphStorage,
    top_k: int = COMPAT_RRF_DEFAULT_TOP_K,
    *,
    rrf_k: int = COMPAT_RRF_K,
    channel_candidate_multiplier: int = COMPAT_RRF_CHANNEL_CANDIDATE_MULTIPLIER,
) -> List[Dict[str, Any]]:
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    if (
        isinstance(rrf_k, bool)
        or not isinstance(rrf_k, int)
        or rrf_k <= 0
        or isinstance(channel_candidate_multiplier, bool)
        or not isinstance(channel_candidate_multiplier, int)
        or channel_candidate_multiplier <= 0
    ):
        raise GraphBuildError("compat RRF parameters must be positive integers")
    signature = getattr(embedder, "signature", None)
    if callable(signature):
        signature = signature()
    if isinstance(signature, Mapping):
        expected_space = tuple(signature.get(field) for field in EMBEDDING_SPACE_FIELDS)
        entity_space = tuple(
            vdb_entity.signature.get(field) for field in EMBEDDING_SPACE_FIELDS
        )
        hyperedge_space = tuple(
            vdb_hyperedge.signature.get(field) for field in EMBEDDING_SPACE_FIELDS
        )
        if expected_space != entity_space or expected_space != hyperedge_space:
            raise GraphBuildError(
                "query embedding space does not match graph vector stores"
            )
    query_vector = embedder.embed_one(query)
    candidate_count = top_k * channel_candidate_multiplier
    entity_hits = vdb_entity.search(query_vector, top_k=candidate_count)
    hyperedge_hits = vdb_hyperedge.search(query_vector, top_k=candidate_count)
    entity_chunks: List[str] = []
    for hit in entity_hits:
        node_id = hit.get("node_id") or hit.get("entity_name")
        for hyperedge_id in graph.predecessors(str(node_id)) if node_id else []:
            node = graph.get_node(hyperedge_id)
            chunk_id = str((node or {}).get("chunk_id") or (node or {}).get("source_id") or "")
            if chunk_id and chunk_id not in entity_chunks:
                entity_chunks.append(chunk_id)
    hyperedge_chunks: List[str] = []
    for hit in hyperedge_hits:
        node_id = hit.get("node_id") or hit.get("hyperedge_name")
        node = graph.get_node(str(node_id)) if node_id else None
        chunk_id = str((node or {}).get("chunk_id") or hit.get("source_id") or "")
        if chunk_id and chunk_id not in hyperedge_chunks:
            hyperedge_chunks.append(chunk_id)
    scores: Dict[str, float] = {}
    for rank, chunk_id in enumerate(entity_chunks):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + rrf_k)
    for rank, chunk_id in enumerate(hyperedge_chunks):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + rrf_k)
    result = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]:
        chunk = kv.get(chunk_id)
        if chunk is not None:
            result.append({**dict(chunk), "score": score, "chunk_id": chunk_id})
    return result


class KGEngine:


    def __init__(
        self,
        kb_dir: str,
        *,
        resolved_config: Any = None,
        expected_profile_id: Optional[str] = None,
        validate: bool = True,
        embedder: Any = None,
    ):
        self._kb_dir = Path(kb_dir)
        self._resolved_config = resolved_config
        self._expected_profile_id = expected_profile_id
        self._embedder = embedder
        self._validation: Optional[Dict[str, Any]] = None
        self._kv_chunks = self._vdb_entity = self._vdb_hyperedge = self._graph = None
        if validate:
            self._load()

    def _load(self) -> None:
        if self._resolved_config is None:




            from chain.config import resolve_config

            self._resolved_config = resolve_config("compat")
        validation = load_graph_bundle_portable(
            self._kb_dir,
            expected_config=self._resolved_config,
            expected_profile_id=self._expected_profile_id,
        )
        signature = validation["embedding_signatures"]["entity"]
        self._validation = validation
        self._kv_chunks = JsonKVStorage(str(self._kb_dir / "chunks.json"))
        self._vdb_entity = VectorStorage(str(self._kb_dir / "entity_vdb.json"), int(signature["dimension"]))
        self._vdb_hyperedge = VectorStorage(str(self._kb_dir / "hyperedge_vdb.json"), int(signature["dimension"]))
        runtime_graph = _decode_graphml_json_values(validation["graph"].copy())
        self._graph = GraphStorage(str(self._kb_dir / "knowledge_graph.graphml"), graph=runtime_graph)

    def _ensure_loaded(self) -> None:
        if self._validation is None:
            self._load()

    @property
    def validation(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return dict(self._validation)

    @property
    def graph_bundle_sha256(self) -> str:
        self._ensure_loaded()
        return str(self._validation["graph_bundle_sha256"])

    @property
    def embedding_signature(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return dict(self._validation["embedding_signatures"]["entity"])

    def _query_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = _default_embedder(self._resolved_config, self.embedding_signature)
        signature = getattr(self._embedder, "signature", None)
        if callable(signature):
            signature = signature()
        if not isinstance(signature, Mapping):
            raise GraphBuildError("query embedder does not expose an embedding-space signature")



        observed = {
            field: signature.get(field) for field in EMBEDDING_SPACE_FIELDS
        }
        expected = {
            field: self.embedding_signature.get(field)
            for field in EMBEDDING_SPACE_FIELDS
        }
        if observed != expected:
            raise GraphBuildError("query embedder is incompatible with the KG vector space")
        return self._embedder

    def ingest(self, path: str, *, force: bool = False) -> Dict[str, Any]:
        config = self._resolved_config
        if config is None:
            from chain.config import resolve_config

            config = resolve_config("compat")
        result = build_graph_bundle(
            path,
            self._kb_dir,
            resolved_config=config,
            profile="compat",
            embedder=self._embedder,
            force=force,
            allow_compat_adapters=True,
        )
        self._resolved_config = config
        self._validation = None
        self._kv_chunks = self._vdb_entity = self._vdb_hyperedge = self._graph = None
        self._load()
        return result

    def stats(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return dict(self._validation["stats"])

    def graph_data(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return {
            "graph_bundle_sha256": self.graph_bundle_sha256,
            "nodes": [
                {**dict(attrs), "id": str(node)}
                for node, attrs in sorted(self._graph._graph.nodes(data=True), key=lambda item: str(item[0]))
            ],
            "links": [
                {**attrs, "source": source, "target": target, "key": key}
                for source, target, key, attrs in self._graph.all_keyed_edges()
            ],
        }

    def retrieve(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        construction = self._validation["manifest"]["resolved_construction_config"]
        return retrieve(
            question,
            self._query_embedder(),
            self._kv_chunks,
            self._vdb_entity,
            self._vdb_hyperedge,
            self._graph,
            top_k=top_k,
            rrf_k=int(construction["compat_rrf_k"]),
            channel_candidate_multiplier=int(
                construction["compat_rrf_channel_candidate_multiplier"]
            ),
        )

    def query(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.retrieve(question, top_k=top_k)

    def search_entity_vectors(
        self,
        query_vector: List[float],
        *,
        top_k: int = 5,
        candidate_node_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:


        self._ensure_loaded()
        hits = self._vdb_entity.search(
            query_vector, top_k=top_k, allowed_node_ids=candidate_node_ids
        )
        return [
            {
                "node_id": hit["node_id"],
                "similarity": float(hit["__metrics__"]),
                "canonical_entity_key": hit.get("canonical_entity_key", ""),
                "embedding_signature": self.embedding_signature,
            }
            for hit in hits
        ]


def standalone_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="kg_builder", description="Standalone CHAIN compatibility graph builder")
    parser.add_argument("--ingest", metavar="PATH")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--kb-dir", default="datasets/KG/chain")
    args = parser.parse_args(argv)
    if not args.ingest and not args.stats:
        parser.print_help()
        return 1
    if args.stats and not args.ingest:
        result = load_graph_bundle_portable(args.kb_dir)
        print(json.dumps(result["stats"], ensure_ascii=False, sort_keys=True))
        return 0
    from chain.config import resolve_config

    config = resolve_config("compat")
    result = build_graph_bundle(
        args.ingest,
        args.kb_dir,
        resolved_config=config,
        profile="compat",
        allow_compat_adapters=True,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.stats:
        validated = load_graph_bundle_portable(args.kb_dir)
        print(json.dumps(validated["stats"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        raise SystemExit(standalone_main())
    except (GraphBuildError, GraphBundleValidationError, OSError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1)


__all__ = [
    "ArtifactCorruptionError",
    "Embedder",
    "ExtractionSchemaError",
    "GraphBuildError",
    "GraphStorage",
    "InputRecordError",
    "JsonKVStorage",
    "KGEngine",
    "LLMExtractor",
    "VectorStorage",
    "build_from_file",
    "build_from_folder",
    "build_graph_bundle",
    "canonical_entity_key",
    "canonical_proposition_key",
    "causal_edge_id",
    "chunk_text",
    "construction_payload_for_record",
    "entity_node_id",
    "extract_chunks",
    "extraction_prompt",
    "hyperedge_node_id",
    "incidence_edge_id",
    "ingest_file",
    "is_supported",
    "load_source_records",
    "normalize_source_descriptor",
    "parse_source_record",
    "record_to_chunks",
    "records_to_chunks",
    "retrieve",
    "standalone_main",
    "validate_extraction_payload",
]
