from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import posixpath
import random
import shutil
import sys
import tempfile
import time
import unicodedata
import uuid
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence


ROOT = Path(".")
sys.path.insert(0, str(ROOT))

EVAL_RUN_SCHEMA_V1 = "chain-eval-run-v1"
EVAL_CASE_SCHEMA_V1 = "chain-eval-case-v1"
EVAL_RUN_SCHEMA = "chain-eval-run-v2"
EVAL_CASE_SCHEMA = "chain-eval-case-v2"
CAUSAL_SUMMARY_SCHEMA = "chain-causal-summary-v1"
INPUT_ADAPTATION_SCHEMA = "chain-eval-input-adaptation-v1"
SOURCE_RECORD_COMPLETENESS_SCHEMA = "chain-source-record-completeness-v1"
PREDICTION_TRACE_ARTIFACT_SCHEMA = "chain-prediction-trace-artifact-v1"
ARTIFACT_BUNDLE_SCHEMA = "chain-eval-artifact-bundle-v1"
EVAL_OPERATIONAL_RECEIPT_SCHEMA = "chain-eval-operational-receipt-v1"
PROVIDER_STAGE_USAGE_SCHEMA = "chain-provider-stage-usage-v1"
METRIC_PROTOCOL_SCHEMA = "chain-metric-protocol-v1"
DEFAULT_ECE_BINS = 10
DEFAULT_ACE_BINS = 10
DEFAULT_RELIABILITY_BINS = 10
DEFAULT_MCE_MIN_BIN_SIZE = 5
DEFAULT_NLL_EPSILON = 1e-7
DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_BOOTSTRAP_SEED = 42



DEFAULT_RUNTIME_WORKERS = 16
_TRACE_GZIP_LEVEL = 6
_TRACE_MAX_CANONICAL_BYTES = 1024 * 1024 * 1024
_TRACE_STREAM_BLOCK_BYTES = 1024 * 1024
_TRACE_STREAM_TEXT_CHARS = 256 * 1024
_SHA256_HEX_LENGTH = 64
_PAPER_PROFILE_ID = "paper"
_COMPAT_PROFILE_ID = "compat"




LEGACY_DEFAULT_CUTOFFS = {
    "clinical_trial_ood.jsonl": "03/10/2026",
}




_FUSION_PARAMETER_FIELDS = frozenset(
    {
        "k_sat",
        "beta_0",
        "Z",
        "alpha_b",
        "alpha_0",
        "Omega_0",
        "zeta",
        "fusion_mode",
        "fixed_alpha",
        "ablation_mode",
    }
)
_RUN_IDENTITY_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "profile_id",
        "construction_config_sha256",
        "scientific_config_sha256",
        "graph_bundle_sha256",
        "test_file_sha256",
        "ordered_case_inputs_sha256",
        "case_count",
        "nonpublishable_reasons",
        "fusion_parameters",
    }
)
_LEGACY_RUN_IDENTITY_FIELDS_V1 = _RUN_IDENTITY_FIELDS_V1 - {"fusion_parameters"}
_RUN_IDENTITY_FIELDS = _RUN_IDENTITY_FIELDS_V1 | {"metric_protocol"}
_CANONICAL_ROW_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "run_identity",
        "run_identity_sha256",
        "case_input_sha256",
        "case_id",
        "id",
        "question",
        "cutoff",
        "binary_axis",
        "binary_axis_sha256",
        "gold_event",
        "ground_truth",
        "predicted_event",
        "pred_answer",
        "confidence",
        "p_llm_event",
        "p_causal_event",
        "p_final_event",
        "p_event",
        "correct",
        "converged",
        "domain",
        "source",
        "profile_id",
        "construction_config_sha256",
        "scientific_config_sha256",
        "graph_bundle_sha256",
        "publishable",
        "collect_rounds",
        "collect_stats",
        "causal_summary",
        "usage",
        "runtime_telemetry",
    }
)
_CANONICAL_ROW_FIELDS = frozenset(
    {
        *_CANONICAL_ROW_FIELDS_V1,
        "source_record",
        "source_record_sha256",
        "source_record_completeness",
        "input_adaptation",
        "prediction_trace_artifact",
    }
)
_SOURCE_RECORD_COMPLETENESS_FIELDS = frozenset(
    {
        "schema_version",
        "schema_fields",
        "schema_fields_sha256",
        "present_fields",
        "missing_fields",
        "completed_record",
        "completed_record_sha256",
    }
)



_SOURCE_RECORD_STANDARD_FIELDS = frozenset(
    {
        "id",
        "question",
        "cutoff",
        "date",
        "binary_axis",
        "gold_event",
        "ground_truth",
        "answer",
        "label",
        "target",
        "domain",
        "source",
        "background",
        "difficulty",
    }
)
_TRACE_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "relative_path",
        "compression",
        "canonical_json_sha256",
        "canonical_json_nbytes",
        "compressed_sha256",
        "compressed_nbytes",
    }
)
LEGACY_EVAL_BASENAMES = frozenset(
    {
        "ai_futures.jsonl",
        "polymarket.jsonl",
        "metaculus_bin.jsonl",
        "future_as_label.jsonl",
        "clinical_trial_ood.jsonl",
        "forecast_ood.jsonl",
        "golf_forecasting_ood.jsonl",
        "kalshi_ood.jsonl",
    }
)


class EvaluationError(RuntimeError):
    pass


class _CaseEvaluationFailure(RuntimeError):


    def __init__(
        self,
        *,
        case_id: str,
        original: BaseException,
        operational: Mapping[str, Any],
    ) -> None:
        self.case_id = case_id
        self.original_type = type(original).__name__
        self.original_message = " ".join(str(original).split())[:1000]
        self.operational = copy.deepcopy(dict(operational))
        super().__init__(f"{self.original_type}: {self.original_message}")


def _canonical_json(value: Any) -> str:


    from chain.config import canonical_json

    return canonical_json(value)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationError(f"cannot read {path}") from exc
    return digest.hexdigest()


def _strict_pairs(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_line(raw: str, *, context: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvaluationError(f"non-finite JSON constant {token!r} in {context}")
            ),
        )
    except EvaluationError:
        raise
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON in {context}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{context} must contain a JSON object")
    return value


def _canonicalize_input_object(
    value: Mapping[str, Any], *, context: str
) -> Dict[str, Any]:


    try:
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{context} is not canonical JSON: {exc}") from exc
    return _strict_json_line(canonical, context=f"{context} canonical form")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"JSONL file does not exist: {path}")
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if raw.strip():
                    rows.append(_strict_json_line(raw, context=f"{path.name}:{line_number}"))
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"JSONL is not UTF-8: {path}") from exc
    if not rows:
        raise EvaluationError(f"JSONL file is empty: {path}")
    return rows


def _source_schema_fields(records: Sequence[Mapping[str, Any]]) -> List[str]:
    fields = set(_SOURCE_RECORD_STANDARD_FIELDS)
    for record in records:
        fields.update(record)
    return sorted(fields)


def _build_source_record_completeness(
    source_record: Mapping[str, Any], *, schema_fields: Sequence[str]
) -> Dict[str, Any]:
    canonical_fields = sorted(schema_fields)
    if (
        not canonical_fields
        or len(set(canonical_fields)) != len(canonical_fields)
        or any(not isinstance(field, str) or not field for field in canonical_fields)
    ):
        raise EvaluationError("source record schema fields are invalid")
    present_fields = sorted(source_record)
    if not set(present_fields).issubset(canonical_fields):
        raise EvaluationError("source record contains a field outside its sealed schema")
    missing_fields = [
        field for field in canonical_fields if field not in source_record
    ]
    completed_record = {
        field: copy.deepcopy(source_record[field]) if field in source_record else None
        for field in canonical_fields
    }
    return {
        "schema_version": SOURCE_RECORD_COMPLETENESS_SCHEMA,
        "schema_fields": canonical_fields,
        "schema_fields_sha256": _sha256_payload(canonical_fields),
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "completed_record": completed_record,
        "completed_record_sha256": _sha256_payload(completed_record),
    }


def _validate_source_record_completeness(
    value: Any,
    *,
    source_record: Mapping[str, Any],
    context: str,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_RECORD_COMPLETENESS_FIELDS:
        raise EvaluationError(f"{context} source-record completeness schema mismatch")
    completeness = copy.deepcopy(dict(value))
    if completeness.get("schema_version") != SOURCE_RECORD_COMPLETENESS_SCHEMA:
        raise EvaluationError(f"{context} source-record completeness version mismatch")
    schema_fields = completeness.get("schema_fields")
    present_fields = completeness.get("present_fields")
    missing_fields = completeness.get("missing_fields")
    if (
        not isinstance(schema_fields, list)
        or not schema_fields
        or schema_fields != sorted(schema_fields)
        or len(set(schema_fields)) != len(schema_fields)
        or any(not isinstance(field, str) or not field for field in schema_fields)
        or not _SOURCE_RECORD_STANDARD_FIELDS.issubset(schema_fields)
    ):
        raise EvaluationError(f"{context} source-record schema fields are invalid")
    if (
        not isinstance(present_fields, list)
        or present_fields != sorted(present_fields)
        or len(set(present_fields)) != len(present_fields)
        or present_fields != sorted(source_record)
    ):
        raise EvaluationError(f"{context} source-record present fields mismatch")
    expected_missing = [field for field in schema_fields if field not in source_record]
    if missing_fields != expected_missing:
        raise EvaluationError(f"{context} source-record missing fields mismatch")
    if set(present_fields).union(expected_missing) != set(schema_fields):
        raise EvaluationError(f"{context} source-record field partition mismatch")
    completed_record = completeness.get("completed_record")
    expected_completed = {
        field: copy.deepcopy(source_record[field]) if field in source_record else None
        for field in schema_fields
    }
    if completed_record != expected_completed:
        raise EvaluationError(f"{context} completed source record mismatch")
    schema_hash = _strict_sha256(
        completeness.get("schema_fields_sha256"),
        f"{context}.source_record_completeness.schema_fields_sha256",
    )
    if schema_hash != _sha256_payload(schema_fields):
        raise EvaluationError(f"{context} source-record schema identity mismatch")
    completed_hash = _strict_sha256(
        completeness.get("completed_record_sha256"),
        f"{context}.source_record_completeness.completed_record_sha256",
    )
    if completed_hash != _sha256_payload(expected_completed):
        raise EvaluationError(f"{context} completed source-record identity mismatch")
    return completeness


def _case_input_payload(case: Mapping[str, Any]) -> Dict[str, Any]:


    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "cutoff": case["cutoff"],
        "binary_axis_sha256": case["binary_axis_sha256"],
        "gold_event": case["gold_event"],
        "domain": case["domain"],
        "source": case["source"],
        "source_record": case["source_record"],
        "source_record_sha256": case["source_record_sha256"],
        "source_record_completeness": case["source_record_completeness"],
        "input_adaptation": case["input_adaptation"],
    }


def _validate_input_adaptation(
    value: Any,
    *,
    source_record: Mapping[str, Any],
    cutoff: str,
    binary_axis: Mapping[str, Any],
    context: str,
) -> Dict[str, Any]:
    root_fields = {"schema_version", "cutoff", "binary_axis"}
    cutoff_fields = {
        "kind", "source_field", "raw_value", "canonical_value", "adapter_id"
    }
    axis_fields = {"kind", "source_field", "adapter_id"}
    if not isinstance(value, Mapping) or set(value) != root_fields:
        raise EvaluationError(f"{context} input_adaptation schema mismatch")
    if value.get("schema_version") != INPUT_ADAPTATION_SCHEMA:
        raise EvaluationError(f"{context} input_adaptation version mismatch")
    cutoff_trace = value.get("cutoff")
    axis_trace = value.get("binary_axis")
    if not isinstance(cutoff_trace, Mapping) or set(cutoff_trace) != cutoff_fields:
        raise EvaluationError(f"{context} cutoff adaptation schema mismatch")
    if not isinstance(axis_trace, Mapping) or set(axis_trace) != axis_fields:
        raise EvaluationError(f"{context} binary-axis adaptation schema mismatch")
    if cutoff_trace.get("canonical_value") != cutoff:
        raise EvaluationError(f"{context} cutoff adaptation value mismatch")
    raw_value = cutoff_trace.get("raw_value")
    if not isinstance(raw_value, str):
        raise EvaluationError(f"{context} cutoff adaptation raw value must be a string")
    cutoff_kind = cutoff_trace.get("kind")
    expected_cutoff_fields = {
        "explicit_cutoff": ("cutoff", ""),
        "legacy_date_alias": ("date", "chain-legacy-date-cutoff-v1"),
        "global_cutoff_fallback": ("--cutoff", "chain-compat-global-cutoff-v1"),
        "legacy_default_cutoff": (
            "registered_default",
            "chain-legacy-clinical-default-cutoff-v1",
        ),
    }
    if cutoff_kind not in expected_cutoff_fields:
        raise EvaluationError(f"{context} cutoff adaptation kind is invalid")
    source_field, adapter_id = expected_cutoff_fields[str(cutoff_kind)]
    if (
        cutoff_trace.get("source_field") != source_field
        or cutoff_trace.get("adapter_id") != adapter_id
    ):
        raise EvaluationError(f"{context} cutoff adaptation identity mismatch")
    if cutoff_kind == "explicit_cutoff":
        if source_record.get("cutoff") != raw_value:
            raise EvaluationError(f"{context} explicit cutoff source mismatch")
    elif cutoff_kind == "legacy_date_alias":
        if (
            source_record.get("date") != raw_value
            or source_record.get("cutoff") not in (None, "")
        ):
            raise EvaluationError(f"{context} legacy date source mismatch")
    elif (
        source_record.get("cutoff") not in (None, "")
        or source_record.get("date") not in (None, "")
    ):
        raise EvaluationError(f"{context} fallback cutoff source mismatch")
    from chain.skills.temporal_validity import parse_cutoff

    try:
        if parse_cutoff(raw_value).canonical != cutoff:
            raise EvaluationError(f"{context} cutoff adaptation canonicalization mismatch")
    except ValueError as exc:
        raise EvaluationError(f"{context} cutoff adaptation raw value is invalid") from exc

    axis_kind = axis_trace.get("kind")
    if axis_kind == "explicit_binary_axis":
        if axis_trace.get("source_field") != "binary_axis":
            raise EvaluationError(f"{context} explicit binary-axis source mismatch")
        if source_record.get("binary_axis") is None:
            raise EvaluationError(f"{context} source record omitted explicit binary_axis")
        from chain.config import validate_binary_axis

        try:
            source_axis = validate_binary_axis(source_record["binary_axis"])
        except Exception as exc:
            raise EvaluationError(
                f"{context} source record explicit binary_axis is invalid"
            ) from exc
        if source_axis != binary_axis:
            raise EvaluationError(f"{context} explicit binary-axis payload mismatch")
        if axis_trace.get("adapter_id") != str(binary_axis.get("adapter_id", "")):
            raise EvaluationError(f"{context} explicit binary-axis adapter mismatch")
    elif axis_kind == "legacy_yes_no_adapter":
        if (
            axis_trace.get("source_field") != "registered_default"
            or axis_trace.get("adapter_id") != "chain-legacy-native-yes-no-v1"
            or "binary_axis" in source_record
            or binary_axis.get("adapter_id") != "chain-legacy-native-yes-no-v1"
        ):
            raise EvaluationError(f"{context} legacy binary-axis adaptation mismatch")
    else:
        raise EvaluationError(f"{context} binary-axis adaptation kind is invalid")
    return copy.deepcopy(dict(value))


def _trace_relative_path(
    *, bundle_name: str, case_index: int, case_id: str
) -> str:
    case_digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24]
    return f"{bundle_name}/traces/{case_index:06d}-{case_digest}.json.gz"


def _iter_canonical_json_text(value: Any) -> Iterator[str]:


    def emit(item: Any) -> Iterator[str]:
        if item is None:
            yield "null"
            return
        if item is True:
            yield "true"
            return
        if item is False:
            yield "false"
            return
        if isinstance(item, str):
            yield json.encoder.encode_basestring(unicodedata.normalize("NFC", item))
            return
        if isinstance(item, int):


            yield int.__repr__(item)
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("non-finite numbers are not canonical JSON values")
            yield float.__repr__(item)
            return
        if isinstance(item, Mapping):
            normalised_items: List[tuple[str, Any]] = []
            seen_keys: set[str] = set()
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError("canonical JSON object keys must be strings")
                normalised_key = unicodedata.normalize("NFC", key)
                if normalised_key in seen_keys:
                    raise ValueError(
                        f"duplicate canonical JSON key: {normalised_key!r}"
                    )
                seen_keys.add(normalised_key)
                normalised_items.append((normalised_key, nested))
            normalised_items.sort(key=lambda pair: pair[0])
            yield "{"
            for index, (key, nested) in enumerate(normalised_items):
                if index:
                    yield ","
                yield json.encoder.encode_basestring(key)
                yield ":"
                yield from emit(nested)
            yield "}"
            return
        if isinstance(item, (list, tuple)):
            yield "["
            for index, nested in enumerate(item):
                if index:
                    yield ","
                yield from emit(nested)
            yield "]"
            return
        raise TypeError(f"unsupported canonical JSON value: {type(item).__name__}")

    yield from emit(value)


def _iter_canonical_json_bytes(value: Any) -> Iterator[bytes]:


    pending: List[str] = []
    pending_chars = 0

    def encoded_blocks() -> Iterator[bytes]:
        nonlocal pending_chars
        if not pending:
            return
        encoded = "".join(pending).encode("utf-8")
        pending.clear()
        pending_chars = 0
        for offset in range(0, len(encoded), _TRACE_STREAM_BLOCK_BYTES):
            yield encoded[offset : offset + _TRACE_STREAM_BLOCK_BYTES]

    for text in _iter_canonical_json_text(value):
        offset = 0
        while offset < len(text):
            room = _TRACE_STREAM_TEXT_CHARS - pending_chars
            take = min(room, len(text) - offset)
            pending.append(text[offset : offset + take])
            pending_chars += take
            offset += take
            if pending_chars == _TRACE_STREAM_TEXT_CHARS:
                yield from encoded_blocks()
    if pending:
        yield from encoded_blocks()


def _write_prediction_trace(
    prediction: Mapping[str, Any],
    *,
    staging_bundle: Path,
    bundle_name: str,
    case_index: int,
    case_id: str,
) -> Dict[str, Any]:
    if not isinstance(prediction, Mapping):
        raise EvaluationError("prediction trace must be an object")
    relative_path = _trace_relative_path(
        bundle_name=bundle_name, case_index=case_index, case_id=case_id
    )
    trace_path = staging_bundle.joinpath(*relative_path.split("/")[1:])
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_digest = hashlib.sha256()
    canonical_nbytes = 0
    try:
        with trace_path.open("xb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                compresslevel=_TRACE_GZIP_LEVEL,
                mtime=0,
            ) as compressed_handle:
                for block in _iter_canonical_json_bytes(prediction):
                    canonical_nbytes += len(block)
                    if canonical_nbytes > _TRACE_MAX_CANONICAL_BYTES:
                        raise EvaluationError(
                            "prediction trace exceeds the 1 GiB per-case safety limit"
                        )
                    canonical_digest.update(block)
                    compressed_handle.write(block)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
    except Exception:
        try:
            trace_path.unlink()
        except OSError:
            pass
        raise
    compressed_nbytes = trace_path.stat().st_size
    return {
        "schema_version": PREDICTION_TRACE_ARTIFACT_SCHEMA,
        "relative_path": relative_path,
        "compression": "gzip",
        "canonical_json_sha256": canonical_digest.hexdigest(),
        "canonical_json_nbytes": canonical_nbytes,
        "compressed_sha256": _sha256_file(trace_path),
        "compressed_nbytes": compressed_nbytes,
    }


def _link_prediction_trace(
    artifact: Mapping[str, Any],
    *,
    source_bundle: Optional[Path],
    destination_bundle: Path,
    destination_bundle_name: str,
    result_path: Path,
    case_index: int,
    case_id: str,
) -> Dict[str, Any]:


    source_artifact, source_path = _validated_trace_path(
        artifact,
        result_path=result_path,
        artifact_root_override=source_bundle,
        context=f"trace link source {case_id}",
    )
    if not source_path.is_file() or source_path.is_symlink():
        raise EvaluationError(f"trace link source is missing or unsafe: {case_id}")
    relative_path = _trace_relative_path(
        bundle_name=destination_bundle_name,
        case_index=case_index,
        case_id=case_id,
    )
    destination_path = destination_bundle.joinpath(*relative_path.split("/")[1:])
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:


        os.link(source_path, destination_path)
    except OSError as link_error:
        if destination_path.exists():
            raise EvaluationError(
                f"trace link destination already exists: {case_id}"
            ) from link_error
        try:
            with source_path.open("rb") as source_handle, destination_path.open(
                "xb"
            ) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise
    else:


        _fsync_directory(destination_path.parent)
    return {
        **source_artifact,
        "relative_path": relative_path,
    }


def _validated_trace_path(
    value: Any,
    *,
    result_path: Path,
    artifact_root_override: Optional[Path],
    context: str,
) -> tuple[Dict[str, Any], Path]:
    if not isinstance(value, Mapping) or set(value) != _TRACE_ARTIFACT_FIELDS:
        raise EvaluationError(f"{context} prediction trace artifact schema mismatch")
    artifact = dict(value)
    if artifact.get("schema_version") != PREDICTION_TRACE_ARTIFACT_SCHEMA:
        raise EvaluationError(f"{context} prediction trace artifact version mismatch")
    if artifact.get("compression") != "gzip":
        raise EvaluationError(f"{context} prediction trace compression mismatch")
    relative = artifact.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise EvaluationError(f"{context} prediction trace relative path is invalid")
    if relative.startswith("/") or posixpath.normpath(relative) != relative:
        raise EvaluationError(f"{context} prediction trace relative path is unsafe")
    parts = relative.split("/")
    if (
        len(parts) != 3
        or not parts[0].startswith(f"{result_path.name}.artifacts-")
        or parts[1] != "traces"
        or not parts[2].endswith(".json.gz")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise EvaluationError(f"{context} prediction trace relative path is invalid")
    if artifact_root_override is None:
        base = result_path.parent.resolve()
        bundle_candidate = result_path.parent / parts[0]
        if bundle_candidate.is_symlink():
            raise EvaluationError(
                f"{context} prediction trace artifact bundle is unsafe"
            )
        bundle_root = bundle_candidate.resolve()
        trace_path = (result_path.parent / relative).resolve()
        if trace_path.parent.parent != bundle_root or bundle_root.parent != base:
            raise EvaluationError(f"{context} prediction trace escaped its artifact bundle")
    else:
        if artifact_root_override.is_symlink():
            raise EvaluationError(
                f"{context} staged prediction trace artifact bundle is unsafe"
            )
        bundle_root = artifact_root_override.resolve()
        trace_path = artifact_root_override.joinpath(*parts[1:]).resolve()
        if trace_path.parent.parent != bundle_root:
            raise EvaluationError(f"{context} staged prediction trace escaped its bundle")
    for field in ("canonical_json_sha256", "compressed_sha256"):
        _strict_sha256(artifact.get(field), f"{context}.{field}")
    canonical_nbytes = _strict_nonnegative_int(
        artifact.get("canonical_json_nbytes"), f"{context}.canonical_json_nbytes"
    )
    compressed_nbytes = _strict_nonnegative_int(
        artifact.get("compressed_nbytes"), f"{context}.compressed_nbytes"
    )
    if canonical_nbytes <= 0 or canonical_nbytes > _TRACE_MAX_CANONICAL_BYTES:
        raise EvaluationError(f"{context} prediction trace canonical size is invalid")
    if compressed_nbytes <= 0:
        raise EvaluationError(f"{context} prediction trace compressed size is invalid")
    return artifact, trace_path


def _load_prediction_trace(
    value: Any,
    *,
    result_path: Path,
    artifact_root_override: Optional[Path] = None,
    context: str,
) -> Dict[str, Any]:
    artifact, trace_path = _validated_trace_path(
        value,
        result_path=result_path,
        artifact_root_override=artifact_root_override,
        context=context,
    )
    if not trace_path.is_file() or trace_path.is_symlink():
        raise EvaluationError(f"{context} prediction trace sidecar is missing or unsafe")
    if trace_path.stat().st_size != artifact["compressed_nbytes"]:
        raise EvaluationError(f"{context} prediction trace compressed size mismatch")
    if _sha256_file(trace_path) != artifact["compressed_sha256"]:
        raise EvaluationError(f"{context} prediction trace compressed hash mismatch")
    chunks: List[bytes] = []
    digest = hashlib.sha256()
    observed = 0
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        with trace_path.open("rb") as handle:
            header = handle.read(10)
            if (
                len(header) != 10
                or header[:3] != b"\x1f\x8b\x08"
                or header[3] != 0
                or header[4:8] != b"\x00\x00\x00\x00"
            ):
                raise EvaluationError(
                    f"{context} prediction trace gzip header is not deterministic"
                )
            compressed_block = header
            while compressed_block:
                if decompressor.eof:
                    raise EvaluationError(
                        f"{context} prediction trace must contain exactly one gzip "
                        "member with no trailing data"
                    )
                block = decompressor.decompress(compressed_block)
                if decompressor.unused_data:
                    raise EvaluationError(
                        f"{context} prediction trace must contain exactly one gzip "
                        "member with no trailing data"
                    )
                if block:
                    observed += len(block)
                    if (
                        observed > artifact["canonical_json_nbytes"]
                        or observed > _TRACE_MAX_CANONICAL_BYTES
                    ):
                        raise EvaluationError(
                            f"{context} prediction trace expands beyond its sealed size"
                        )
                    digest.update(block)
                    chunks.append(block)
                compressed_block = handle.read(64 * 1024)
    except (OSError, EOFError, zlib.error) as exc:
        raise EvaluationError(f"{context} prediction trace gzip stream is invalid") from exc
    if not decompressor.eof:
        raise EvaluationError(f"{context} prediction trace gzip stream is invalid")
    raw = b"".join(chunks)
    if observed != artifact["canonical_json_nbytes"]:
        raise EvaluationError(f"{context} prediction trace canonical size mismatch")
    if digest.hexdigest() != artifact["canonical_json_sha256"]:
        raise EvaluationError(f"{context} prediction trace canonical hash mismatch")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"{context} prediction trace is not UTF-8") from exc
    prediction = _strict_json_line(decoded, context=f"{context} prediction trace")
    if _canonical_json(prediction).encode("utf-8") != raw:
        raise EvaluationError(f"{context} prediction trace is not canonical JSON")
    return prediction


def _validate_prediction_trace_artifact_metadata(
    value: Any,
    *,
    result_path: Path,
    artifact_root_override: Optional[Path] = None,
    context: str,
) -> Dict[str, Any]:


    artifact, trace_path = _validated_trace_path(
        value,
        result_path=result_path,
        artifact_root_override=artifact_root_override,
        context=context,
    )
    if not trace_path.is_file() or trace_path.is_symlink():
        raise EvaluationError(f"{context} prediction trace sidecar is missing or unsafe")
    if trace_path.stat().st_size != artifact["compressed_nbytes"]:
        raise EvaluationError(f"{context} prediction trace compressed size mismatch")
    return artifact


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field} must be a JSON number, not {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise EvaluationError(f"{field} must be finite and in [0,1]")
    return parsed


def _legacy_probability(row: Mapping[str, Any]) -> float:
    answer = str(row.get("pred_answer", "")).strip().casefold()
    confidence = _finite_probability(row.get("confidence"), "confidence")
    if answer == "yes":
        return confidence
    if answer == "no":
        return 1.0 - confidence




    return 0.5


def _p_event(row: Mapping[str, Any], *, allow_legacy: bool = False) -> float:
    if "p_event" not in row:
        if allow_legacy:
            return _legacy_probability(row)
        raise EvaluationError("canonical metrics require p_event")
    probability = _finite_probability(row["p_event"], "p_event")
    if "p_final_event" not in row:
        raise EvaluationError("canonical row requires p_final_event")
    p_final = _finite_probability(row["p_final_event"], "p_final_event")
    if probability != p_final:
        raise EvaluationError("p_event must exactly equal p_final_event")
    return probability


def _gold_event(row: Mapping[str, Any], *, allow_legacy: bool = False) -> int:
    value = row.get("gold_event")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if allow_legacy:
        legacy = str(row.get("ground_truth", "")).strip().casefold()
        if legacy in {"yes", "true", "1"}:
            return 1
        if legacy in {"no", "false", "0"}:
            return 0
    raise EvaluationError("canonical metrics require gold_event in {0,1}")


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{field} must be a non-negative integer")
    return value


def _strict_nonnegative_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field} must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise EvaluationError(f"{field} must be finite and non-negative")
    return parsed


def _strict_identity(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{field} must be a non-empty string")
    return value


def _strict_sha256(value: Any, field: str) -> str:
    parsed = _strict_identity(value, field)
    if len(parsed) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in parsed
    ):
        raise EvaluationError(f"{field} must be a lowercase SHA-256 hex digest")
    return parsed


def _fusion_parameters_from_config(config: Any) -> Dict[str, Any]:


    return {
        "k_sat": int(getattr(config, "k_sat")),
        "beta_0": float(getattr(config, "beta_0")),
        "Z": float(getattr(config, "Z")),
        "alpha_b": float(getattr(config, "alpha_b")),
        "alpha_0": float(getattr(config, "alpha_0")),
        "Omega_0": float(getattr(config, "Omega_0")),
        "zeta": float(getattr(config, "zeta")),
        "fusion_mode": str(getattr(config, "fusion_mode")),
        "fixed_alpha": (
            None
            if getattr(config, "fixed_alpha") is None
            else float(getattr(config, "fixed_alpha"))
        ),
        "ablation_mode": str(getattr(config, "ablation_mode", "none") or "none"),
    }


def _validate_fusion_parameters(value: Any, *, context: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FUSION_PARAMETER_FIELDS:
        raise EvaluationError(f"{context} fusion parameter schema mismatch")
    k_sat = value["k_sat"]
    if isinstance(k_sat, bool) or not isinstance(k_sat, int) or k_sat <= 0:
        raise EvaluationError(f"{context}.k_sat must be a positive integer")

    def finite_number(field: str) -> float:
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EvaluationError(f"{context}.{field} must be a JSON number")
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise EvaluationError(f"{context}.{field} must be finite")
        return parsed

    beta_0 = finite_number("beta_0")
    if not 0.0 < beta_0 < 1.0:
        raise EvaluationError(f"{context}.beta_0 must be in (0,1)")
    normalizer = finite_number("Z")
    if normalizer <= 0.0:
        raise EvaluationError(f"{context}.Z must be > 0")
    alpha_b = finite_number("alpha_b")
    if alpha_b <= 0.0:
        raise EvaluationError(f"{context}.alpha_b must be > 0")
    alpha_0 = finite_number("alpha_0")
    if not 0.0 < alpha_0 <= 1.0:
        raise EvaluationError(f"{context}.alpha_0 must be in (0,1]")
    omega_0 = finite_number("Omega_0")
    if not 0.0 <= omega_0 <= 1.0:
        raise EvaluationError(f"{context}.Omega_0 must be in [0,1]")
    zeta = finite_number("zeta")
    if zeta <= 0.0:
        raise EvaluationError(f"{context}.zeta must be > 0")

    fusion_mode = value["fusion_mode"]
    if fusion_mode not in {"adaptive", "fixed"}:
        raise EvaluationError(f"{context}.fusion_mode is invalid")
    fixed_alpha_raw = value["fixed_alpha"]
    fixed_alpha: Optional[float]
    if fixed_alpha_raw is None:
        fixed_alpha = None
    else:
        if isinstance(fixed_alpha_raw, bool) or not isinstance(
            fixed_alpha_raw, (int, float)
        ):
            raise EvaluationError(f"{context}.fixed_alpha must be null or a JSON number")
        fixed_alpha = float(fixed_alpha_raw)
        if not math.isfinite(fixed_alpha) or not 0.0 <= fixed_alpha <= 1.0:
            raise EvaluationError(f"{context}.fixed_alpha must be in [0,1]")
    if fusion_mode == "fixed" and fixed_alpha is None:
        raise EvaluationError(f"{context}.fixed_alpha is required for fixed fusion")
    if fusion_mode == "adaptive" and fixed_alpha is not None:
        raise EvaluationError(f"{context}.fixed_alpha is invalid for adaptive fusion")

    ablation_mode = value["ablation_mode"]
    if ablation_mode not in {
        "none",
        "without_ctvf",
        "without_noisy_or",
        "without_adaptive_alpha",
        "infer_tvf",
        "without_tvf",
        "without_all",
    }:
        raise EvaluationError(f"{context}.ablation_mode is invalid")
    return {
        "k_sat": k_sat,
        "beta_0": beta_0,
        "Z": normalizer,
        "alpha_b": alpha_b,
        "alpha_0": alpha_0,
        "Omega_0": omega_0,
        "zeta": zeta,
        "fusion_mode": fusion_mode,
        "fixed_alpha": fixed_alpha,
        "ablation_mode": ablation_mode,
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def _assert_close(expected: float, actual: Any, field: str) -> None:
    observed = _finite_probability(actual, field)
    if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
        raise EvaluationError(
            f"{field} does not close over the canonical Eq. (18)-(22) recomputation"
        )


def _validate_fusion_closure(
    *,
    probabilities: Mapping[str, Any],
    causal_summary: Mapping[str, Any],
    fusion_parameters: Mapping[str, Any],
    context: str,
) -> None:


    params = _validate_fusion_parameters(fusion_parameters, context=context)
    k = _strict_nonnegative_int(causal_summary.get("n_chains"), f"{context}.n_chains")
    chain_diversity = _strict_nonnegative_int(
        causal_summary.get("chain_diversity"), f"{context}.chain_diversity"
    )
    if chain_diversity != k:
        raise EvaluationError(f"{context}.chain_diversity is inconsistent")
    mean_score = _finite_probability(
        causal_summary.get("mean_chain_score"), f"{context}.mean_chain_score"
    )
    n_supporting = _strict_nonnegative_int(
        causal_summary.get("n_supporting"), f"{context}.n_supporting"
    )
    n_opposing = _strict_nonnegative_int(
        causal_summary.get("n_opposing"), f"{context}.n_opposing"
    )
    if n_supporting + n_opposing != k:
        raise EvaluationError(f"{context} polarity counts do not close over n_chains")
    avg_chain_ctvf = _finite_probability(
        causal_summary.get("avg_chain_ctvf"), f"{context}.avg_chain_ctvf"
    )
    if mean_score != avg_chain_ctvf:
        raise EvaluationError(f"{context} mean chain score fields disagree")
    if k == 0 and mean_score != 0.0:
        raise EvaluationError(f"{context} zero-chain summary must have zero mean score")
    status = causal_summary.get("causal_status")
    if k == 0 and status not in {"valid_no_chain", "valid_no_target"}:
        raise EvaluationError(f"{context} zero-chain status is inconsistent")
    if k > 0 and status != "valid":
        raise EvaluationError(f"{context} non-empty chain status is inconsistent")
    reliability = (
        0.0
        if k == 0
        else min(math.log1p(k) / math.log1p(params["k_sat"]), 1.0)
    )
    balance = min(n_supporting, n_opposing) / max(n_supporting, n_opposing, 1)
    floored_balance = params["beta_0"] + (1.0 - params["beta_0"]) * balance
    coverage = min(
        k * mean_score * reliability * floored_balance / params["Z"],
        1.0,
    )
    if k == 0:
        alpha = 0.0
    elif params["ablation_mode"] == "without_all":
        alpha = 0.0
    elif params["ablation_mode"] == "without_adaptive_alpha":


        alpha = 0.5
    elif params["fusion_mode"] == "adaptive":
        alpha = min(
            2.0
            * params["alpha_b"]
            * _sigmoid(params["zeta"] * (coverage - params["Omega_0"])),
            params["alpha_0"],
        )
    else:
        alpha = (
            params["alpha_b"]
            if params["fixed_alpha"] is None
            else params["fixed_alpha"]
        )
    p_llm = _finite_probability(probabilities.get("p_llm_event"), f"{context}.p_llm_event")
    p_causal = _finite_probability(
        probabilities.get("p_causal_event"), f"{context}.p_causal_event"
    )
    p_final = _finite_probability(
        probabilities.get("p_final_event"), f"{context}.p_final_event"
    )
    expected_final = alpha * p_causal + (1.0 - alpha) * p_llm
    _assert_close(reliability, causal_summary.get("reliability"), f"{context}.reliability")
    _assert_close(balance, causal_summary.get("balance"), f"{context}.balance")
    _assert_close(
        floored_balance,
        causal_summary.get("floored_balance"),
        f"{context}.floored_balance",
    )
    _assert_close(coverage, causal_summary.get("coverage"), f"{context}.coverage")
    _assert_close(alpha, causal_summary.get("alpha"), f"{context}.alpha")
    _assert_close(expected_final, p_final, f"{context}.p_final_event")


def _causal_summary_from_prediction(
    prediction: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    config: Any,
    graph_validation: Mapping[str, Any],
    probabilities: Mapping[str, float],
) -> Dict[str, Any]:
    causal = prediction.get("causal_inference")
    if not isinstance(causal, Mapping):
        raise EvaluationError("prediction omitted causal_inference audit payload")
    causal_prediction = causal.get("prediction")
    if not isinstance(causal_prediction, Mapping):
        raise EvaluationError("causal_inference omitted prediction audit payload")
    for field in ("p_llm_event", "p_causal_event", "p_final_event", "p_event"):
        value = _finite_probability(causal.get(field), f"causal_inference.{field}")
        if value != probabilities[field]:
            raise EvaluationError(f"causal_inference {field} disagrees with prediction")
    alpha = _finite_probability(causal.get("alpha"), "causal_inference.alpha")
    coverage = _finite_probability(causal_prediction.get("coverage"), "causal_inference.coverage")
    reliability = _finite_probability(
        causal_prediction.get("reliability"), "causal_inference.reliability"
    )
    balance = _finite_probability(causal_prediction.get("balance"), "causal_inference.balance")
    floored_balance = _finite_probability(
        causal_prediction.get("floored_balance"), "causal_inference.floored_balance"
    )
    mean_chain_score = _finite_probability(
        causal_prediction.get("mean_chain_score"), "causal_inference.mean_chain_score"
    )
    n_chains = _strict_nonnegative_int(causal.get("n_chains"), "causal_inference.n_chains")
    n_chains_raw = _strict_nonnegative_int(
        causal.get("n_chains_raw"), "causal_inference.n_chains_raw"
    )
    chain_diversity = _strict_nonnegative_int(
        causal.get("chain_diversity"), "causal_inference.chain_diversity"
    )
    n_supporting = _strict_nonnegative_int(
        causal_prediction.get("n_supporting"), "causal_inference.n_supporting"
    )
    n_opposing = _strict_nonnegative_int(
        causal_prediction.get("n_opposing"), "causal_inference.n_opposing"
    )
    if n_chains_raw < n_chains or chain_diversity != n_chains:
        raise EvaluationError("causal chain counts are inconsistent")
    status = causal.get("causal_status")
    if status not in {"valid", "valid_no_target", "valid_no_chain"}:
        raise EvaluationError("causal_inference has an invalid causal_status")
    top_chains = causal.get("top_chains")
    if not isinstance(top_chains, list) or len(top_chains) > n_chains:
        raise EvaluationError("causal_inference top_chains is inconsistent with n_chains")
    top_chain_summary = []
    for index, chain in enumerate(top_chains):
        if not isinstance(chain, Mapping):
            raise EvaluationError(f"causal_inference.top_chains[{index}] must be an object")
        path = chain.get("typed_path_identity")
        if not isinstance(path, list) or not path or any(
            not isinstance(item, str) or not item for item in path
        ):
            raise EvaluationError(f"causal_inference.top_chains[{index}] has invalid path identity")
        direction = chain.get("direction")
        if direction not in {"supports", "opposes"}:
            raise EvaluationError(f"causal_inference.top_chains[{index}] has invalid direction")
        top_chain_summary.append(
            {
                "typed_path_identity": list(path),
                "score": _finite_probability(chain.get("score"), f"causal_inference.top_chains[{index}].score"),
                "direction": direction,
                "target_id": _strict_identity(
                    chain.get("target_id"), f"causal_inference.top_chains[{index}].target_id"
                ),
            }
        )
    expected_identity = {
        "case_id": case["case_id"],
        "cutoff": case["cutoff"],
        "binary_axis_sha256": case["binary_axis_sha256"],
        "graph_bundle_sha256": graph_validation["graph_bundle_sha256"],
        "construction_config_sha256": config.construction_config_sha256,
        "scientific_config_sha256": config.scientific_config_sha256,
    }
    for field, expected in expected_identity.items():
        if causal.get(field) != expected:
            raise EvaluationError(f"causal_inference {field} identity mismatch")
    return {
        "schema_version": CAUSAL_SUMMARY_SCHEMA,
        "causal_status": status,
        "n_chains": n_chains,
        "n_chains_raw": n_chains_raw,
        "chain_diversity": chain_diversity,
        "avg_chain_ctvf": _finite_probability(
            causal.get("avg_chain_ctvf"), "causal_inference.avg_chain_ctvf"
        ),
        "p_causal_event": probabilities["p_causal_event"],
        "alpha": alpha,
        "coverage": coverage,
        "reliability": reliability,
        "balance": balance,
        "floored_balance": floored_balance,
        "mean_chain_score": mean_chain_score,
        "n_supporting": n_supporting,
        "n_opposing": n_opposing,
        "top_chains": top_chain_summary,
        "active_view_id": _strict_identity(causal.get("active_view_id"), "causal_inference.active_view_id"),
        "inference_identity": _strict_identity(
            causal.get("inference_identity"), "causal_inference.inference_identity"
        ),
        **expected_identity,
    }


def _validate_causal_summary(
    value: Any,
    *,
    case: Mapping[str, Any],
    run_payload: Mapping[str, Any],
    graph_bundle_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != CAUSAL_SUMMARY_SCHEMA:
        raise EvaluationError("canonical row causal_summary schema mismatch")
    required = {
        "schema_version", "causal_status", "n_chains", "n_chains_raw", "chain_diversity",
        "avg_chain_ctvf", "p_causal_event", "alpha", "coverage", "reliability", "balance",
        "floored_balance", "mean_chain_score", "n_supporting", "n_opposing", "top_chains",
        "active_view_id", "inference_identity", "case_id", "cutoff", "binary_axis_sha256",
        "graph_bundle_sha256", "construction_config_sha256", "scientific_config_sha256",
    }
    if set(value) != required:
        raise EvaluationError("canonical row causal_summary fields mismatch")
    if value["causal_status"] not in {"valid", "valid_no_target", "valid_no_chain"}:
        raise EvaluationError("canonical row causal_summary status is invalid")
    n_chains = _strict_nonnegative_int(value["n_chains"], "causal_summary.n_chains")
    n_chains_raw = _strict_nonnegative_int(value["n_chains_raw"], "causal_summary.n_chains_raw")
    if n_chains_raw < n_chains or value["chain_diversity"] != n_chains:
        raise EvaluationError("canonical row causal_summary chain counts are inconsistent")
    for field in (
        "avg_chain_ctvf", "p_causal_event", "alpha", "coverage", "reliability", "balance",
        "floored_balance", "mean_chain_score",
    ):
        _finite_probability(value[field], f"causal_summary.{field}")
    for field in ("n_supporting", "n_opposing"):
        _strict_nonnegative_int(value[field], f"causal_summary.{field}")
    top_chains = value["top_chains"]
    if not isinstance(top_chains, list) or len(top_chains) > n_chains:
        raise EvaluationError("canonical row causal_summary top_chains is invalid")
    for index, chain in enumerate(top_chains):
        if not isinstance(chain, Mapping) or set(chain) != {
            "typed_path_identity", "score", "direction", "target_id"
        }:
            raise EvaluationError(f"canonical row causal_summary top chain {index} is invalid")
        path = chain["typed_path_identity"]
        if not isinstance(path, list) or not path or any(
            not isinstance(item, str) or not item for item in path
        ):
            raise EvaluationError(f"canonical row causal_summary top chain {index} path is invalid")
        _finite_probability(chain["score"], f"causal_summary.top_chains[{index}].score")
        if chain["direction"] not in {"supports", "opposes"}:
            raise EvaluationError(f"canonical row causal_summary top chain {index} direction is invalid")
        _strict_identity(chain["target_id"], f"causal_summary.top_chains[{index}].target_id")
    expected = {
        "case_id": case["case_id"],
        "cutoff": case["cutoff"],
        "binary_axis_sha256": case["binary_axis_sha256"],
        "graph_bundle_sha256": graph_bundle_sha256,
        "construction_config_sha256": run_payload.get("construction_config_sha256", value["construction_config_sha256"]),
        "scientific_config_sha256": run_payload["scientific_config_sha256"],
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise EvaluationError(f"canonical row causal_summary {field} mismatch")
    _strict_identity(value["active_view_id"], "causal_summary.active_view_id")
    _strict_identity(value["inference_identity"], "causal_summary.inference_identity")
    return dict(value)


def _validate_usage(value: Any, *, context: str) -> Dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "prompt_tokens", "completion_tokens", "total_tokens", "api_calls"
    }:
        raise EvaluationError(f"{context} usage schema mismatch")
    return {
        field: _strict_nonnegative_int(value[field], f"{context}.{field}")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls")
    }


def _metric_rows(results: Sequence[Mapping[str, Any]], *, allow_legacy: bool = False) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in results:
        normalized.append(
            {
                "p_event": _p_event(row, allow_legacy=allow_legacy),
                "gold_event": _gold_event(row, allow_legacy=allow_legacy),
            }
        )
    return normalized


def accuracy(results: Sequence[Mapping[str, Any]]) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    return sum((row["p_event"] >= 0.5) == bool(row["gold_event"]) for row in results) / len(results)


def brier_score(results: Sequence[Mapping[str, Any]]) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    return sum((row["p_event"] - row["gold_event"]) ** 2 for row in results) / len(results)


def _validate_bin_count(n_bins: int) -> None:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise EvaluationError("n_bins must be a positive integer")


def _equal_width_bins(
    results: Sequence[Mapping[str, Any]], n_bins: int
) -> List[List[Mapping[str, Any]]]:


    _validate_bin_count(n_bins)
    bins: List[List[Mapping[str, Any]]] = [[] for _ in range(n_bins)]
    for row in results:
        bins[min(int(row["p_event"] * n_bins), n_bins - 1)].append(row)
    return bins


def _equal_mass_bins(
    results: Sequence[Mapping[str, Any]], n_bins: int
) -> List[List[Mapping[str, Any]]]:


    _validate_bin_count(n_bins)
    ranked = [
        row
        for _, row in sorted(
            enumerate(results), key=lambda indexed: (indexed[1]["p_event"], indexed[0])
        )
    ]
    quotient, remainder = divmod(len(ranked), n_bins)
    bins: List[List[Mapping[str, Any]]] = []
    start = 0
    for index in range(n_bins):
        size = quotient + (1 if index < remainder else 0)
        if size:
            bins.append(ranked[start : start + size])
        start += size
    return bins


def _calibration_gap(bucket: Sequence[Mapping[str, Any]]) -> float:
    mean_probability = sum(row["p_event"] for row in bucket) / len(bucket)
    event_rate = sum(row["gold_event"] for row in bucket) / len(bucket)
    return abs(mean_probability - event_rate)


def ece(
    results: Sequence[Mapping[str, Any]], n_bins: int = DEFAULT_ECE_BINS
) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    bins = _equal_width_bins(results, n_bins)
    total = len(results)
    return sum(
        (len(bucket) / total) * _calibration_gap(bucket)
        for bucket in bins
        if bucket
    )


def ace(
    results: Sequence[Mapping[str, Any]], n_bins: int = DEFAULT_ACE_BINS
) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    bins = _equal_mass_bins(results, n_bins)
    total = len(results)
    return sum((len(bucket) / total) * _calibration_gap(bucket) for bucket in bins)


def mce(
    results: Sequence[Mapping[str, Any]],
    n_bins: int = DEFAULT_ECE_BINS,
    min_bin_size: int = DEFAULT_MCE_MIN_BIN_SIZE,
) -> Optional[float]:
    if not results:
        raise EvaluationError("metrics require at least one result")
    if (
        isinstance(min_bin_size, bool)
        or not isinstance(min_bin_size, int)
        or min_bin_size <= 0
    ):
        raise EvaluationError("min_bin_size must be a positive integer")
    eligible = [
        _calibration_gap(bucket)
        for bucket in _equal_width_bins(results, n_bins)
        if len(bucket) >= min_bin_size
    ]

    return max(eligible) if eligible else None


def reliability(
    results: Sequence[Mapping[str, Any]], n_bins: int = DEFAULT_RELIABILITY_BINS
) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    bins = _equal_width_bins(results, n_bins)
    total = len(results)
    return sum(
        (len(bucket) / total) * _calibration_gap(bucket) ** 2
        for bucket in bins
        if bucket
    )


def auc_roc(results: Sequence[Mapping[str, Any]]) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    labels = [row["gold_event"] for row in results]
    if len(set(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    return float(roc_auc_score(labels, [row["p_event"] for row in results]))


def nll(
    results: Sequence[Mapping[str, Any]], epsilon: float = DEFAULT_NLL_EPSILON
) -> float:
    if not results:
        raise EvaluationError("metrics require at least one result")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or epsilon <= 0
        or epsilon >= 0.5
    ):
        raise EvaluationError("epsilon must be finite and in (0,0.5)")
    total = 0.0
    for row in results:
        probability = min(max(row["p_event"], epsilon), 1.0 - epsilon)
        gold = row["gold_event"]
        total += -(gold * math.log(probability) + (1 - gold) * math.log(1.0 - probability))
    return total / len(results)


def bootstrap_ci(
    results: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[Sequence[Mapping[str, Any]]], float],
    n_boot: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, float]:
    if not results:
        raise EvaluationError("bootstrap requires at least one result")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot <= 0:
        raise EvaluationError("n_boot must be a positive integer")
    rng = random.Random(seed)
    estimates = [
        float(metric_fn([results[rng.randrange(len(results))] for _ in results]))
        for _ in range(n_boot)
    ]
    estimates.sort()
    mean = sum(estimates) / len(estimates)
    variance = sum((estimate - mean) ** 2 for estimate in estimates) / len(estimates)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "ci_low": estimates[int(0.025 * (len(estimates) - 1))],
        "ci_high": estimates[int(0.975 * (len(estimates) - 1))],
    }


def compute_metrics(
    results: Sequence[Mapping[str, Any]],
    label: str = "",
    with_ci: bool = False,
    *,
    allow_legacy: bool = False,
    ece_bins: int = DEFAULT_ECE_BINS,
    ace_bins: int = DEFAULT_ACE_BINS,
    reliability_bins: int = DEFAULT_RELIABILITY_BINS,
    mce_min_bin_size: int = DEFAULT_MCE_MIN_BIN_SIZE,
    nll_epsilon: float = DEFAULT_NLL_EPSILON,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    rows = _metric_rows(results, allow_legacy=allow_legacy)
    auc = auc_roc(rows)
    maximum_error = mce(rows, ece_bins, mce_min_bin_size)
    output: Dict[str, Any] = {
        "label": label,
        "n": len(rows),
        "accuracy": accuracy(rows),
        "brier_score": brier_score(rows),
        "ece": ece(rows, ece_bins),
        "ace": ace(rows, ace_bins),
        "mce": maximum_error if maximum_error is not None else "N/A",
        "reliability": reliability(rows, reliability_bins),
        "auc_roc": auc if math.isfinite(auc) else "N/A",
        "nll": nll(rows, nll_epsilon),
        "pred_positive": sum(row["p_event"] >= 0.5 for row in rows),
        "pred_negative": sum(row["p_event"] < 0.5 for row in rows),
        "legacy_probability_adapter": allow_legacy and any("p_event" not in row for row in results),
    }
    if with_ci and len(rows) >= 10:
        output["accuracy_ci"] = bootstrap_ci(rows, accuracy, bootstrap_draws, bootstrap_seed)
        output["brier_ci"] = bootstrap_ci(rows, brier_score, bootstrap_draws, bootstrap_seed)
        output["ece_ci"] = bootstrap_ci(
            rows, lambda sample: ece(sample, ece_bins), bootstrap_draws, bootstrap_seed
        )
        output["nll_ci"] = bootstrap_ci(
            rows, lambda sample: nll(sample, nll_epsilon), bootstrap_draws, bootstrap_seed
        )
    return output


def _validate_prediction_trace_closure(
    row: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    run_payload: Mapping[str, Any],
    result_path: Path,
    artifact_root_override: Optional[Path] = None,
    context: str,
) -> Dict[str, Any]:
    prediction = _load_prediction_trace(
        row.get("prediction_trace_artifact"),
        result_path=result_path,
        artifact_root_override=artifact_root_override,
        context=context,
    )
    _validate_prediction_trace_completeness(prediction, context=context)
    identity_expectations = {
        "case_id": case["case_id"],
        "cutoff": case["cutoff"],
        "binary_axis_sha256": case["binary_axis_sha256"],
        "profile_id": run_payload["profile_id"],
        "scientific_config_sha256": run_payload["scientific_config_sha256"],
        "graph_bundle_sha256": run_payload["graph_bundle_sha256"],
    }
    for field, expected in identity_expectations.items():
        if prediction.get(field) != expected:
            raise EvaluationError(f"{context} prediction trace {field} mismatch")
    if prediction.get("binary_axis") != case["binary_axis"]:
        raise EvaluationError(f"{context} prediction trace binary_axis mismatch")
    probabilities = {}
    for field in ("p_llm_event", "p_causal_event", "p_final_event", "p_event"):
        probability = _finite_probability(
            prediction.get(field), f"{context}.prediction_trace.{field}"
        )
        if probability != row.get(field):
            raise EvaluationError(f"{context} prediction trace {field} mismatch")
        probabilities[field] = probability
    if probabilities["p_event"] != probabilities["p_final_event"]:
        raise EvaluationError(f"{context} prediction trace final probability mismatch")
    if prediction.get("answer") != row.get("pred_answer"):
        raise EvaluationError(f"{context} prediction trace answer mismatch")
    if _finite_probability(
        prediction.get("confidence"), f"{context}.prediction_trace.confidence"
    ) != row.get("confidence"):
        raise EvaluationError(f"{context} prediction trace confidence mismatch")
    trace_converged = prediction.get("converged")
    if not isinstance(trace_converged, bool):
        raise EvaluationError(f"{context} prediction trace converged must be boolean")
    if trace_converged is not row.get("converged"):
        raise EvaluationError(f"{context} prediction trace convergence mismatch")
    if prediction.get("collect_rounds") != row.get("collect_rounds"):
        raise EvaluationError(f"{context} prediction trace collect_rounds mismatch")
    if prediction.get("collect_stats") != row.get("collect_stats"):
        raise EvaluationError(f"{context} prediction trace collect_stats mismatch")
    trace_publishable = prediction.get("publishable")
    if not isinstance(trace_publishable, bool):
        raise EvaluationError(f"{context} prediction trace publishable must be boolean")
    expected_trace_publishable = run_payload.get("profile_id") == _PAPER_PROFILE_ID
    if trace_publishable is not expected_trace_publishable:
        raise EvaluationError(f"{context} prediction trace publishability mismatch")
    expected_row_publishable = bool(
        trace_publishable and not run_payload.get("nonpublishable_reasons")
    )
    if row.get("publishable") is not expected_row_publishable:
        raise EvaluationError(f"{context} row/trace publishability mismatch")
    _validate_provider_traces(
        prediction.get("provider_traces"),
        expected_api_calls=_strict_nonnegative_int(
            row.get("usage", {}).get("api_calls")
            if isinstance(row.get("usage"), Mapping)
            else None,
            f"{context}.usage.api_calls",
        ),
        context=context,
    )
    trace_config = SimpleNamespace(
        construction_config_sha256=run_payload["construction_config_sha256"],
        scientific_config_sha256=run_payload["scientific_config_sha256"],
    )
    trace_summary = _causal_summary_from_prediction(
        prediction,
        case=case,
        config=trace_config,
        graph_validation={"graph_bundle_sha256": run_payload["graph_bundle_sha256"]},
        probabilities=probabilities,
    )
    if trace_summary != row.get("causal_summary"):
        raise EvaluationError(f"{context} prediction trace causal summary mismatch")
    _validate_fusion_closure(
        probabilities=probabilities,
        causal_summary=trace_summary,
        fusion_parameters=run_payload.get("fusion_parameters"),
        context=f"{context} prediction trace",
    )
    return prediction


def _find_prohibited_prediction_field(
    value: Any,
    *,
    path: str = "prediction",
) -> Optional[str]:
    prohibited = {
        "gold_event",
        "ground_truth",
        "ground_truth_raw",
        "source_record",
        "background",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.casefold() in prohibited:
                return child_path
            found = _find_prohibited_prediction_field(item, path=child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_prohibited_prediction_field(
                item, path=f"{path}[{index}]"
            )
            if found is not None:
                return found
    return None


def _provider_call_attempt_count(value: Any, *, context: str) -> int:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{context} provider call trace must be an object")
    if value.get("trace_schema_version") != "chain-provider-trace-v1":
        raise EvaluationError(f"{context} provider call trace version mismatch")
    if value.get("status") != "ok":
        raise EvaluationError(f"{context} provider call trace is not successful")
    attempts = value.get("attempts", [])
    if not isinstance(attempts, list):
        raise EvaluationError(f"{context} provider call attempts must be a list")
    attempt_count = _strict_nonnegative_int(
        value.get("attempt_count"), f"{context}.attempt_count"
    )
    if attempt_count != len(attempts):
        raise EvaluationError(f"{context} provider call attempt count mismatch")
    if attempts and (
        not isinstance(attempts[-1], Mapping)
        or attempts[-1].get("status") != "ok"
    ):
        raise EvaluationError(f"{context} successful provider call lacks a final ok attempt")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or attempt.get("status") not in {
            "ok",
            "error",
        }:
            raise EvaluationError(f"{context} provider attempt {index} is invalid")
    return attempt_count


def _validate_provider_traces(
    value: Any,
    *,
    expected_api_calls: int,
    context: str,
) -> None:
    if not isinstance(value, list):
        raise EvaluationError(f"{context} prediction provider_traces must be a list")
    observed_api_calls = 0
    for index, trace in enumerate(value):
        trace_context = f"{context}.provider_traces[{index}]"
        if not isinstance(trace, Mapping):
            raise EvaluationError(f"{trace_context} must be an object")
        stage = trace.get("stage")
        if not isinstance(stage, str) or not stage:
            raise EvaluationError(f"{trace_context}.stage must be non-empty")
        if trace.get("status") != "ok":
            raise EvaluationError(f"{trace_context} is not a successful trace")
        schema = trace.get("trace_schema_version")
        if schema == "chain-provider-json-trace-v1":
            calls = trace.get("provider_calls")
            if not isinstance(calls, list) or not calls:
                raise EvaluationError(
                    f"{trace_context} JSON trace requires provider_calls"
                )
            correction_count = _strict_nonnegative_int(
                trace.get("correction_count"),
                f"{trace_context}.correction_count",
            )
            if len(calls) != correction_count + 1:
                raise EvaluationError(
                    f"{trace_context} JSON correction/provider-call count mismatch"
                )
            operational = trace.get("operational")
            if (
                not isinstance(operational, Mapping)
                or operational.get("provider_call_count") != len(calls)
            ):
                raise EvaluationError(
                    f"{trace_context} JSON operational call count mismatch"
                )
            for call_index, call in enumerate(calls):
                observed_api_calls += _provider_call_attempt_count(
                    call,
                    context=f"{trace_context}.provider_calls[{call_index}]",
                )
        elif schema == "chain-provider-trace-v1":
            observed_api_calls += _provider_call_attempt_count(
                trace, context=trace_context
            )
        else:
            raise EvaluationError(f"{trace_context} trace version is unsupported")
    if observed_api_calls != expected_api_calls:
        raise EvaluationError(
            f"{context} provider trace/API usage call count mismatch"
        )


def _validate_prediction_trace_completeness(
    prediction: Mapping[str, Any], *, context: str
) -> None:


    leaked_field = _find_prohibited_prediction_field(prediction)
    if leaked_field is not None:
        raise EvaluationError(
            f"{context} prediction trace contains evaluation-only field {leaked_field}"
        )
    required = {
        "reasoning_summary",
        "key_evidence",
        "reasoning_graph",
        "direction",
        "rounds",
        "latest_round",
        "outcomes",
        "outcome_mapping",
        "target_resolution",
        "reasoning_rounds",
        "provider_traces",
        "causal_inference",
    }
    missing = sorted(required - set(prediction))
    if missing:
        raise EvaluationError(
            f"{context} prediction trace is incomplete; missing {missing}"
        )
    if not isinstance(prediction.get("reasoning_summary"), str):
        raise EvaluationError(f"{context} prediction reasoning_summary must be a string")
    if not isinstance(prediction.get("key_evidence"), list):
        raise EvaluationError(f"{context} prediction key_evidence must be a list")
    reasoning_graph = prediction.get("reasoning_graph")
    if not isinstance(reasoning_graph, Mapping):
        raise EvaluationError(f"{context} prediction reasoning_graph must be an object")
    try:
        from chain.graph.reasoning_graph import ReasoningGraph

        restored_graph = ReasoningGraph.from_dict(reasoning_graph)
    except Exception as exc:
        raise EvaluationError(f"{context} prediction reasoning_graph is invalid") from exc
    if restored_graph.to_dict() != dict(reasoning_graph):
        raise EvaluationError(f"{context} prediction reasoning_graph round-trip mismatch")
    if reasoning_graph.get("case_id") != prediction.get("case_id"):
        raise EvaluationError(f"{context} prediction reasoning_graph case identity mismatch")
    if reasoning_graph.get("graph_bundle_sha256") != prediction.get(
        "graph_bundle_sha256"
    ):
        raise EvaluationError(f"{context} prediction reasoning_graph bundle mismatch")
    if not isinstance(prediction.get("direction"), Mapping):
        raise EvaluationError(f"{context} prediction direction must be an object")
    rounds = _strict_nonnegative_int(prediction.get("rounds"), f"{context}.prediction.rounds")
    latest_round = _strict_nonnegative_int(
        prediction.get("latest_round"), f"{context}.prediction.latest_round"
    )
    reasoning_rounds = prediction.get("reasoning_rounds")
    if not isinstance(reasoning_rounds, list) or len(reasoning_rounds) != rounds:
        raise EvaluationError(f"{context} prediction reasoning round count mismatch")
    if rounds <= 0:
        raise EvaluationError(f"{context} prediction requires at least one reasoning round")
    if latest_round != reasoning_rounds[-1].get("round"):
        raise EvaluationError(f"{context} prediction latest_round mismatch")
    graph_round_log = reasoning_graph.get("round_log")
    if not isinstance(graph_round_log, list) or len(graph_round_log) != rounds:
        raise EvaluationError(f"{context} prediction reasoning graph round count mismatch")
    for index, item in enumerate(reasoning_rounds):
        round_context = f"{context}.prediction.reasoning_rounds[{index}]"
        if not isinstance(item, Mapping) or set(item) != {
            "schema_version",
            "round",
            "iteration",
            "reason_i",
            "step_result",
        }:
            raise EvaluationError(f"{context} prediction reasoning round {index} is invalid")
        if item.get("schema_version") != "chain-reasoning-round-trace-v1":
            raise EvaluationError(f"{round_context} version mismatch")
        round_number = _strict_nonnegative_int(
            item.get("round"), f"{round_context}.round"
        )
        if round_number != index + 1:
            raise EvaluationError(f"{round_context} is not contiguous and ordered")
        iteration = _strict_nonnegative_int(
            item.get("iteration"), f"{round_context}.iteration"
        )
        reason_i = _strict_nonnegative_int(
            item.get("reason_i"), f"{round_context}.reason_i"
        )
        if iteration <= 0 or reason_i <= 0:
            raise EvaluationError(f"{round_context} iteration indices must be positive")
        step_result = item.get("step_result")
        if not isinstance(step_result, Mapping):
            raise EvaluationError(
                f"{context} prediction reasoning round {index} lacks step_result"
            )
        scored_context = step_result.get("scored_context")
        if not isinstance(scored_context, Mapping):
            raise EvaluationError(f"{round_context} lacks complete scored_context")
        context_id = scored_context.get("context_id")
        if not isinstance(context_id, str) or not context_id:
            raise EvaluationError(f"{round_context} scored_context identity is invalid")
        _finite_probability(
            step_result.get("confidence"), f"{round_context}.step_result.confidence"
        )
        round_outcomes = step_result.get("outcomes")
        if not isinstance(round_outcomes, list) or not round_outcomes:
            raise EvaluationError(f"{round_context} outcomes are incomplete")
        graph_log = graph_round_log[index]
        if (
            not isinstance(graph_log, Mapping)
            or graph_log.get("round") != round_number
            or graph_log.get("action") != "reason"
            or graph_log.get("context_id") != context_id
            or graph_log.get("confidence") != step_result.get("confidence")
            or graph_log.get("reasoning") != step_result.get("reasoning")
        ):
            raise EvaluationError(f"{round_context} does not close over reasoning_graph")
    outcomes = prediction.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise EvaluationError(f"{context} prediction outcomes must be a non-empty list")
    outcome_ids: set[str] = set()
    normalized_outcomes: List[Dict[str, Any]] = []
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise EvaluationError(f"{context} prediction outcome {index} is invalid")
        outcome_id = outcome.get("outcome_id")
        if not isinstance(outcome_id, str) or not outcome_id or outcome_id in outcome_ids:
            raise EvaluationError(f"{context} prediction outcome ids are invalid")
        outcome_ids.add(outcome_id)
        probability = _finite_probability(
            outcome.get("probability"), f"{context}.prediction.outcomes[{index}].probability"
        )
        name = outcome.get("name")
        description = outcome.get("description")
        if not isinstance(name, str) or not name or not isinstance(description, str):
            raise EvaluationError(f"{context} prediction outcome {index} text is invalid")
        normalized_outcomes.append(
            {
                "outcome_id": outcome_id,
                "name": name,
                "description": description,
                "probability": probability,
            }
        )
    if not math.isclose(
        sum(item["probability"] for item in normalized_outcomes),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise EvaluationError(f"{context} prediction outcome probability mass mismatch")
    latest_step_outcomes = reasoning_rounds[-1]["step_result"]["outcomes"]
    normalized_latest: List[Dict[str, Any]] = []
    for index, outcome in enumerate(latest_step_outcomes):
        if not isinstance(outcome, Mapping):
            raise EvaluationError(f"{context} latest round outcome {index} is invalid")
        normalized_latest.append(
            {
                "outcome_id": outcome.get("outcome_id"),
                "name": outcome.get("name"),
                "description": outcome.get("description"),
                "probability": _finite_probability(
                    outcome.get("probability"),
                    f"{context}.prediction.latest_outcomes[{index}].probability",
                ),
            }
        )
    if normalized_latest != normalized_outcomes:
        raise EvaluationError(
            f"{context} final outcomes do not match the latest reasoning round"
        )
    mapping = prediction.get("outcome_mapping")
    if not isinstance(mapping, Mapping) or not isinstance(mapping.get("mappings"), list):
        raise EvaluationError(f"{context} prediction outcome_mapping is incomplete")
    mapped_ids: set[str] = set()
    for index, item in enumerate(mapping["mappings"]):
        if not isinstance(item, Mapping):
            raise EvaluationError(f"{context} prediction outcome mapping {index} is invalid")
        outcome_id = item.get("outcome_id")
        if outcome_id not in outcome_ids or outcome_id in mapped_ids:
            raise EvaluationError(f"{context} prediction outcome mapping coverage mismatch")
        if item.get("axis_side") not in {"positive", "negative"}:
            raise EvaluationError(f"{context} prediction outcome mapping axis side is invalid")
        mapped_ids.add(outcome_id)
    if mapped_ids != outcome_ids:
        raise EvaluationError(f"{context} prediction outcome mapping is not an exact cover")
    side_by_id = {
        item["outcome_id"]: item["axis_side"] for item in mapping["mappings"]
    }
    positive_mass = sum(
        item["probability"]
        for item in normalized_outcomes
        if side_by_id[item["outcome_id"]] == "positive"
    )
    if not math.isclose(
        positive_mass,
        _finite_probability(
            prediction.get("p_llm_event"), f"{context}.prediction.p_llm_event"
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise EvaluationError(f"{context} mapped outcome mass does not match p_llm_event")
    target_resolution = prediction.get("target_resolution")
    if not isinstance(target_resolution, Mapping):
        raise EvaluationError(f"{context} prediction target_resolution must be an object")
    if not isinstance(target_resolution.get("target_signs"), Mapping):
        raise EvaluationError(f"{context} prediction target_resolution target_signs is invalid")
    for target_id, sign in target_resolution["target_signs"].items():
        if (
            not isinstance(target_id, str)
            or not target_id
            or isinstance(sign, bool)
            or sign not in {-1, 1}
        ):
            raise EvaluationError(
                f"{context} prediction target_resolution target sign is invalid"
            )
    if not isinstance(target_resolution.get("dropped"), list):
        raise EvaluationError(f"{context} prediction target_resolution dropped is invalid")
    if not isinstance(prediction.get("provider_traces"), list):
        raise EvaluationError(f"{context} prediction provider_traces must be a list")
    causal = prediction.get("causal_inference")
    if not isinstance(causal, Mapping):
        raise EvaluationError(f"{context} prediction causal_inference must be an object")
    causal_required = {
        "prediction",
        "scored_chains",
        "deduplicated_chains",
        "top_chains",
        "n_chains",
        "n_chains_raw",
        "chain_diversity",
    }
    causal_missing = sorted(causal_required - set(causal))
    if causal_missing:
        raise EvaluationError(
            f"{context} prediction causal trace is incomplete; missing {causal_missing}"
        )
    scored = causal.get("scored_chains")
    deduplicated = causal.get("deduplicated_chains")
    top = causal.get("top_chains")
    if not isinstance(scored, list) or not isinstance(deduplicated, list) or not isinstance(top, list):
        raise EvaluationError(f"{context} prediction causal chain collections are invalid")
    n_raw = _strict_nonnegative_int(causal.get("n_chains_raw"), f"{context}.prediction.n_chains_raw")
    n_chains = _strict_nonnegative_int(causal.get("n_chains"), f"{context}.prediction.n_chains")
    diversity = _strict_nonnegative_int(causal.get("chain_diversity"), f"{context}.prediction.chain_diversity")
    if n_raw != len(scored) or n_chains != len(deduplicated) or diversity != n_chains:
        raise EvaluationError(f"{context} prediction causal chain counts mismatch")
    if len(top) > n_chains or top != deduplicated[: len(top)]:
        raise EvaluationError(f"{context} prediction causal top-chain prefix mismatch")
    scored_by_identity: Dict[tuple[str, ...], Mapping[str, Any]] = {}
    collection_identities: Dict[str, List[tuple[str, ...]]] = {}
    for collection_name, collection in (
        ("scored_chains", scored),
        ("deduplicated_chains", deduplicated),
        ("top_chains", top),
    ):
        identities: List[tuple[str, ...]] = []
        for index, chain in enumerate(collection):
            if not isinstance(chain, Mapping):
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] is invalid"
                )
            if chain.get("direction") not in {"supports", "opposes"}:
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] direction is invalid"
                )
            identity = chain.get("typed_path_identity")
            if (
                not isinstance(identity, list)
                or not identity
                or any(not isinstance(item, str) or not item for item in identity)
            ):
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] path identity is invalid"
                )
            identity_tuple = tuple(identity)
            if identity_tuple in identities:
                raise EvaluationError(
                    f"{context} prediction {collection_name} has duplicate path identities"
                )
            identities.append(identity_tuple)
            _finite_probability(
                chain.get("score"),
                f"{context}.prediction.{collection_name}[{index}].score",
            )
            target_id = chain.get("target_id")
            if not isinstance(target_id, str) or not target_id:
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] target is invalid"
                )
            edges = chain.get("edges")
            nodes = chain.get("chain")
            if (
                not isinstance(edges, list)
                or len(edges) != len(identity)
                or not isinstance(nodes, list)
                or len(nodes) != len(edges) + 1
                or any(not isinstance(node, str) or not node for node in nodes)
                or len(set(nodes)) != len(nodes)
                or _strict_nonnegative_int(
                    chain.get("depth"),
                    f"{context}.prediction.{collection_name}[{index}].depth",
                )
                != len(edges)
                or target_id != nodes[-1]
            ):
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] path topology is invalid"
                )
            logical_ids = []
            prevents_count = 0
            for edge_index, edge in enumerate(edges):
                if not isinstance(edge, Mapping):
                    raise EvaluationError(
                        f"{context} prediction {collection_name}[{index}] edge is invalid"
                    )
                logical_edge_id = edge.get("logical_edge_id")
                if not isinstance(logical_edge_id, str) or not logical_edge_id:
                    raise EvaluationError(
                        f"{context} prediction {collection_name}[{index}] edge identity is invalid"
                    )
                if (
                    edge.get("cause_id") != nodes[edge_index]
                    or edge.get("effect_id") != nodes[edge_index + 1]






                    or edge.get("causal_type")
                    not in {"causes", "enables", "prevents"}
                ):
                    raise EvaluationError(
                        f"{context} prediction {collection_name}[{index}] edge topology is invalid"
                    )
                prevents_count += edge.get("causal_type") == "prevents"
                logical_ids.append(logical_edge_id)
            if logical_ids != identity:
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] typed path mismatch"
                )
            target_sign = chain.get("target_sign")
            if (
                isinstance(target_sign, bool)
                or target_sign not in {-1, 1}
                or _strict_nonnegative_int(
                    chain.get("prevents_count"),
                    f"{context}.prediction.{collection_name}[{index}].prevents_count",
                )
                != prevents_count
            ):
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] path polarity is invalid"
                )
            expected_direction = (
                "supports"
                if target_sign * (-1 if prevents_count % 2 else 1) > 0
                else "opposes"
            )
            if chain.get("direction") != expected_direction:
                raise EvaluationError(
                    f"{context} prediction {collection_name}[{index}] path direction mismatch"
                )
            if collection_name == "scored_chains":
                scored_by_identity[identity_tuple] = chain
        collection_identities[collection_name] = identities
    for index, chain in enumerate(deduplicated):
        identity = collection_identities["deduplicated_chains"][index]
        if identity not in scored_by_identity or scored_by_identity[identity] != chain:
            raise EvaluationError(
                f"{context} prediction deduplicated chains are not a subset of scored chains"
            )
    retained_supporting = sum(
        chain.get("direction") == "supports" for chain in deduplicated
    )
    retained_opposing = sum(
        chain.get("direction") == "opposes" for chain in deduplicated
    )
    causal_prediction = causal.get("prediction")
    if (
        not isinstance(causal_prediction, Mapping)
        or causal_prediction.get("n_supporting") != retained_supporting
        or causal_prediction.get("n_opposing") != retained_opposing
    ):
        raise EvaluationError(f"{context} prediction retained chain polarity mismatch")
    causal_probability = causal.get("causal_probability")
    if not isinstance(causal_probability, Mapping):
        raise EvaluationError(f"{context} prediction causal_probability is incomplete")
    raw_supporting = sum(chain.get("direction") == "supports" for chain in scored)
    raw_opposing = sum(chain.get("direction") == "opposes" for chain in scored)
    if (
        causal_probability.get("n_supporting") != retained_supporting
        or causal_probability.get("n_opposing") != retained_opposing
        or causal_probability.get("n_supporting_raw") != raw_supporting
        or causal_probability.get("n_opposing_raw") != raw_opposing
    ):
        raise EvaluationError(f"{context} prediction causal chain polarity counts mismatch")
    causal_prediction = causal.get("prediction")
    if not isinstance(causal_prediction, Mapping):
        raise EvaluationError(f"{context} prediction causal prediction is invalid")
    supporting = sum(chain.get("direction") == "supports" for chain in deduplicated)
    opposing = sum(chain.get("direction") == "opposes" for chain in deduplicated)
    if (
        _strict_nonnegative_int(
            causal_prediction.get("n_supporting"),
            f"{context}.prediction.causal_inference.prediction.n_supporting",
        )
        != supporting
        or _strict_nonnegative_int(
            causal_prediction.get("n_opposing"),
            f"{context}.prediction.causal_inference.prediction.n_opposing",
        )
        != opposing
    ):
        raise EvaluationError(f"{context} prediction causal direction counts mismatch")


def print_metrics(metrics: Mapping[str, Any]) -> None:
    auc = metrics["auc_roc"]
    auc_text = f"{auc:.6f}" if isinstance(auc, float) else str(auc)
    maximum_error = metrics["mce"]
    mce_text = (
        f"{maximum_error:.6f}" if isinstance(maximum_error, float) else str(maximum_error)
    )
    print(
        f"  [{metrics['label']}] n={metrics['n']} "
        f"Acc={metrics['accuracy']:.6f} Brier={metrics['brier_score']:.6f} "
        f"ECE={metrics['ece']:.6f} ACE={metrics['ace']:.6f} "
        f"MCE={mce_text} Rel={metrics['reliability']:.6f} "
        f"NLL={metrics['nll']:.6f} AUC={auc_text} "
        f"(positive={metrics['pred_positive']} negative={metrics['pred_negative']})"
    )


def _validate_paper_report_rows(
    results: Sequence[Mapping[str, Any]], *, config: Any, result_path: Optional[Path] = None
) -> bool:


    if not results:
        raise EvaluationError("paper report requires at least one canonical row")
    from chain.config import ResolvedConfig

    if not isinstance(config, ResolvedConfig) or not config.is_paper or not config.publishable:
        raise EvaluationError("paper report requires a locked paper ResolvedConfig")
    if result_path is None:
        raise EvaluationError("paper v2 report requires the result path")
    expected_fusion_parameters = _fusion_parameters_from_config(config)
    expected_run_hash: Optional[str] = None
    seen_cases: set[str] = set()
    run_payload: Optional[Dict[str, Any]] = None
    row_case_inputs: List[str] = []
    sealed_source_schema_fields: Optional[List[str]] = None
    for index, row in enumerate(results):
        context = f"paper report row {index}"
        if not isinstance(row, Mapping) or set(row) != _CANONICAL_ROW_FIELDS:
            raise EvaluationError(f"{context} canonical fields mismatch")
        if row.get("schema_version") != EVAL_CASE_SCHEMA:
            raise EvaluationError(f"{context} has an unsupported schema")
        if row.get("profile_id") != _PAPER_PROFILE_ID:
            raise EvaluationError("paper report requires paper-profile rows")
        run_identity = row.get("run_identity")
        if not isinstance(run_identity, Mapping) or set(run_identity) != _RUN_IDENTITY_FIELDS:
            raise EvaluationError(f"{context} run identity schema mismatch")
        run_hash = _strict_sha256(row.get("run_identity_sha256"), f"{context}.run_identity_sha256")
        if run_hash != _sha256_payload(run_identity):
            raise EvaluationError(f"{context} run identity hash mismatch")
        if expected_run_hash is None:
            expected_run_hash = run_hash
            run_payload = dict(run_identity)
        elif run_hash != expected_run_hash or dict(run_identity) != run_payload:
            raise EvaluationError("paper report mixes multiple run identities")
        assert run_payload is not None
        if run_payload.get("schema_version") != EVAL_RUN_SCHEMA:
            raise EvaluationError("paper report run identity schema mismatch")
        if run_payload.get("profile_id") != _PAPER_PROFILE_ID:
            raise EvaluationError("paper report run identity profile mismatch")
        expected_config_identity = {
            "construction_config_sha256": config.construction_config_sha256,
            "scientific_config_sha256": config.scientific_config_sha256,
        }
        for field, expected_value in expected_config_identity.items():
            if run_payload.get(field) != expected_value:
                raise EvaluationError(f"paper report run/config mismatch for {field}")
        reasons = run_payload.get("nonpublishable_reasons")
        if reasons not in ([], ["limited_case_selection"]):
            raise EvaluationError("paper report run identity has unsupported nonpublishable reasons")
        case_count = run_payload.get("case_count")
        if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
            raise EvaluationError("paper report run case_count must be a positive integer")
        _strict_sha256(
            run_payload.get("ordered_case_inputs_sha256"),
            "paper report ordered_case_inputs_sha256",
        )
        expected_publishable = reasons == []
        if row.get("publishable") is not expected_publishable:
            raise EvaluationError("paper report row publishability does not match its sealed run")
        _strict_sha256(run_payload.get("test_file_sha256"), "paper report test_file_sha256")
        for field in (
            "construction_config_sha256",
            "scientific_config_sha256",
            "graph_bundle_sha256",
        ):
            _strict_sha256(run_payload.get(field), f"paper report run.{field}")
            if row.get(field) != run_payload.get(field):
                raise EvaluationError(f"{context} row/run mismatch for {field}")
        fusion_parameters = _validate_fusion_parameters(
            run_payload.get("fusion_parameters"), context="paper report run"
        )
        if fusion_parameters != expected_fusion_parameters:
            raise EvaluationError("paper report fusion parameters do not match resolved config")
        metric_protocol = _validate_metric_protocol(
            run_payload.get("metric_protocol"), context="paper report run"
        )
        if metric_protocol != _metric_protocol_from_config(config):
            raise EvaluationError("paper report metric protocol does not match resolved config")

        case_id = _strict_identity(row.get("case_id"), f"{context}.case_id")
        if case_id != case_id.strip():
            raise EvaluationError(f"{context}.case_id is not canonical")
        if case_id in seen_cases:
            raise EvaluationError("paper report contains an invalid/duplicate case identity")
        seen_cases.add(case_id)
        if row.get("id") != case_id:
            raise EvaluationError(f"{context}.id/case_id mismatch")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(f"{context}.question must be non-empty")
        if question != question.strip():
            raise EvaluationError(f"{context}.question is not canonical")
        cutoff = row.get("cutoff")
        if not isinstance(cutoff, str) or not cutoff.strip():
            raise EvaluationError(f"{context}.cutoff must be non-empty")
        from chain.skills.temporal_validity import parse_cutoff

        try:
            canonical_cutoff = parse_cutoff(cutoff).canonical
        except ValueError as exc:
            raise EvaluationError(f"{context}.cutoff is invalid") from exc
        if cutoff != canonical_cutoff:
            raise EvaluationError(f"{context}.cutoff is not canonical")
        from chain.config import validate_binary_axis

        try:
            axis = validate_binary_axis(row.get("binary_axis"))
        except Exception as exc:
            raise EvaluationError(f"{context}.binary_axis is invalid") from exc
        if (
            axis.get("source") == "legacy_adapter"
            or axis.get("adapter_id") == "chain-legacy-native-yes-no-v1"
        ):
            raise EvaluationError("paper report requires an explicit non-legacy binary axis")
        if row.get("binary_axis") != axis:
            raise EvaluationError(f"{context}.binary_axis is not canonical")
        if row.get("binary_axis_sha256") != axis["binary_axis_sha256"]:
            raise EvaluationError(f"{context}.binary_axis_sha256 mismatch")
        gold = _gold_event(row)
        if row.get("gold_event") != gold or isinstance(row.get("gold_event"), bool):
            raise EvaluationError(f"{context}.gold_event must be canonical integer")
        expected_case_payload = {
            "case_id": case_id,
            "question": question,
            "cutoff": cutoff,
            "binary_axis_sha256": axis["binary_axis_sha256"],
            "gold_event": gold,
            "domain": row.get("domain"),
            "source": row.get("source"),
        }
        if not isinstance(expected_case_payload["domain"], str) or not isinstance(
            expected_case_payload["source"], str
        ):
            raise EvaluationError(f"{context} domain/source must be strings")
        source_record = row.get("source_record")
        if not isinstance(source_record, Mapping):
            raise EvaluationError(f"{context}.source_record must be an object")
        _validate_source_record_case_closure(
            source_record,
            case_id=case_id,
            question=question,
            domain=str(expected_case_payload["domain"]),
            source=str(expected_case_payload["source"]),
            gold_event=gold,
            binary_axis=axis,
            context=context,
        )
        source_record_hash = _strict_sha256(
            row.get("source_record_sha256"), f"{context}.source_record_sha256"
        )
        if source_record_hash != _sha256_payload(source_record):
            raise EvaluationError(f"{context}.source_record identity mismatch")
        source_record_completeness = _validate_source_record_completeness(
            row.get("source_record_completeness"),
            source_record=source_record,
            context=context,
        )
        row_source_schema_fields = source_record_completeness["schema_fields"]
        if sealed_source_schema_fields is None:
            sealed_source_schema_fields = row_source_schema_fields
        elif row_source_schema_fields != sealed_source_schema_fields:
            raise EvaluationError("paper report mixes source-record schemas")
        input_adaptation = _validate_input_adaptation(
            row.get("input_adaptation"),
            source_record=source_record,
            cutoff=cutoff,
            binary_axis=axis,
            context=context,
        )
        expected_case_payload.update(
            {
                "source_record": copy.deepcopy(dict(source_record)),
                "source_record_sha256": source_record_hash,
                "source_record_completeness": source_record_completeness,
                "input_adaptation": input_adaptation,
            }
        )
        case_input_hash = _strict_sha256(row.get("case_input_sha256"), f"{context}.case_input_sha256")
        if case_input_hash != _sha256_payload(expected_case_payload):
            raise EvaluationError(f"{context}.case input identity mismatch")
        row_case_inputs.append(case_input_hash)

        probability = _p_event(row)
        for field in ("p_llm_event", "p_causal_event", "p_final_event"):
            _finite_probability(row.get(field), f"{context}.{field}")
        predicted_event = int(probability >= 0.5)
        if row.get("predicted_event") != predicted_event or isinstance(
            row.get("predicted_event"), bool
        ):
            raise EvaluationError(f"{context}.predicted_event mismatch")
        side = axis["positive" if predicted_event else "negative"]
        gold_side = axis["positive" if gold else "negative"]
        if row.get("pred_answer") != str(side["label"]):
            raise EvaluationError(f"{context}.pred_answer mismatch")
        if row.get("ground_truth") != str(gold_side["label"]):
            raise EvaluationError(f"{context}.ground_truth mismatch")
        expected_confidence = probability if predicted_event else 1.0 - probability
        if _finite_probability(row.get("confidence"), f"{context}.confidence") != expected_confidence:
            raise EvaluationError(f"{context}.confidence mismatch")
        if not isinstance(row.get("correct"), bool) or row["correct"] != (predicted_event == gold):
            raise EvaluationError(f"{context}.correct mismatch")
        if not isinstance(row.get("converged"), bool):
            raise EvaluationError(f"{context}.converged must be boolean")
        if row.get("collect_rounds") != 0 or row.get("collect_stats") != []:
            raise EvaluationError(f"{context} contains collector activity")
        causal_summary = _validate_causal_summary(
            row.get("causal_summary"),
            case={
                "case_id": case_id,
                "cutoff": cutoff,
                "binary_axis_sha256": axis["binary_axis_sha256"],
            },
            run_payload=run_payload,
            graph_bundle_sha256=str(run_payload["graph_bundle_sha256"]),
        )
        if causal_summary["p_causal_event"] != row["p_causal_event"]:
            raise EvaluationError(f"{context}.causal_summary probability mismatch")
        _validate_fusion_closure(
            probabilities=row,
            causal_summary=causal_summary,
            fusion_parameters=fusion_parameters,
            context=context,
        )
        _validate_usage(row.get("usage"), context=context)
        runtime = row.get("runtime_telemetry")
        if not isinstance(runtime, Mapping) or set(runtime) != {"elapsed_seconds"}:
            raise EvaluationError(f"{context}.runtime_telemetry schema mismatch")
        _strict_nonnegative_float(runtime.get("elapsed_seconds"), f"{context}.elapsed_seconds")
        _validate_prediction_trace_closure(
            row,
            case={
                **expected_case_payload,
                "binary_axis": axis,
                "case_input_sha256": case_input_hash,
            },
            run_payload=run_payload,
            result_path=result_path,
            context=context,
        )

    assert run_payload is not None
    if run_payload.get("case_count") != len(results):
        raise EvaluationError("paper report case count does not close over the sealed run")
    if run_payload.get("ordered_case_inputs_sha256") != _sha256_payload(row_case_inputs):
        raise EvaluationError("paper report ordered case input identity mismatch")
    return run_payload["nonpublishable_reasons"] == ["limited_case_selection"]


def _validate_compat_canonical_report_rows(
    results: Sequence[Mapping[str, Any]],
    *,
    result_path: Optional[Path] = None,
    validate_traces: bool = True,
) -> bool:


    if not results:
        raise EvaluationError("compat canonical report requires at least one row")
    first = results[0]
    schema = first.get("schema_version") if isinstance(first, Mapping) else None
    if schema == EVAL_CASE_SCHEMA:
        expected_fields = _CANONICAL_ROW_FIELDS
        expected_run_schema = EVAL_RUN_SCHEMA
        allowed_run_fields = {_RUN_IDENTITY_FIELDS}
        if result_path is None:
            raise EvaluationError("v2 compat report requires the result path")
    elif schema == EVAL_CASE_SCHEMA_V1:
        expected_fields = _CANONICAL_ROW_FIELDS_V1
        expected_run_schema = EVAL_RUN_SCHEMA_V1
        allowed_run_fields = {
            _RUN_IDENTITY_FIELDS_V1,
            _LEGACY_RUN_IDENTITY_FIELDS_V1,
        }
    else:
        raise EvaluationError("compat canonical report has an unsupported schema")
    if not isinstance(first, Mapping) or set(first) != expected_fields:
        raise EvaluationError("compat canonical report row 0 fields mismatch")
    run_identity = first.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise EvaluationError("compat canonical report run identity is missing")
    run_fields = frozenset(run_identity)
    if run_fields not in allowed_run_fields:
        raise EvaluationError("compat canonical report run identity schema mismatch")
    legacy_canonical_identity = (
        schema == EVAL_CASE_SCHEMA_V1
        and run_fields == _LEGACY_RUN_IDENTITY_FIELDS_V1
    )
    run_payload = dict(run_identity)
    run_hash = _strict_sha256(
        first.get("run_identity_sha256"),
        "compat canonical report run_identity_sha256",
    )
    if _sha256_payload(run_payload) != run_hash:
        raise EvaluationError("compat canonical report run identity hash mismatch")
    if run_payload.get("schema_version") != expected_run_schema:
        raise EvaluationError("compat canonical report run schema mismatch")
    if run_payload.get("profile_id") != _COMPAT_PROFILE_ID:
        raise EvaluationError("compat canonical report requires compat-profile rows")
    case_count = run_payload.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise EvaluationError("compat canonical report case_count must be positive")
    reasons = run_payload.get("nonpublishable_reasons")
    if (
        not isinstance(reasons, list)
        or any(
            not isinstance(reason, str)
            or not reason
            or reason != reason.strip()
            for reason in reasons
        )
        or len(set(reasons)) != len(reasons)
        or "compat_profile" not in reasons
    ):
        raise EvaluationError("compat canonical report nonpublishable reasons are invalid")
    for field in (
        "construction_config_sha256",
        "scientific_config_sha256",
        "graph_bundle_sha256",
        "test_file_sha256",
        "ordered_case_inputs_sha256",
    ):
        _strict_sha256(run_payload.get(field), f"compat canonical report run.{field}")
    if not legacy_canonical_identity:
        _validate_fusion_parameters(
            run_payload.get("fusion_parameters"),
            context="compat canonical report run",
        )
    if schema == EVAL_CASE_SCHEMA:
        _validate_metric_protocol(
            run_payload.get("metric_protocol"),
            context="compat canonical report run",
        )

    from chain.config import validate_binary_axis
    from chain.skills.temporal_validity import parse_cutoff

    cases_by_id: Dict[str, Dict[str, Any]] = {}
    ordered_case_hashes: List[str] = []
    sealed_source_schema_fields: Optional[List[str]] = None
    for index, row in enumerate(results):
        context = f"compat canonical report row {index}"
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise EvaluationError(f"{context} fields mismatch")
        sealed_run = row.get("run_identity")
        sealed_hash = _strict_sha256(
            row.get("run_identity_sha256"), f"{context}.run_identity_sha256"
        )
        if (
            not isinstance(sealed_run, Mapping)
            or dict(sealed_run) != run_payload
            or sealed_hash != run_hash
            or _sha256_payload(sealed_run) != sealed_hash
        ):
            raise EvaluationError(f"{context} run identity mismatch")
        if row.get("schema_version") != schema:
            raise EvaluationError(f"{context} schema mismatch")
        if row.get("profile_id") != _COMPAT_PROFILE_ID:
            raise EvaluationError(f"{context} profile mismatch")

        case_id = _strict_identity(row.get("case_id"), f"{context}.case_id")
        if case_id != case_id.strip() or case_id in cases_by_id:
            raise EvaluationError(f"{context} case identity is invalid or duplicated")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip() or question != question.strip():
            raise EvaluationError(f"{context}.question is not canonical")
        cutoff = row.get("cutoff")
        if not isinstance(cutoff, str) or not cutoff:
            raise EvaluationError(f"{context}.cutoff is invalid")
        try:
            canonical_cutoff = parse_cutoff(cutoff).canonical
        except ValueError as exc:
            raise EvaluationError(f"{context}.cutoff is invalid") from exc
        if cutoff != canonical_cutoff:
            raise EvaluationError(f"{context}.cutoff is not canonical")
        try:
            axis = validate_binary_axis(row.get("binary_axis"))
        except Exception as exc:
            raise EvaluationError(f"{context}.binary_axis is invalid") from exc
        if row.get("binary_axis") != axis:
            raise EvaluationError(f"{context}.binary_axis is not canonical")
        if row.get("binary_axis_sha256") != axis["binary_axis_sha256"]:
            raise EvaluationError(f"{context}.binary_axis_sha256 mismatch")
        gold = _gold_event(row)
        if isinstance(row.get("gold_event"), bool) or row.get("gold_event") != gold:
            raise EvaluationError(f"{context}.gold_event is not canonical")
        domain = row.get("domain")
        source = row.get("source")
        if not isinstance(domain, str) or not isinstance(source, str):
            raise EvaluationError(f"{context} domain/source must be strings")
        case_payload = {
            "case_id": case_id,
            "question": question,
            "cutoff": cutoff,
            "binary_axis_sha256": axis["binary_axis_sha256"],
            "gold_event": gold,
            "domain": domain,
            "source": source,
        }
        if schema == EVAL_CASE_SCHEMA:
            source_record = row.get("source_record")
            if not isinstance(source_record, Mapping):
                raise EvaluationError(f"{context}.source_record must be an object")
            _validate_source_record_case_closure(
                source_record,
                case_id=case_id,
                question=question,
                domain=domain,
                source=source,
                gold_event=gold,
                binary_axis=axis,
                context=context,
            )
            source_record_hash = _strict_sha256(
                row.get("source_record_sha256"),
                f"{context}.source_record_sha256",
            )
            if source_record_hash != _sha256_payload(source_record):
                raise EvaluationError(f"{context} source record identity mismatch")
            source_record_completeness = _validate_source_record_completeness(
                row.get("source_record_completeness"),
                source_record=source_record,
                context=context,
            )
            row_source_schema_fields = source_record_completeness["schema_fields"]
            if sealed_source_schema_fields is None:
                sealed_source_schema_fields = row_source_schema_fields
            elif row_source_schema_fields != sealed_source_schema_fields:
                raise EvaluationError(
                    "compat canonical report mixes source-record schemas"
                )
            adaptation = _validate_input_adaptation(
                row.get("input_adaptation"),
                source_record=source_record,
                cutoff=cutoff,
                binary_axis=axis,
                context=context,
            )
            case_payload.update(
                {
                    "source_record": copy.deepcopy(dict(source_record)),
                    "source_record_sha256": source_record_hash,
                    "source_record_completeness": source_record_completeness,
                    "input_adaptation": adaptation,
                }
            )
        case_hash = _strict_sha256(
            row.get("case_input_sha256"), f"{context}.case_input_sha256"
        )
        if case_hash != _sha256_payload(case_payload):
            raise EvaluationError(f"{context} case input identity mismatch")
        ordered_case_hashes.append(case_hash)
        cases_by_id[case_id] = {
            **case_payload,
            "binary_axis": axis,
            "case_input_sha256": case_hash,
        }

    if case_count != len(results):
        raise EvaluationError("compat canonical report case count mismatch")
    if run_payload["ordered_case_inputs_sha256"] != _sha256_payload(ordered_case_hashes):
        raise EvaluationError("compat canonical report ordered case identity mismatch")
    accepted = _validate_existing_rows(
        results,
        run_identity_sha256=run_hash,
        cases_by_id=cases_by_id,
        run_payload=run_payload,
        result_path=result_path,
        allow_v1_report_only=schema == EVAL_CASE_SCHEMA_V1,
        validate_traces=validate_traces,
    )
    if len(accepted) != len(results):
        raise EvaluationError("compat canonical report is incomplete")
    return legacy_canonical_identity


def generate_report(
    result_file: str,
    *,
    profile: str = "compat",
    config: Any = None,
    validation_mode: str = "full",
) -> Dict[str, Any]:


    if validation_mode not in {"full", "fast"}:
        raise EvaluationError("validation_mode must be full or fast")
    if profile == "paper" and validation_mode != "full":
        raise EvaluationError("paper reports require full trace validation")

    result_path = Path(result_file)
    results = _load_jsonl(result_path)
    probability_presence = ["p_event" in row for row in results]
    canonical_markers = {
        "schema_version",
        "run_identity",
        "run_identity_sha256",
        "case_id",
        "p_final_event",
    }
    looks_canonical = any(
        isinstance(row, Mapping) and bool(canonical_markers.intersection(row))
        for row in results
    )
    if any(probability_presence) and not all(probability_presence):
        raise EvaluationError("report cannot mix canonical and legacy result rows")
    if looks_canonical and not any(probability_presence):
        raise EvaluationError("canonical result row is missing p_event")
    uses_legacy = not all(probability_presence)
    canonical_schema: Optional[str] = None
    if not uses_legacy:
        schemas = {row.get("schema_version") for row in results}
        if len(schemas) != 1:
            raise EvaluationError("report cannot mix canonical v1/v2 result rows")
        canonical_schema = next(iter(schemas))
        if canonical_schema not in {EVAL_CASE_SCHEMA_V1, EVAL_CASE_SCHEMA}:
            raise EvaluationError("report canonical schema is unsupported")
    if profile not in {"compat", "paper"}:
        raise EvaluationError("report profile must be compat or paper")
    paper_dry_run = False
    legacy_canonical_identity = False
    if profile == "paper":
        if uses_legacy:
            raise EvaluationError("paper report forbids legacy probability reconstruction")
        if canonical_schema != EVAL_CASE_SCHEMA:
            raise EvaluationError("paper report requires canonical v2 rows")
        paper_dry_run = _validate_paper_report_rows(
            results, config=config, result_path=result_path
        )
        if paper_dry_run:
            print("[NON-PUBLISHABLE PAPER DRY-RUN] limited case selection")
    elif not uses_legacy:
        legacy_canonical_identity = _validate_compat_canonical_report_rows(
            results,
            result_path=result_path,
            validate_traces=validation_mode == "full",
        )
        if legacy_canonical_identity:
            print(
                "[NON-PAPER] pre-fusion canonical compat identity validated; "
                "Eq. (18)-(22) closure is unavailable"
            )
        else:
            trace_text = (
                (
                    "complete prediction sidecars validated"
                    if validation_mode == "full"
                    else "trace descriptors validated; full trace audit deferred"
                )
                if canonical_schema == EVAL_CASE_SCHEMA
                else "historical v1 row validated; full prediction trace unavailable"
            )
            print(
                "[NON-PAPER] canonical compat identity and Eq. (18)-(22) "
                f"closure validated; {trace_text}"
            )
    if uses_legacy:
        print(
            "[NON-PAPER] legacy pred_answer+confidence probability adapter is active; "
            "unknown/empty pred_answer maps to p_event=0.5"
        )
    if canonical_schema == EVAL_CASE_SCHEMA:
        metric_protocol = _validate_metric_protocol(
            results[0]["run_identity"].get("metric_protocol"),
            context="report run",
        )
    else:
        metric_protocol = _default_metric_protocol()
    metric_kwargs = {
        "ece_bins": metric_protocol["ece_bins"],
        "ace_bins": metric_protocol["ace_bins"],
        "reliability_bins": metric_protocol["reliability_bins"],
        "mce_min_bin_size": metric_protocol["mce_min_bin_size"],
        "nll_epsilon": metric_protocol["nll_epsilon"],
        "bootstrap_draws": metric_protocol["bootstrap_draws"],
        "bootstrap_seed": metric_protocol["bootstrap_seed"],
    }
    overall = compute_metrics(
        results,
        "Overall",
        with_ci=True,
        allow_legacy=uses_legacy,
        **metric_kwargs,
    )
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print_metrics(overall)
    if "accuracy_ci" in overall:
        print("\n  -- 95% Bootstrap CI --")
        for key, label in (("accuracy_ci", "Accuracy"), ("brier_ci", "Brier"), ("ece_ci", "ECE"), ("nll_ci", "NLL")):
            interval = overall[key]
            print(f"  {label:10s} = {interval['mean']:.6f} [{interval['ci_low']:.6f}, {interval['ci_high']:.6f}]")
    for field, title in (("difficulty", "Difficulty"), ("domain", "Domain")):
        values = sorted({str(row.get(field, "")).strip() for row in results if row.get(field)})
        if values:
            print(f"\n-- By {title} --")
            for value in values:
                subset = [row for row in results if str(row.get(field, "")).strip() == value]
                print_metrics(
                    compute_metrics(
                        subset,
                        value,
                        allow_legacy=uses_legacy,
                        **metric_kwargs,
                    )
                )
    print("=" * 60)
    return {
        "nonpaper_legacy_adapter": uses_legacy,
        "nonpublishable_paper_dry_run": paper_dry_run,
        "canonical_compat_validated": profile == "compat" and not uses_legacy,
        "legacy_canonical_identity": legacy_canonical_identity,
        "canonical_schema_version": canonical_schema,
        "trace_complete": canonical_schema == EVAL_CASE_SCHEMA,
        "metric_protocol": metric_protocol,
        "metrics": overall,
    }


def _axis_tokens(side: Mapping[str, Any]) -> set[str]:
    tokens = {
        str(side.get("id", "")).strip().casefold(),
        str(side.get("label", "")).strip().casefold(),
    }
    aliases = side.get("aliases", [])
    if isinstance(aliases, list):
        tokens.update(str(value).strip().casefold() for value in aliases)
    return {token for token in tokens if token}


def _case_gold_event(item: Mapping[str, Any], axis: Mapping[str, Any]) -> int:
    value = item.get("gold_event")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raw = item.get("ground_truth", item.get("answer"))
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int) and raw in (0, 1):
        return raw
    token = str(raw).strip().casefold()
    positive = _axis_tokens(axis["positive"])
    negative = _axis_tokens(axis["negative"])
    if token in positive and token not in negative:
        return 1
    if token in negative and token not in positive:
        return 0
    raise EvaluationError("ground truth does not match the explicit binary axis")


def _optional_record_text(
    record: Mapping[str, Any], field: str, *, context: str
) -> str:
    value = record.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise EvaluationError(f"{context} {field} must be a string or null")
    return value


def _validate_source_record_case_closure(
    source_record: Mapping[str, Any],
    *,
    case_id: str,
    question: str,
    domain: str,
    source: str,
    gold_event: int,
    binary_axis: Mapping[str, Any],
    context: str,
) -> None:
    raw_id = source_record.get("id")
    raw_question = source_record.get("question")
    if not isinstance(raw_id, str) or raw_id.strip() != case_id:
        raise EvaluationError(f"{context} source_record.id/case_id mismatch")
    if not isinstance(raw_question, str) or raw_question.strip() != question:
        raise EvaluationError(f"{context} source_record.question mismatch")
    raw_domain = _optional_record_text(
        source_record, "domain", context=context
    )
    raw_source = _optional_record_text(
        source_record, "source", context=context
    )
    if raw_domain != domain:
        raise EvaluationError(f"{context} source_record.domain mismatch")
    if raw_source != source:
        raise EvaluationError(f"{context} source_record.source mismatch")
    if _case_gold_event(source_record, binary_axis) != gold_event:
        raise EvaluationError(f"{context} source_record gold-event mismatch")


def _metric_protocol_from_config(config: Any) -> Dict[str, Any]:
    return {
        "schema_version": METRIC_PROTOCOL_SCHEMA,
        "ece_bins": int(config.ece_bins),
        "ace_bins": int(config.ace_bins),
        "reliability_bins": int(config.reliability_bins),
        "mce_min_bin_size": int(config.mce_min_bin_size),
        "oc_uc_confidence_gap": float(config.oc_uc_confidence_gap),
        "nll_epsilon": float(config.nll_epsilon),
        "bootstrap_draws": int(config.bootstrap_draws),
        "bootstrap_seed": int(config.bootstrap_seed),
    }


def _default_metric_protocol() -> Dict[str, Any]:


    return {
        "schema_version": METRIC_PROTOCOL_SCHEMA,
        "ece_bins": DEFAULT_ECE_BINS,
        "ace_bins": DEFAULT_ACE_BINS,
        "reliability_bins": DEFAULT_RELIABILITY_BINS,
        "mce_min_bin_size": DEFAULT_MCE_MIN_BIN_SIZE,
        "oc_uc_confidence_gap": 0.1,
        "nll_epsilon": DEFAULT_NLL_EPSILON,
        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
    }


def _validate_metric_protocol(value: Any, *, context: str) -> Dict[str, Any]:
    required = {
        "schema_version",
        "ece_bins",
        "ace_bins",
        "reliability_bins",
        "mce_min_bin_size",
        "nll_epsilon",
        "bootstrap_draws",
        "bootstrap_seed",
    }
    optional = {"oc_uc_confidence_gap"}
    if not isinstance(value, Mapping) or set(value) not in (
        required,
        required | optional,
    ):
        raise EvaluationError(f"{context} metric_protocol schema mismatch")
    if value.get("schema_version") != METRIC_PROTOCOL_SCHEMA:
        raise EvaluationError(f"{context} metric_protocol version mismatch")
    for field in (
        "ece_bins",
        "ace_bins",
        "reliability_bins",
        "mce_min_bin_size",
        "bootstrap_draws",
    ):
        parsed = _strict_nonnegative_int(value.get(field), f"{context}.{field}")
        if parsed <= 0:
            raise EvaluationError(f"{context}.{field} must be positive")
    seed = value.get("bootstrap_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError(f"{context}.bootstrap_seed must be an integer")
    epsilon = value.get("nll_epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or not 0.0 < float(epsilon) < 0.5
    ):
        raise EvaluationError(f"{context}.nll_epsilon is invalid")
    threshold = value.get("oc_uc_confidence_gap", 0.1)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise EvaluationError(f"{context}.oc_uc_confidence_gap is invalid")
    normalized = dict(value)
    normalized["oc_uc_confidence_gap"] = float(threshold)
    return normalized


def _prepare_cases(
    test_path: Path,
    *,
    profile: str,
    global_cutoff: Optional[str],
    limit: Optional[int],
) -> tuple[List[Dict[str, Any]], List[str]]:
    from chain.config import legacy_yes_no_axis, validate_binary_axis
    from chain.skills.temporal_validity import parse_cutoff

    all_raw_items = [
        _canonicalize_input_object(
            item,
            context=f"{test_path.name}:{index}",
        )
        for index, item in enumerate(_load_jsonl(test_path), 1)
    ]
    source_schema_fields = _source_schema_fields(all_raw_items)
    raw_items = all_raw_items
    if limit is not None:
        if isinstance(limit, bool) or limit <= 0:
            raise EvaluationError("--limit must be positive")
        raw_items = raw_items[:limit]
    legacy_registered = test_path.name in LEGACY_EVAL_BASENAMES
    prepared: List[Dict[str, Any]] = []
    nonpublishable_reasons: List[str] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_items):
        raw_case_id = item.get("id", "")
        raw_question = item.get("question", "")
        if not isinstance(raw_case_id, str) or not isinstance(raw_question, str):
            raise EvaluationError(f"case {index} id and question must be strings")
        case_id = raw_case_id.strip()
        question = raw_question.strip()
        if not case_id or not question:
            raise EvaluationError(f"case {index} requires non-empty id and question")
        if case_id in seen_ids:
            raise EvaluationError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        raw_cutoff = item.get("cutoff", "")
        if raw_cutoff not in (None, "") and not isinstance(raw_cutoff, str):
            raise EvaluationError(f"case {case_id} cutoff must be a string")
        cutoff_text = raw_cutoff.strip() if isinstance(raw_cutoff, str) else ""
        try:
            cutoff = parse_cutoff(cutoff_text).canonical if cutoff_text else ""
        except ValueError as exc:
            raise EvaluationError(f"case {case_id} has an invalid cutoff") from exc
        cutoff_adaptation_kind = "explicit_cutoff" if cutoff else ""
        cutoff_adaptation_raw = raw_cutoff if isinstance(raw_cutoff, str) else ""





        if profile == "compat" and legacy_registered:
            raw_date = item.get("date", "")
            if raw_date not in (None, "") and not isinstance(raw_date, str):
                raise EvaluationError(f"case {case_id} date must be a string")
            date_text = raw_date.strip() if isinstance(raw_date, str) else ""
            try:
                date_cutoff = parse_cutoff(date_text).canonical if date_text else ""
            except ValueError as exc:
                raise EvaluationError(f"case {case_id} has an invalid date cutoff") from exc
            if cutoff and date_cutoff and cutoff != date_cutoff:
                raise EvaluationError(f"case {case_id} has conflicting cutoff/date values")
            if not cutoff and date_cutoff:
                cutoff = date_cutoff
                cutoff_adaptation_kind = "legacy_date_alias"
                cutoff_adaptation_raw = raw_date if isinstance(raw_date, str) else ""
                nonpublishable_reasons.append("legacy_date_cutoff_adapter")

        if not cutoff:
            if profile == "paper":
                raise EvaluationError(f"paper case {case_id} requires an explicit cutoff")
            if global_cutoff not in (None, "") and not isinstance(global_cutoff, str):
                raise EvaluationError("--cutoff must be a string")
            fallback_text = global_cutoff.strip() if isinstance(global_cutoff, str) else ""
            fallback_reason = "global_cutoff_fallback"





            historical_default = LEGACY_DEFAULT_CUTOFFS.get(test_path.name, "")
            if not fallback_text and profile == "compat" and historical_default:
                fallback_text = historical_default
                fallback_reason = "legacy_default_cutoff_adapter"
            if not fallback_text:
                raise EvaluationError(f"case {case_id} requires an explicit cutoff")
            try:
                cutoff = parse_cutoff(fallback_text).canonical
            except ValueError as exc:
                raise EvaluationError("--cutoff is invalid") from exc
            cutoff_adaptation_kind = (
                "legacy_default_cutoff"
                if fallback_reason == "legacy_default_cutoff_adapter"
                else "global_cutoff_fallback"
            )
            cutoff_adaptation_raw = fallback_text
            nonpublishable_reasons.append(fallback_reason)
        if "binary_axis" in item:
            axis = validate_binary_axis(item["binary_axis"])
            axis_adaptation_kind = "explicit_binary_axis"
            axis_source_field = "binary_axis"
        elif profile == "compat" and legacy_registered:
            axis = legacy_yes_no_axis()
            axis_adaptation_kind = "legacy_yes_no_adapter"
            axis_source_field = "registered_default"
            nonpublishable_reasons.append("legacy_yes_no_axis_adapter")
        else:
            raise EvaluationError(
                "explicit binary_axis required; legacy adapter is registered only for the eight historical basenames in compat"
            )
        if profile == "paper" and axis.get("source") == "legacy_adapter":
            raise EvaluationError("paper profile rejects the legacy binary-axis adapter")
        gold_event = _case_gold_event(item, axis)
        cutoff_source_field, cutoff_adapter_id = {
            "explicit_cutoff": ("cutoff", ""),
            "legacy_date_alias": ("date", "chain-legacy-date-cutoff-v1"),
            "global_cutoff_fallback": (
                "--cutoff",
                "chain-compat-global-cutoff-v1",
            ),
            "legacy_default_cutoff": (
                "registered_default",
                "chain-legacy-clinical-default-cutoff-v1",
            ),
        }[cutoff_adaptation_kind]
        input_adaptation = {
            "schema_version": INPUT_ADAPTATION_SCHEMA,
            "cutoff": {
                "kind": cutoff_adaptation_kind,
                "source_field": cutoff_source_field,
                "raw_value": cutoff_adaptation_raw,
                "canonical_value": cutoff,
                "adapter_id": cutoff_adapter_id,
            },
            "binary_axis": {
                "kind": axis_adaptation_kind,
                "source_field": axis_source_field,
                "adapter_id": str(axis.get("adapter_id", "")),
            },
        }
        source_record = copy.deepcopy(item)
        source_record_sha256 = _sha256_payload(source_record)
        source_record_completeness = _build_source_record_completeness(
            source_record,
            schema_fields=source_schema_fields,
        )
        case_payload = {
            "case_id": case_id,
            "question": question,
            "cutoff": cutoff,
            "binary_axis_sha256": axis["binary_axis_sha256"],
            "gold_event": gold_event,
            "domain": _optional_record_text(
                item, "domain", context=f"case {case_id}"
            ),
            "source": _optional_record_text(
                item, "source", context=f"case {case_id}"
            ),
            "source_record": source_record,
            "source_record_sha256": source_record_sha256,
            "source_record_completeness": source_record_completeness,
            "input_adaptation": input_adaptation,
        }
        prepared.append(
            {
                **case_payload,
                "binary_axis": axis,
                "case_input_sha256": _sha256_payload(case_payload),
            }
        )
    if limit is not None:
        nonpublishable_reasons.append("limited_case_selection")
    if profile == "compat":
        nonpublishable_reasons.append("compat_profile")
    return prepared, sorted(set(nonpublishable_reasons))


def _run_identity(
    *,
    config: Any,
    graph_validation: Mapping[str, Any],
    test_path: Path,
    cases: Sequence[Mapping[str, Any]],
    nonpublishable_reasons: Sequence[str],
) -> tuple[Dict[str, Any], str]:
    payload = {
        "schema_version": EVAL_RUN_SCHEMA,
        "profile_id": config.profile_id,
        "construction_config_sha256": config.construction_config_sha256,
        "scientific_config_sha256": config.scientific_config_sha256,
        "graph_bundle_sha256": graph_validation["graph_bundle_sha256"],
        "test_file_sha256": _sha256_file(test_path),
        "ordered_case_inputs_sha256": _sha256_payload([case["case_input_sha256"] for case in cases]),
        "case_count": len(cases),
        "nonpublishable_reasons": list(nonpublishable_reasons),
        "fusion_parameters": _fusion_parameters_from_config(config),
        "metric_protocol": _metric_protocol_from_config(config),
    }
    return payload, _sha256_payload(payload)


def _validate_existing_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_identity_sha256: str,
    cases_by_id: Mapping[str, Mapping[str, Any]],
    run_payload: Optional[Mapping[str, Any]] = None,
    result_path: Optional[Path] = None,
    artifact_root_override: Optional[Path] = None,
    allow_v1_report_only: bool = False,
    validate_traces: bool = True,
) -> Dict[str, Dict[str, Any]]:
    if run_payload is None:
        raise EvaluationError("resume/skip validation requires the complete run payload")
    if _sha256_payload(run_payload) != run_identity_sha256:
        raise EvaluationError("resume/skip expected run identity hash mismatch")
    run_fields = frozenset(run_payload)
    run_schema = run_payload.get("schema_version")
    if run_schema == EVAL_RUN_SCHEMA:
        if run_fields != _RUN_IDENTITY_FIELDS:
            raise EvaluationError("resume/skip v2 run identity schema mismatch")
        row_schema = EVAL_CASE_SCHEMA
        expected_fields = _CANONICAL_ROW_FIELDS
        has_fusion_closure = True
        if result_path is None:
            raise EvaluationError("v2 canonical validation requires the result path")
        _validate_metric_protocol(
            run_payload.get("metric_protocol"), context="resume/skip run"
        )
    elif run_schema == EVAL_RUN_SCHEMA_V1 and allow_v1_report_only:
        if run_fields not in {
            _RUN_IDENTITY_FIELDS_V1,
            _LEGACY_RUN_IDENTITY_FIELDS_V1,
        }:
            raise EvaluationError("v1 canonical report run identity schema mismatch")
        row_schema = EVAL_CASE_SCHEMA_V1
        expected_fields = _CANONICAL_ROW_FIELDS_V1
        has_fusion_closure = run_fields == _RUN_IDENTITY_FIELDS_V1
    elif run_schema == EVAL_RUN_SCHEMA_V1:
        raise EvaluationError("v1 canonical rows are report-only and cannot resume/skip")
    else:
        raise EvaluationError("resume/skip run identity schema mismatch")
    if has_fusion_closure:
        _validate_fusion_parameters(run_payload.get("fusion_parameters"), context="resume/skip run")
    if run_payload.get("case_count") != len(cases_by_id):
        raise EvaluationError("resume/skip run case_count does not match prepared cases")
    accepted: Dict[str, Dict[str, Any]] = {}
    observed_order: List[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise EvaluationError("resume/skip output canonical row fields mismatch")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationError("resume/skip output has invalid case identity")
        if case_id not in cases_by_id or case_id in accepted:
            raise EvaluationError("resume/skip output has unknown or duplicate case identity")
        case = cases_by_id[case_id]
        observed_order.append(case_id)
        sealed_run = row.get("run_identity")
        if (
            not isinstance(sealed_run, Mapping)
            or set(sealed_run) != run_fields
            or dict(sealed_run) != dict(run_payload)
        ):
            raise EvaluationError("resume/skip output run_identity payload mismatch")
        sealed_hash = row.get("run_identity_sha256")
        if sealed_hash != run_identity_sha256 or _sha256_payload(sealed_run) != sealed_hash:
            raise EvaluationError("resume/skip output run identity mismatch")
        if row.get("case_input_sha256") != case["case_input_sha256"]:
            raise EvaluationError("resume/skip output case input identity mismatch")
        if row.get("schema_version") != row_schema:
            raise EvaluationError("resume/skip output schema mismatch")
        for field in ("case_id", "question", "cutoff", "binary_axis_sha256", "domain", "source"):
            if row.get(field) != case[field]:
                raise EvaluationError(f"resume/skip output {field} mismatch")
        if row.get("id") != case_id:
            raise EvaluationError("resume/skip output id/case_id mismatch")
        if row.get("binary_axis") != case["binary_axis"]:
            raise EvaluationError("resume/skip output binary_axis payload mismatch")
        if isinstance(row.get("gold_event"), bool) or row.get("gold_event") != case["gold_event"]:
            raise EvaluationError("resume/skip output gold_event mismatch")
        if row_schema == EVAL_CASE_SCHEMA:
            source_record = row.get("source_record")
            if not isinstance(source_record, Mapping):
                raise EvaluationError("resume/skip output source_record must be an object")
            if source_record != case["source_record"]:
                raise EvaluationError("resume/skip output source_record mismatch")
            _validate_source_record_case_closure(
                source_record,
                case_id=case_id,
                question=case["question"],
                domain=case["domain"],
                source=case["source"],
                gold_event=case["gold_event"],
                binary_axis=case["binary_axis"],
                context="resume/skip output",
            )
            source_hash = _strict_sha256(
                row.get("source_record_sha256"),
                "resume/skip output source_record_sha256",
            )
            if (
                source_hash != case["source_record_sha256"]
                or source_hash != _sha256_payload(source_record)
            ):
                raise EvaluationError("resume/skip output source_record identity mismatch")
            completeness = _validate_source_record_completeness(
                row.get("source_record_completeness"),
                source_record=source_record,
                context="resume/skip output",
            )
            if completeness != case["source_record_completeness"]:
                raise EvaluationError(
                    "resume/skip output source_record_completeness mismatch"
                )
            adaptation = _validate_input_adaptation(
                row.get("input_adaptation"),
                source_record=source_record,
                cutoff=case["cutoff"],
                binary_axis=case["binary_axis"],
                context="resume/skip output",
            )
            if adaptation != case["input_adaptation"]:
                raise EvaluationError("resume/skip output input_adaptation mismatch")
            if row.get("case_input_sha256") != _sha256_payload(
                _case_input_payload(case)
            ):
                raise EvaluationError("resume/skip output complete case identity mismatch")
        for field in (
            "profile_id", "construction_config_sha256", "scientific_config_sha256",
            "graph_bundle_sha256",
        ):
            if row.get(field) != run_payload.get(field):
                raise EvaluationError(f"resume/skip output {field} mismatch")

        probability = _p_event(row)
        p_llm = _finite_probability(row.get("p_llm_event"), "p_llm_event")
        p_causal = _finite_probability(row.get("p_causal_event"), "p_causal_event")
        del p_llm
        gold = _gold_event(row)
        predicted_event = int(probability >= 0.5)
        if isinstance(row.get("predicted_event"), bool) or row.get("predicted_event") != predicted_event:
            raise EvaluationError("resume/skip output predicted_event mismatch")
        side = case["binary_axis"]["positive" if predicted_event else "negative"]
        gold_side = case["binary_axis"]["positive" if gold else "negative"]
        expected_confidence = probability if predicted_event else 1.0 - probability
        if row.get("pred_answer") != str(side["label"]):
            raise EvaluationError("resume/skip output pred_answer mismatch")
        if row.get("ground_truth") != str(gold_side["label"]):
            raise EvaluationError("resume/skip output ground_truth mismatch")
        if _finite_probability(row.get("confidence"), "confidence") != expected_confidence:
            raise EvaluationError("resume/skip output confidence mismatch")
        if not isinstance(row.get("correct"), bool) or row["correct"] != (predicted_event == gold):
            raise EvaluationError("resume/skip output correct mismatch")
        if not isinstance(row.get("converged"), bool):
            raise EvaluationError("resume/skip output converged must be boolean")
        expected_publishable = (
            not run_payload.get("nonpublishable_reasons")
            and run_payload.get("profile_id") == "paper"
        )
        if row.get("publishable") is not expected_publishable:
            raise EvaluationError("resume/skip output publishable mismatch")
        if row.get("collect_rounds") != 0 or row.get("collect_stats") != []:
            raise EvaluationError("resume/skip prebuilt output contains collector activity")
        causal_summary = _validate_causal_summary(
            row.get("causal_summary"),
            case=case,
            run_payload=run_payload,
            graph_bundle_sha256=str(run_payload["graph_bundle_sha256"]),
        )
        if causal_summary["p_causal_event"] != p_causal:
            raise EvaluationError("resume/skip causal summary probability mismatch")
        if has_fusion_closure:
            _validate_fusion_closure(
                probabilities=row,
                causal_summary=causal_summary,
                fusion_parameters=run_payload.get("fusion_parameters"),
                context=f"resume/skip {case_id}",
            )
        _validate_usage(row.get("usage"), context="resume/skip")
        runtime = row.get("runtime_telemetry")
        if not isinstance(runtime, Mapping) or set(runtime) != {"elapsed_seconds"}:
            raise EvaluationError("resume/skip runtime telemetry schema mismatch")
        _strict_nonnegative_float(runtime["elapsed_seconds"], "runtime_telemetry.elapsed_seconds")
        if row_schema == EVAL_CASE_SCHEMA:
            assert result_path is not None
            if validate_traces:
                _validate_prediction_trace_closure(
                    row,
                    case=case,
                    run_payload=run_payload,
                    result_path=result_path,
                    artifact_root_override=artifact_root_override,
                    context=f"resume/skip {case_id}",
                )
            else:
                _validate_prediction_trace_artifact_metadata(
                    row.get("prediction_trace_artifact"),
                    result_path=result_path,
                    artifact_root_override=artifact_root_override,
                    context=f"resume/skip {case_id}",
                )
        accepted[case_id] = dict(row)
    expected_order = [case_id for case_id in cases_by_id if case_id in accepted]
    if observed_order != expected_order:
        raise EvaluationError("resume/skip output case order mismatch")
    if row_schema == EVAL_CASE_SCHEMA and rows:
        assert result_path is not None
        _validate_artifact_bundle_manifest(
            rows,
            run_identity_sha256=run_identity_sha256,
            result_path=result_path,
            artifact_root_override=artifact_root_override,
            context="resume/skip output",
        )
    return accepted


def _load_case_id_reuse_rows(
    source_path: Path,
    *,
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:


    if source_path.is_symlink() or not source_path.is_file():
        raise EvaluationError(
            f"case-ID reuse source is missing or unsafe: {source_path}"
        )
    rows = _load_jsonl(source_path)
    reused: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EvaluationError(
                f"case-ID reuse source row {index} is not an object"
            )
        case_id = row.get("case_id", row.get("id"))
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvaluationError(
                f"case-ID reuse source row {index} has an invalid case_id"
            )
        case_id = case_id.strip()
        if case_id in reused:
            raise EvaluationError(
                f"case-ID reuse source contains duplicate case_id: {case_id}"
            )


        if case_id in cases_by_id:
            reused[case_id] = dict(row)
    if not reused:
        raise EvaluationError(
            "case-ID reuse source has no case IDs in the current test file"
        )
    return reused


def _canonical_prediction_row(
    *,
    case: Mapping[str, Any],
    prediction: Mapping[str, Any],
    config: Any,
    graph_validation: Mapping[str, Any],
    run_payload: Mapping[str, Any],
    run_identity_sha256: str,
    usage: Mapping[str, int],
    elapsed_seconds: float,
    prediction_trace_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    for field in ("p_llm_event", "p_causal_event", "p_final_event", "p_event"):
        if field not in prediction:
            raise EvaluationError(f"prediction omitted {field}")
    p_llm = _finite_probability(prediction["p_llm_event"], "p_llm_event")
    p_causal = _finite_probability(prediction["p_causal_event"], "p_causal_event")
    p_final = _finite_probability(prediction["p_final_event"], "p_final_event")
    probability = _finite_probability(prediction["p_event"], "p_event")
    probabilities = {
        "p_llm_event": p_llm,
        "p_causal_event": p_causal,
        "p_final_event": p_final,
        "p_event": probability,
    }
    if probability != p_final:
        raise EvaluationError("prediction p_event must exactly equal p_final_event")
    if prediction.get("case_id") != case["case_id"]:
        raise EvaluationError("prediction case identity mismatch")
    if prediction.get("cutoff") != case["cutoff"]:
        raise EvaluationError("prediction cutoff mismatch")
    if prediction.get("binary_axis_sha256") != case["binary_axis_sha256"]:
        raise EvaluationError("prediction binary-axis identity mismatch")
    if prediction.get("binary_axis") != case["binary_axis"]:
        raise EvaluationError("prediction binary-axis payload mismatch")
    if prediction.get("scientific_config_sha256") != config.scientific_config_sha256:
        raise EvaluationError("prediction config identity mismatch")
    if prediction.get("graph_bundle_sha256") != graph_validation["graph_bundle_sha256"]:
        raise EvaluationError("prediction graph-bundle identity mismatch")
    if prediction.get("profile_id") != config.profile_id:
        raise EvaluationError("prediction profile identity mismatch")
    collect_rounds = _strict_nonnegative_int(
        prediction.get("collect_rounds"), "prediction.collect_rounds"
    )
    collect_stats = prediction.get("collect_stats")
    if collect_rounds != 0 or collect_stats != []:
        raise EvaluationError("prebuilt evaluation requires collect_rounds=0 and collect_stats=[]")
    causal_summary = _causal_summary_from_prediction(
        prediction,
        case=case,
        config=config,
        graph_validation=graph_validation,
        probabilities=probabilities,
    )
    _validate_fusion_closure(
        probabilities=probabilities,
        causal_summary=causal_summary,
        fusion_parameters=run_payload.get("fusion_parameters"),
        context=f"prediction {case['case_id']}",
    )
    usage_payload = _validate_usage(usage, context="prediction")
    elapsed = _strict_nonnegative_float(elapsed_seconds, "runtime_telemetry.elapsed_seconds")
    predicted_event = int(probability >= 0.5)
    side = case["binary_axis"]["positive" if predicted_event else "negative"]
    gold_side = case["binary_axis"]["positive" if case["gold_event"] else "negative"]
    expected_answer = str(side["label"])
    expected_confidence = probability if predicted_event else 1.0 - probability
    if prediction.get("answer") != expected_answer:
        raise EvaluationError("prediction answer does not match p_event and binary axis")
    if _finite_probability(
        prediction.get("confidence"), "prediction.confidence"
    ) != expected_confidence:
        raise EvaluationError("prediction confidence does not match p_event")
    converged = prediction.get("converged")
    if not isinstance(converged, bool):
        raise EvaluationError("prediction.converged must be boolean")
    prediction_publishable = prediction.get("publishable")
    if not isinstance(prediction_publishable, bool):
        raise EvaluationError("prediction.publishable must be boolean")
    if prediction_publishable is not bool(config.publishable):
        raise EvaluationError("prediction publishability does not match resolved config")
    return {
        "schema_version": EVAL_CASE_SCHEMA,
        "run_identity": dict(run_payload),
        "run_identity_sha256": run_identity_sha256,
        "case_input_sha256": case["case_input_sha256"],
        "case_id": case["case_id"],
        "id": case["case_id"],
        "question": case["question"],
        "cutoff": case["cutoff"],
        "binary_axis": case["binary_axis"],
        "binary_axis_sha256": case["binary_axis_sha256"],
        "gold_event": case["gold_event"],
        "ground_truth": str(gold_side["label"]),
        "predicted_event": predicted_event,
        "pred_answer": expected_answer,
        "confidence": expected_confidence,
        "p_llm_event": p_llm,
        "p_causal_event": p_causal,
        "p_final_event": p_final,
        "p_event": probability,
        "correct": predicted_event == case["gold_event"],
        "converged": converged,
        "domain": case["domain"],
        "source": case["source"],
        "source_record": copy.deepcopy(case["source_record"]),
        "source_record_sha256": case["source_record_sha256"],
        "source_record_completeness": copy.deepcopy(
            case["source_record_completeness"]
        ),
        "input_adaptation": copy.deepcopy(case["input_adaptation"]),
        "profile_id": config.profile_id,
        "construction_config_sha256": config.construction_config_sha256,
        "scientific_config_sha256": config.scientific_config_sha256,
        "graph_bundle_sha256": graph_validation["graph_bundle_sha256"],
        "publishable": not run_payload["nonpublishable_reasons"] and config.publishable,
        "collect_rounds": collect_rounds,
        "collect_stats": [],
        "causal_summary": causal_summary,
        "usage": usage_payload,
        "runtime_telemetry": {"elapsed_seconds": elapsed},
        "prediction_trace_artifact": dict(prediction_trace_artifact),
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _operational_receipt_path(output_path: Path) -> Path:
    return Path(f"{output_path}.operational.json")


def _archive_failed_operational_receipt(receipt_path: Path) -> Optional[Path]:


    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    if receipt_path.is_symlink():
        raise EvaluationError(
            f"refusing to reuse unsafe operational receipt symlink: {receipt_path}"
        )
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"cannot inspect existing operational receipt: {receipt_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise EvaluationError(
            f"existing operational receipt is not an object: {receipt_path}"
        )
    status = payload.get("status")
    if status not in {"failed", "publication_failed"}:
        raise EvaluationError(
            f"refusing to overwrite non-retryable operational receipt "
            f"({status!r}): {receipt_path}"
        )
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    for _ in range(32):
        archived = receipt_path.with_name(
            f"{receipt_path.name}.attempt-{stamp}-{uuid.uuid4().hex[:12]}"
        )
        try:
            os.replace(receipt_path, archived)
        except FileNotFoundError:


            return None
        except OSError as exc:
            raise EvaluationError(
                f"cannot archive failed operational receipt: {receipt_path}"
            ) from exc
        return archived
    raise EvaluationError(
        f"could not allocate an archive name for operational receipt: {receipt_path}"
    )


def _archive_reused_operational_receipt(receipt_path: Path) -> Optional[Path]:


    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    if receipt_path.is_symlink():
        raise EvaluationError(
            f"refusing to reuse unsafe operational receipt symlink: {receipt_path}"
        )
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archived = receipt_path.with_name(
        f"{receipt_path.name}.case-id-reuse-{stamp}-{uuid.uuid4().hex[:12]}"
    )
    try:
        os.replace(receipt_path, archived)
    except OSError as exc:
        raise EvaluationError(
            f"cannot archive operational receipt for case-ID reuse: {receipt_path}"
        ) from exc
    return archived


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:


    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EvaluationError(
                f"refusing to overwrite operational receipt: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _usage_from_attempt(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    result: Dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = value.get(field, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        result[field] = int(raw)
    return result


def _strict_attempt_usage(value: Any) -> tuple[Dict[str, int], bool]:


    zero = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if not isinstance(value, Mapping):
        return zero, False
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(field not in value for field in required):
        return zero, False
    parsed: Dict[str, int] = {}
    for field in required:
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            return zero, False
        parsed[field] = int(raw)
    if parsed["total_tokens"] != parsed["prompt_tokens"] + parsed["completion_tokens"]:
        return zero, False
    return parsed, True


def _summarize_provider_stages(traces: Any) -> Dict[str, Any]:


    stages: Dict[str, Dict[str, Any]] = {}

    def stage_row(stage: str) -> Dict[str, Any]:
        return stages.setdefault(
            stage,
            {
                "api_calls": 0,
                "attempts_with_reported_usage": 0,
                "attempts_without_reported_usage": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "provider_elapsed_seconds": 0.0,
                "attempts_with_reported_latency": 0,
                "attempts_without_reported_latency": 0,
                "models": set(),
            },
        )

    if isinstance(traces, list):
        for wrapper in traces:
            if not isinstance(wrapper, Mapping):
                continue
            logical_stage = str(wrapper.get("stage") or "unknown")
            if wrapper.get("trace_schema_version") == "chain-provider-json-trace-v1":
                calls = wrapper.get("provider_calls", [])
            else:
                calls = [wrapper]
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                call_stage = logical_stage or str(call.get("stage") or "unknown")
                row = stage_row(call_stage)
                attempts = call.get("attempts", [])
                if not isinstance(attempts, list):
                    continue
                for attempt in attempts:
                    if not isinstance(attempt, Mapping):
                        continue
                    row["api_calls"] += 1
                    model = attempt.get("model") or call.get("model")
                    if isinstance(model, str) and model:
                        row["models"].add(model)
                    usage = attempt.get("usage")
                    parsed_usage, usage_complete = _strict_attempt_usage(usage)
                    if usage_complete:
                        row["attempts_with_reported_usage"] += 1
                    else:
                        row["attempts_without_reported_usage"] += 1
                    for field, amount in parsed_usage.items():
                        row[field] += amount
                    latency = attempt.get("elapsed_seconds")
                    if (
                        isinstance(latency, (int, float))
                        and not isinstance(latency, bool)
                        and math.isfinite(float(latency))
                        and float(latency) >= 0.0
                    ):
                        row["provider_elapsed_seconds"] += float(latency)
                        row["attempts_with_reported_latency"] += 1
                    else:
                        row["attempts_without_reported_latency"] += 1

    totals = {
        "api_calls": 0,
        "attempts_with_reported_usage": 0,
        "attempts_without_reported_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider_elapsed_seconds": 0.0,
        "attempts_with_reported_latency": 0,
        "attempts_without_reported_latency": 0,
    }
    serializable: Dict[str, Any] = {}
    for stage, row in sorted(stages.items()):
        item = {**row, "models": sorted(row["models"])}
        serializable[stage] = item
        for field in totals:
            totals[field] += item[field]
    return {
        "schema_version": PROVIDER_STAGE_USAGE_SCHEMA,
        "stages": serializable,
        "totals": totals,
    }


def _case_operational_summary(
    *,
    case: Mapping[str, Any],
    status: str,
    elapsed_seconds: float,
    usage: Mapping[str, Any],
    provider_traces: Any,
    prediction: Optional[Mapping[str, Any]] = None,
    error: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    usage_payload = _validate_usage(usage, context=f"operational case {case['case_id']}")
    stage_usage = _summarize_provider_stages(provider_traces)
    stage_totals = stage_usage["totals"]
    usage_closure = {
        "status": "closed"
        if all(
            int(stage_totals[field]) == int(usage_payload[field])
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "api_calls",
            )
        )
        else "mismatch",
        "declared": usage_payload,
        "recomputed": {
            field: int(stage_totals[field])
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "api_calls",
            )
        },
    }
    runtime = None
    if isinstance(prediction, Mapping):
        candidate = prediction.get("operational_runtime_telemetry")
        if isinstance(candidate, Mapping):
            runtime = copy.deepcopy(dict(candidate))
    result: Dict[str, Any] = {
        "case_id": case["case_id"],
        "case_input_sha256": case["case_input_sha256"],
        "status": status,
        "elapsed_seconds": _strict_nonnegative_float(
            elapsed_seconds, f"operational case {case['case_id']} elapsed_seconds"
        ),
        "usage": usage_payload,
        "provider_stage_usage": stage_usage,
        "usage_closure": usage_closure,
        "operational_runtime_telemetry": runtime,
    }
    if error is not None:
        result["error"] = dict(error)


        complete_case_input = _case_input_payload(case)





        complete_case_input["binary_axis"] = copy.deepcopy(case["binary_axis"])
        result["complete_case_input"] = complete_case_input
    return result


def _sum_case_usage(cases: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }
    for case in cases:
        usage = case.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for field in totals:
            totals[field] += int(usage.get(field, 0))
    return totals


def _evaluation_operational_receipt(
    *,
    status: str,
    output_path: Path,
    test_path: Path,
    run_identity_sha256: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    initialization_seconds: float,
    case_execution_seconds: float,
    publication_seconds: float,
    workers: int,
    publication_validation: str = "full",
    cases: Sequence[Mapping[str, Any]],
    result_path: Optional[Path] = None,
    artifact_bundle: Optional[Path] = None,
    errors: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if publication_validation not in {"full", "fast"}:
        raise EvaluationError(
            "operational receipt publication_validation must be full or fast"
        )
    succeeded = sum(case.get("status") in {"ok", "resumed"} for case in cases)
    failed = sum(case.get("status") == "failed" for case in cases)
    receipt: Dict[str, Any] = {
        "schema_version": EVAL_OPERATIONAL_RECEIPT_SCHEMA,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": round(float(elapsed_seconds), 9),
        "initialization_seconds": round(float(initialization_seconds), 9),
        "case_execution_seconds": round(float(case_execution_seconds), 9),
        "publication_seconds": round(float(publication_seconds), 9),
        "workers": int(workers),
        "publication_validation": publication_validation,
        "test_file": str(test_path),
        "output": str(output_path),
        "run_identity_sha256": run_identity_sha256,
        "case_counts": {
            "total": len(cases),
            "succeeded": succeeded,
            "failed": failed,
        },
        "provider_usage": _sum_case_usage(cases),
        "cases": [copy.deepcopy(dict(case)) for case in cases],
        "errors": list(errors or []),
        "monetary_cost": {
            "status": "not_configured",
            "currency": None,
            "known_total": None,
            "reason": (
                "No versioned pricing snapshot was supplied to offline analysis; "
                "exact provider/model token usage is retained without inventing rates."
            ),
        },
    }
    if result_path is not None and result_path.is_file():
        receipt["result"] = {
            "path": str(result_path),
            "sha256": _sha256_file(result_path),
            "bytes": result_path.stat().st_size,
        }
    if artifact_bundle is not None and artifact_bundle.is_dir():
        receipt["artifact_bundle"] = str(artifact_bundle)
    return receipt


def _ensure_reused_operational_receipt(
    *,
    output_path: Path,
    test_path: Path,
    run_identity_sha256: str,
    run_payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    workers: int,
    publication_validation: str = "full",
) -> None:


    receipt_path = _operational_receipt_path(output_path)
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink():
            raise EvaluationError(
                f"refusing to reuse unsafe operational receipt symlink: {receipt_path}"
            )
        try:
            receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"cannot inspect existing operational receipt: {receipt_path}"
            ) from exc
        if not isinstance(receipt_payload, Mapping):
            raise EvaluationError(
                f"existing operational receipt is not an object: {receipt_path}"
            )
        status = receipt_payload.get("status")
        if status in {"success", "resumed"}:
            return
        if status not in {"failed", "publication_failed"}:
            raise EvaluationError(
                f"refusing to overwrite non-retryable operational receipt "
                f"({status!r}): {receipt_path}"
            )
        _archive_failed_operational_receipt(receipt_path)
    operational_cases: list[dict[str, Any]] = []
    elapsed_total = 0.0
    for case in cases:
        row = rows_by_id.get(str(case["case_id"]))
        if not isinstance(row, Mapping):
            raise EvaluationError(
                f"cannot backfill operational receipt; missing row {case['case_id']}"
            )
        prediction = _load_prediction_trace(
            row.get("prediction_trace_artifact"),
            result_path=output_path,
            context=f"reuse receipt {case['case_id']}",
        )
        runtime = row.get("runtime_telemetry")
        elapsed = (
            float(runtime.get("elapsed_seconds", 0.0))
            if isinstance(runtime, Mapping)
            else 0.0
        )
        elapsed_total += elapsed
        operational_cases.append(
            _case_operational_summary(
                case=case,
                status="resumed",
                elapsed_seconds=elapsed,
                usage=row.get("usage", {}),
                provider_traces=prediction.get("provider_traces", []),
                prediction=prediction,
            )
        )
    now = _utc_now()
    receipt = _evaluation_operational_receipt(
        status="resumed",
        output_path=output_path,
        test_path=test_path,
        run_identity_sha256=run_identity_sha256,
        started_at_utc=now,
        finished_at_utc=now,
        elapsed_seconds=elapsed_total,
        initialization_seconds=0.0,
        case_execution_seconds=elapsed_total,
        publication_seconds=0.0,
        workers=workers,
        publication_validation=publication_validation,
        cases=operational_cases,
        result_path=output_path,
        errors=[],
    )
    _atomic_create_json(receipt_path, receipt)


def _atomic_publish_jsonl(output_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent))
    staging_file = staging_dir / output_path.name
    try:
        with staging_file.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging_file, output_path)
        try:
            directory_fd = os.open(output_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:


    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _new_artifact_bundle_path(output_path: Path, run_identity_sha256: str) -> Path:


    suffix = uuid.uuid4().hex
    return output_path.parent / (
        f"{output_path.name}.artifacts-{run_identity_sha256}-{suffix}"
    )


def _checkpoint_paths(
    output_path: Path, run_identity_sha256: str
) -> tuple[Path, Path]:


    checkpoint_index = Path(f"{output_path}.checkpoint.jsonl")


    checkpoint_bundle = output_path.parent / (
        f"{output_path.name}.artifacts-{run_identity_sha256}-checkpoint"
    )
    return checkpoint_index, checkpoint_bundle


def _discover_checkpoint_identity(
    output_path: Path,
) -> Optional[tuple[Path, Path, Dict[str, Any], str]]:


    checkpoint_index = Path(f"{output_path}.checkpoint.jsonl")
    if not checkpoint_index.exists():
        return None
    if checkpoint_index.is_symlink() or not checkpoint_index.is_file():
        raise EvaluationError("evaluation checkpoint index is unsafe")
    rows = _load_jsonl(checkpoint_index)
    if not rows:
        raise EvaluationError("evaluation checkpoint is empty")
    first = rows[0]
    run_payload = first.get("run_identity")
    if not isinstance(run_payload, Mapping):
        raise EvaluationError("evaluation checkpoint is missing a sealed run identity")
    run_payload = dict(run_payload)
    run_hash = _strict_sha256(
        first.get("run_identity_sha256"),
        "evaluation checkpoint run_identity_sha256",
    )
    if _sha256_payload(run_payload) != run_hash:
        raise EvaluationError("evaluation checkpoint run identity hash mismatch")
    artifact = first.get("prediction_trace_artifact")
    if not isinstance(artifact, Mapping):
        raise EvaluationError("evaluation checkpoint is missing a trace artifact")
    relative_path = artifact.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise EvaluationError("evaluation checkpoint trace path is invalid")
    parts = Path(relative_path).parts
    if not parts or parts[0] in {"", ".", ".."} or Path(relative_path).is_absolute():
        raise EvaluationError("evaluation checkpoint trace path is unsafe")
    bundle_name = parts[0]
    expected_prefix = f"{output_path.name}.artifacts-"
    if not (
        bundle_name.startswith(expected_prefix)
        and bundle_name.endswith("-checkpoint")
    ):
        raise EvaluationError("evaluation checkpoint trace bundle does not match output")
    checkpoint_bundle = output_path.parent / bundle_name
    if checkpoint_bundle.is_symlink() or not checkpoint_bundle.is_dir():
        raise EvaluationError("evaluation checkpoint artifact bundle is missing or unsafe")
    return checkpoint_index, checkpoint_bundle, run_payload, run_hash


def _validate_resume_identity_compatibility(
    current_payload: Mapping[str, Any],
    sealed_payload: Mapping[str, Any],
) -> None:


    if set(current_payload) != set(sealed_payload):
        raise EvaluationError("evaluation checkpoint run identity schema differs from current run")
    for field in current_payload:
        if current_payload.get(field) != sealed_payload.get(field):
            raise EvaluationError(
                "evaluation checkpoint differs from current run in "
                f"identity field: {field}"
            )


def _checkpoint_prune_orphans(
    checkpoint_bundle: Path,
    *,
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:


    if checkpoint_bundle.is_symlink():
        raise EvaluationError("checkpoint artifact bundle is unsafe")
    traces_root = checkpoint_bundle / "traces"
    if traces_root.is_symlink():
        raise EvaluationError("checkpoint trace directory is unsafe")
    if not checkpoint_bundle.is_dir() or not traces_root.is_dir():
        raise EvaluationError("checkpoint artifact bundle is missing or unsafe")
    expected_names: set[str] = set()
    for index, row in enumerate(rows):
        artifact, trace_path = _validated_trace_path(
            row.get("prediction_trace_artifact"),
            result_path=output_path,
            artifact_root_override=checkpoint_bundle,
            context=f"checkpoint row {index}",
        )
        del artifact
        expected_names.add(trace_path.name)
    try:
        entries = list(traces_root.iterdir())
    except OSError as exc:
        raise EvaluationError("checkpoint trace directory is unreadable") from exc
    for path in entries:
        if path.is_symlink() or path.is_dir():
            raise EvaluationError("checkpoint trace directory contains unsafe entries")
        if not path.is_file():
            raise EvaluationError("checkpoint trace directory contains invalid entries")
        if path.name not in expected_names:


            if not path.name.endswith(".json.gz"):
                raise EvaluationError("checkpoint trace directory contains unknown files")
            path.unlink()
    _fsync_directory(traces_root)


def _checkpoint_write_snapshot(
    checkpoint_index: Path,
    checkpoint_bundle: Path,
    *,
    output_path: Path,
    run_identity_sha256: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:


    if not rows:
        return








    _write_artifact_bundle_manifest(
        checkpoint_bundle,
        run_identity_sha256=run_identity_sha256,
        rows=rows,
        replace_existing=True,
    )
    _atomic_publish_jsonl(checkpoint_index, rows)


def _load_checkpoint_rows(
    checkpoint_index: Path,
    checkpoint_bundle: Path,
    *,
    output_path: Path,
    run_identity_sha256: str,
    cases_by_id: Mapping[str, Mapping[str, Any]],
    run_payload: Mapping[str, Any],
    validate_traces: bool = True,
) -> Dict[str, Dict[str, Any]]:


    index_exists = checkpoint_index.is_file()
    bundle_exists = checkpoint_bundle.exists()
    if not index_exists and not bundle_exists:
        return {}
    if index_exists != (bundle_exists and checkpoint_bundle.is_dir()):
        raise EvaluationError(
            "evaluation checkpoint is incomplete; keep both checkpoint.jsonl "
            "and its artifact bundle or remove them before restarting"
        )
    if checkpoint_index.is_symlink() or checkpoint_bundle.is_symlink():
        raise EvaluationError("evaluation checkpoint is unsafe")
    rows = _load_jsonl(checkpoint_index)
    if not rows:
        return {}


    _checkpoint_prune_orphans(
        checkpoint_bundle, output_path=output_path, rows=rows
    )
    _write_artifact_bundle_manifest(
        checkpoint_bundle,
        run_identity_sha256=run_identity_sha256,
        rows=rows,
        replace_existing=True,
    )
    return _validate_existing_rows(
        rows,
        run_identity_sha256=run_identity_sha256,
        cases_by_id=cases_by_id,
        run_payload=run_payload,
        result_path=output_path,
        artifact_root_override=checkpoint_bundle,
        validate_traces=validate_traces,
    )


def _remove_checkpoint(
    checkpoint_index: Path, checkpoint_bundle: Path
) -> None:


    for path in (checkpoint_index,):
        if path.is_symlink():
            raise EvaluationError(f"refusing to remove unsafe checkpoint symlink: {path}")
        path.unlink(missing_ok=True)
    if checkpoint_bundle.is_symlink():
        raise EvaluationError(
            f"refusing to remove unsafe checkpoint bundle symlink: {checkpoint_bundle}"
        )
    if checkpoint_bundle.exists():
        if not checkpoint_bundle.is_dir():
            raise EvaluationError(
                f"refusing to remove non-directory checkpoint bundle: {checkpoint_bundle}"
            )
        shutil.rmtree(checkpoint_bundle)


def _write_artifact_bundle_manifest(
    staging_bundle: Path,
    *,
    run_identity_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    replace_existing: bool = False,
) -> None:
    manifest = {
        "schema_version": ARTIFACT_BUNDLE_SCHEMA,
        "run_identity_sha256": run_identity_sha256,
        "case_count": len(rows),
        "prediction_traces": [
            {
                "case_id": row["case_id"],
                "artifact": row["prediction_trace_artifact"],
            }
            for row in rows
        ],
    }
    encoded = _canonical_json(manifest).encode("utf-8")
    manifest_path = staging_bundle / "manifest.json"
    if replace_existing:
        _atomic_replace_bytes(manifest_path, encoded)
    else:
        with manifest_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    try:
        observed = manifest_path.read_bytes()
    except OSError as exc:
        raise EvaluationError("cannot reread staged artifact bundle manifest") from exc
    if observed != encoded:
        raise EvaluationError("staged artifact bundle manifest verification failed")
    _strict_json_line(observed.decode("utf-8"), context="artifact bundle manifest")
    _fsync_directory(staging_bundle / "traces")
    _fsync_directory(staging_bundle)


def _validate_artifact_bundle_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_identity_sha256: str,
    result_path: Path,
    artifact_root_override: Optional[Path] = None,
    context: str,
) -> None:
    if not rows:
        raise EvaluationError(f"{context} artifact bundle cannot be empty")
    bundle_names: set[str] = set()
    expected_trace_names: set[str] = set()
    expected_entries: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        artifact, trace_path = _validated_trace_path(
            row.get("prediction_trace_artifact"),
            result_path=result_path,
            artifact_root_override=artifact_root_override,
            context=f"{context} row {index}",
        )
        parts = artifact["relative_path"].split("/")
        bundle_names.add(parts[0])
        if trace_path.name in expected_trace_names:
            raise EvaluationError(f"{context} artifact bundle reuses a trace path")
        expected_trace_names.add(trace_path.name)
        expected_entries.append(
            {
                "case_id": row.get("case_id"),
                "artifact": artifact,
            }
        )
    if len(bundle_names) != 1:
        raise EvaluationError(f"{context} rows reference multiple artifact bundles")
    bundle_name = next(iter(bundle_names))
    if artifact_root_override is None:
        bundle_candidate = result_path.parent / bundle_name
        if bundle_candidate.is_symlink():
            raise EvaluationError(f"{context} artifact bundle directory is unsafe")
        bundle_root = bundle_candidate.resolve()
    else:
        if artifact_root_override.is_symlink():
            raise EvaluationError(f"{context} staged artifact bundle directory is unsafe")
        bundle_root = artifact_root_override.resolve()
    traces_root = bundle_root / "traces"
    if (
        not bundle_root.is_dir()
        or bundle_root.is_symlink()
        or not traces_root.is_dir()
        or traces_root.is_symlink()
    ):
        raise EvaluationError(f"{context} artifact bundle directory is missing or unsafe")
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvaluationError(f"{context} artifact bundle manifest is missing or unsafe")
    try:
        bundle_root_entries = {path.name for path in bundle_root.iterdir()}
    except OSError as exc:
        raise EvaluationError(f"{context} artifact bundle directory is unreadable") from exc
    if bundle_root_entries != {"manifest.json", "traces"}:
        raise EvaluationError(f"{context} artifact bundle root closure mismatch")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest_text = manifest_raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"{context} artifact bundle manifest is unreadable") from exc
    manifest = _strict_json_line(
        manifest_text,
        context=f"{context} artifact bundle manifest",
    )
    expected_manifest = {
        "schema_version": ARTIFACT_BUNDLE_SCHEMA,
        "run_identity_sha256": run_identity_sha256,
        "case_count": len(rows),
        "prediction_traces": expected_entries,
    }
    if manifest != expected_manifest:
        raise EvaluationError(f"{context} artifact bundle manifest closure mismatch")
    if _canonical_json(manifest).encode("utf-8") != manifest_raw:
        raise EvaluationError(f"{context} artifact bundle manifest is not canonical JSON")
    try:
        observed_entries = list(traces_root.iterdir())
    except OSError as exc:
        raise EvaluationError(f"{context} artifact trace directory is unreadable") from exc
    if (
        any(not path.is_file() or path.is_symlink() for path in observed_entries)
        or {path.name for path in observed_entries} != expected_trace_names
    ):
        raise EvaluationError(f"{context} artifact trace directory closure mismatch")


def run_evaluation(
    test_file: str,
    output_file: Optional[str],
    limit: Optional[int] = None,
    resume: bool = False,
    cutoff: Optional[str] = None,
    *,
    profile: str = "compat",
    graph_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    skip_if_done: bool = False,
    resolved_config: Any = None,
    _config_resolver: Optional[Callable[..., Any]] = None,
    _bundle_validator: Optional[Callable[..., Mapping[str, Any]]] = None,
    _agent_factory: Optional[Callable[..., Any]] = None,
    runtime_workers: Optional[int] = None,
    publication_validation: str = "full",
    resume_by_case_id: bool = False,
    reuse_case_file: Optional[str] = None,
) -> str:


    run_started_monotonic = time.monotonic()
    run_started_at_utc = _utc_now()

    from chain.config import PAPER_PROFILE, ResolvedConfig, resolve_config
    from chain.graph.completion import load_graph_bundle_portable

    if profile not in {"compat", PAPER_PROFILE}:
        raise EvaluationError("profile must be compat or paper")
    if profile == PAPER_PROFILE and any(
        seam is not None for seam in (_config_resolver, _bundle_validator, _agent_factory)
    ):
        raise EvaluationError(
            "paper profile forbids _config_resolver, _bundle_validator, and _agent_factory seams"
        )
    if profile == PAPER_PROFILE:
        if not graph_dir or not config_path or not output_file:
            raise EvaluationError("paper prediction requires --graph-dir, --config, and --output")
        if resume or skip_if_done:
            raise EvaluationError("paper profile prohibits --resume and --skip-if-done")



    if resume_by_case_id:
        resume = True
    if resume and skip_if_done:
        raise EvaluationError("--resume and --skip-if-done are mutually exclusive")
    if resume_by_case_id and profile == PAPER_PROFILE:
        raise EvaluationError("paper profile prohibits --resume-by-case-id")
    if resume_by_case_id and skip_if_done:
        raise EvaluationError(
            "--resume-by-case-id and --skip-if-done are mutually exclusive"
        )
    if reuse_case_file is not None and not resume_by_case_id:
        raise EvaluationError("--reuse-case-file requires --resume-by-case-id")
    if resolved_config is not None and _config_resolver is not None:
        raise EvaluationError("resolved_config and _config_resolver are mutually exclusive")
    if resolved_config is None:
        resolver = _config_resolver or resolve_config
        config = resolver(profile, config_path, env=os.environ)
    else:
        config = resolved_config
    if not isinstance(config, ResolvedConfig):
        raise EvaluationError("config resolver did not return ResolvedConfig")
    if config.profile != profile:
        raise EvaluationError("resolved config profile does not match evaluation profile")
    if publication_validation not in {"full", "fast"}:
        raise EvaluationError("publication_validation must be full or fast")
    if profile == PAPER_PROFILE and publication_validation != "full":
        raise EvaluationError("paper evaluation requires full publication validation")




    if runtime_workers is None:
        configured_workers = max(1, int(config.chain_eval_workers))




        effective_workers = (
            min(configured_workers, DEFAULT_RUNTIME_WORKERS)
            if profile == "compat"
            else configured_workers
        )
    else:
        if (
            isinstance(runtime_workers, bool)
            or not isinstance(runtime_workers, int)
            or runtime_workers <= 0
        ):
            raise EvaluationError("runtime_workers must be a positive integer")
        effective_workers = int(runtime_workers)
    if profile == PAPER_PROFILE and not config.publishable:
        raise EvaluationError("paper prediction requires a real locked environment seal")
    resolved_graph_dir = str(graph_dir or config.hypergraph_dir).strip()
    if not resolved_graph_dir:
        raise EvaluationError("prediction requires a graph directory")



    if _bundle_validator is None:
        graph_validation = dict(
            load_graph_bundle_portable(
                resolved_graph_dir,
                expected_config=config,
                expected_profile_id=config.profile_id,
                require_publishable=config.publishable,
            )
        )
    else:

        graph_validation = dict(
            _bundle_validator(
                resolved_graph_dir,
                expected_profile_id=config.profile_id,
                expected_config=config,
                require_publishable=config.publishable,
            )
        )
    test_path = Path(test_file)
    cases, nonpublishable_reasons = _prepare_cases(test_path, profile=profile, global_cutoff=cutoff, limit=limit)
    run_payload, run_identity_sha256 = _run_identity(
        config=config,
        graph_validation=graph_validation,
        test_path=test_path,
        cases=cases,
        nonpublishable_reasons=nonpublishable_reasons,
    )
    cases_by_id = {case["case_id"]: case for case in cases}
    if output_file is None:
        output_path = _default_output_path(
            str(test_path), config.llm_model, config.tvf_enabled
        )
    else:
        output_path = Path(output_file)
    existing: Dict[str, Dict[str, Any]] = {}
    external_reuse: Dict[str, Dict[str, Any]] = {}
    external_reuse_path: Optional[Path] = None
    if resume_by_case_id:
        external_reuse_path = Path(reuse_case_file) if reuse_case_file else output_path




        if not external_reuse_path.exists() and external_reuse_path.resolve() == output_path.resolve():



            print(
                f"[EVAL] ID reuse source not published yet; evaluating all "
                f"pending cases for {output_path}",
                file=sys.stderr,
                flush=True,
            )
            external_reuse = {}
        else:
            external_reuse = _load_case_id_reuse_rows(
                external_reuse_path,
                cases_by_id=cases_by_id,
            )
        if external_reuse_path.resolve() == output_path.resolve() and output_path.is_file():
            _archive_reused_operational_receipt(
                _operational_receipt_path(output_path)
            )
    checkpoint_index, checkpoint_bundle = _checkpoint_paths(
        output_path, run_identity_sha256
    )







    if resume and profile == "compat":
        discovered_checkpoint = _discover_checkpoint_identity(output_path)
        if discovered_checkpoint is not None:
            (
                checkpoint_index,
                checkpoint_bundle,
                sealed_payload,
                sealed_hash,
            ) = discovered_checkpoint
            _validate_resume_identity_compatibility(run_payload, sealed_payload)
            run_payload = sealed_payload
            run_identity_sha256 = sealed_hash





    if resume and not output_path.is_file():
        _archive_failed_operational_receipt(
            _operational_receipt_path(output_path)
        )
    if resume and output_path.is_file() and not resume_by_case_id:
        existing = _validate_existing_rows(
            _load_jsonl(output_path),
            run_identity_sha256=run_identity_sha256,
            cases_by_id=cases_by_id,
            run_payload=run_payload,
            result_path=output_path,
            validate_traces=publication_validation == "full",
        )
        if set(existing) == set(cases_by_id):
            _ensure_reused_operational_receipt(
                output_path=output_path,
                test_path=test_path,
                run_identity_sha256=run_identity_sha256,
                run_payload=run_payload,
                cases=cases,
                rows_by_id=existing,
                workers=effective_workers,
                publication_validation=publication_validation,
            )
            if checkpoint_index.exists() or checkpoint_bundle.exists():
                try:
                    _remove_checkpoint(checkpoint_index, checkpoint_bundle)
                except EvaluationError as cleanup_error:
                    print(
                        f"[EVAL] checkpoint cleanup deferred: {cleanup_error}",
                        file=sys.stderr,
                        flush=True,
                    )
            return str(output_path)



        _archive_failed_operational_receipt(
            _operational_receipt_path(output_path)
        )
    checkpoint_present = checkpoint_index.exists() or checkpoint_bundle.exists()
    if checkpoint_present and not resume:
        raise EvaluationError(
            "an evaluation checkpoint exists; rerun with --resume to continue "
            f"or inspect {checkpoint_index}"
        )
    checkpoint_rows: Dict[str, Dict[str, Any]] = {}
    if resume and checkpoint_present:
        checkpoint_rows = _load_checkpoint_rows(
            checkpoint_index,
            checkpoint_bundle,
            output_path=output_path,
            run_identity_sha256=run_identity_sha256,
            cases_by_id=cases_by_id,
            run_payload=run_payload,
            validate_traces=publication_validation == "full",
        )
        print(
            f"[EVAL] checkpoint loaded: {len(checkpoint_rows)}/{len(cases_by_id)} "
            f"cases from {checkpoint_index}",
            file=sys.stderr,
            flush=True,
        )
        overlap = set(existing).intersection(checkpoint_rows)
        if overlap:
            raise EvaluationError(
                "evaluation checkpoint overlaps published rows: "
                + ", ".join(sorted(overlap))
            )
    if skip_if_done:
        candidates = [output_path] if output_path.is_file() else sorted(
            output_path.parent.glob(f"{test_path.stem}_*.jsonl"),
            key=lambda path: path.as_posix(),
        )
        for candidate in candidates:
            try:
                complete = _validate_existing_rows(
                    _load_jsonl(candidate),
                    run_identity_sha256=run_identity_sha256,
                    cases_by_id=cases_by_id,
                    run_payload=run_payload,
                    result_path=candidate,
                    validate_traces=publication_validation == "full",
                )
            except EvaluationError:
                continue
            if set(complete) == set(cases_by_id):
                _ensure_reused_operational_receipt(
                    output_path=candidate,
                    test_path=test_path,
                    run_identity_sha256=run_identity_sha256,
                    run_payload=run_payload,
                    cases=cases,
                    rows_by_id=complete,
                    workers=effective_workers,
                    publication_validation=publication_validation,
                )
                return str(candidate)

    if _agent_factory is None:
        from chain.agent import CHAINAgent
        agent_factory = CHAINAgent
    else:
        agent_factory = _agent_factory
    agent = agent_factory(hypergraph_dir=resolved_graph_dir, config=config, graph_validation=graph_validation)
    initialization_seconds = time.monotonic() - run_started_monotonic
    reused_case_ids = set(existing) | set(checkpoint_rows) | set(external_reuse)
    pending = [case for case in cases if case["case_id"] not in reused_case_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_bundle = _new_artifact_bundle_path(output_path, run_identity_sha256)
    staging_bundle = Path(
        tempfile.mkdtemp(
            prefix=f".{final_bundle.name}.staging-", dir=output_path.parent
        )
    )
    checkpoint_bundle.mkdir(parents=True, exist_ok=True)
    (checkpoint_bundle / "traces").mkdir(parents=True, exist_ok=True)
    bundle_published = False
    case_indexes = {case["case_id"]: index for index, case in enumerate(cases)}
    completed: Dict[str, Dict[str, Any]] = {}
    case_operational: Dict[str, Dict[str, Any]] = {}
    case_execution_seconds = 0.0
    publication_seconds = 0.0
    try:



        for case in cases:
            case_id = case["case_id"]
            if case_id not in reused_case_ids:
                continue
            if case_id in existing:
                old_row = existing[case_id]
                source_artifact_root = None
                source_result_path = output_path
                old_prediction = _load_prediction_trace(
                    old_row["prediction_trace_artifact"],
                    result_path=source_result_path,
                    artifact_root_override=source_artifact_root,
                    context=f"resume copy {case_id}",
                )
                artifact = _link_prediction_trace(
                    old_row["prediction_trace_artifact"],
                    source_bundle=source_artifact_root,
                    destination_bundle=staging_bundle,
                    destination_bundle_name=final_bundle.name,
                    result_path=source_result_path,
                    case_index=case_indexes[case_id],
                    case_id=case_id,
                )
                copied_row = dict(old_row)
                copied_row["prediction_trace_artifact"] = artifact
                completed[case_id] = copied_row
                case_operational[case_id] = _case_operational_summary(
                    case=case,
                    status="resumed",
                    elapsed_seconds=float(
                        old_row.get("runtime_telemetry", {}).get("elapsed_seconds", 0.0)
                    ),
                    usage=old_row.get("usage", {}),
                    provider_traces=old_prediction.get("provider_traces", []),
                    prediction=old_prediction,
                )
                continue
            elif case_id in checkpoint_rows:
                old_row = checkpoint_rows[case_id]
                source_artifact_root = checkpoint_bundle
                source_result_path = output_path
                old_prediction = _load_prediction_trace(
                    old_row["prediction_trace_artifact"],
                    result_path=source_result_path,
                    artifact_root_override=source_artifact_root,
                    context=f"resume copy {case_id}",
                )
                artifact = _link_prediction_trace(
                    old_row["prediction_trace_artifact"],
                    source_bundle=source_artifact_root,
                    destination_bundle=staging_bundle,
                    destination_bundle_name=final_bundle.name,
                    result_path=source_result_path,
                    case_index=case_indexes[case_id],
                    case_id=case_id,
                )
                copied_row = dict(old_row)
                copied_row["prediction_trace_artifact"] = artifact
                completed[case_id] = copied_row
                case_operational[case_id] = _case_operational_summary(
                    case=case,
                    status="resumed",
                    elapsed_seconds=float(
                        old_row.get("runtime_telemetry", {}).get("elapsed_seconds", 0.0)
                    ),
                    usage=old_row.get("usage", {}),
                    provider_traces=old_prediction.get("provider_traces", []),
                    prediction=old_prediction,
                )
                continue
            else:





                old_row = external_reuse[case_id]
                if external_reuse_path is None:
                    raise EvaluationError("case-ID reuse source path is missing")
                source_artifact_root = None
                source_result_path = external_reuse_path
                old_prediction = _load_prediction_trace(
                    old_row.get("prediction_trace_artifact"),
                    result_path=source_result_path,
                    artifact_root_override=source_artifact_root,
                    context=f"case-ID reuse {case_id}",
                )



                old_prediction = dict(old_prediction)




                usage = old_row.get("usage")
                if not isinstance(usage, Mapping):
                    raise EvaluationError(
                        f"case-ID reuse {case_id} has invalid usage telemetry"
                    )
                runtime = old_row.get("runtime_telemetry")
                elapsed_seconds = (
                    runtime.get("elapsed_seconds", 0.0)
                    if isinstance(runtime, Mapping)
                    else 0.0
                )
                artifact = _write_prediction_trace(
                    old_prediction,
                    staging_bundle=staging_bundle,
                    bundle_name=final_bundle.name,
                    case_index=case_indexes[case_id],
                    case_id=case_id,
                )
                copied_row = _canonical_prediction_row(
                    case=case,
                    prediction=old_prediction,
                    config=config,
                    graph_validation=graph_validation,
                    run_payload=run_payload,
                    run_identity_sha256=run_identity_sha256,
                    usage=usage,
                    elapsed_seconds=elapsed_seconds,
                    prediction_trace_artifact=artifact,
                )
                completed[case_id] = copied_row
                case_operational[case_id] = _case_operational_summary(
                    case=case,
                    status="resumed",
                    elapsed_seconds=float(elapsed_seconds),
                    usage=usage,
                    provider_traces=old_prediction.get("provider_traces", []),
                    prediction=old_prediction,
                )
                continue

        def evaluate_case(
            case: Mapping[str, Any],
        ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
            from chain.llm import get_provider_traces, get_usage, reset_usage

            reset_usage()
            started = time.monotonic()
            try:
                prediction = agent.predict(
                    case["question"],
                    time_cutoff=case["cutoff"],
                    skip_collect=True,
                    case_id=case["case_id"],
                    binary_axis=case["binary_axis"],
                    inference_config=config,
                )
                elapsed_seconds = time.monotonic() - started
                usage = get_usage()
                if not isinstance(prediction, Mapping):
                    raise EvaluationError("agent prediction must be an object")




                prediction = dict(prediction)










                provider_traces = get_provider_traces()
                if not isinstance(provider_traces, list):
                    raise EvaluationError("provider trace collector returned a non-list")
                prediction["provider_traces"] = provider_traces
                operational = _case_operational_summary(
                    case=case,
                    status="ok",
                    elapsed_seconds=elapsed_seconds,
                    usage=usage,
                    provider_traces=provider_traces,
                    prediction=prediction,
                )
                artifact = _write_prediction_trace(
                    prediction,
                    staging_bundle=staging_bundle,
                    bundle_name=final_bundle.name,
                    case_index=case_indexes[case["case_id"]],
                    case_id=case["case_id"],
                )
                row = _canonical_prediction_row(
                    case=case,
                    prediction=prediction,
                    config=config,
                    graph_validation=graph_validation,
                    run_payload=run_payload,
                    run_identity_sha256=run_identity_sha256,
                    usage=usage,
                    elapsed_seconds=elapsed_seconds,
                    prediction_trace_artifact=artifact,
                )




                del prediction
                return case["case_id"], row, operational
            except Exception as exc:
                elapsed_seconds = time.monotonic() - started
                usage = get_usage()
                provider_traces = get_provider_traces()
                operational = _case_operational_summary(
                    case=case,
                    status="failed",
                    elapsed_seconds=elapsed_seconds,
                    usage=usage,
                    provider_traces=provider_traces,
                    error={
                        "type": type(exc).__name__,
                        "message": " ".join(str(exc).split())[:1000],
                    },
                )
                raise _CaseEvaluationFailure(
                    case_id=case["case_id"],
                    original=exc,
                    operational=operational,
                ) from exc

        errors: List[tuple[int, str]] = []
        workers = effective_workers
        total_cases = len(cases)
        resumed_count = len(reused_case_ids)
        pending_total = len(pending)
        progress_done = resumed_count
        progress_ok = resumed_count
        progress_failed = 0
        progress_started = time.monotonic()

        def report_progress(*, case_id: Optional[str], status: str) -> None:


            width = 24
            fraction = 1.0 if total_cases == 0 else progress_done / total_cases
            filled = min(width, max(0, int(width * fraction)))
            bar = "=" * filled + "-" * (width - filled)
            elapsed = time.monotonic() - progress_started
            suffix = "" if case_id is None else f" case_id={case_id!r}"
            print(
                f"[EVAL] [{bar}] {progress_done}/{total_cases} "
                f"({fraction * 100.0:.1f}%) ok={progress_ok} "
                f"failed={progress_failed} elapsed={elapsed:.1f}s "
                f"remaining={max(0, total_cases - progress_done)} "
                f"status={status}{suffix}",
                file=sys.stderr,
                flush=True,
            )

        print(
            f"[EVAL] cases={total_cases} resumed={resumed_count} "
            f"pending={pending_total} workers={workers}",
            file=sys.stderr,
            flush=True,
        )
        report_progress(
            case_id=None,
            status="already_complete" if pending_total == 0 else "started",
        )
        case_execution_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:





            pending_iter = iter(pending)
            futures = {}

            def submit_next() -> bool:
                try:
                    next_case = next(pending_iter)
                except StopIteration:
                    return False
                futures[pool.submit(evaluate_case, next_case)] = next_case
                return True

            for _ in range(min(workers, pending_total)):
                submit_next()
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    case = futures.pop(future)
                    status = "interrupted"
                    try:
                        case_id, row, operational = future.result()
                        case_operational[case_id] = operational
                        checkpoint_artifact = _link_prediction_trace(
                            row["prediction_trace_artifact"],
                            source_bundle=staging_bundle,
                            destination_bundle=checkpoint_bundle,
                            destination_bundle_name=checkpoint_bundle.name,
                            result_path=output_path,
                            case_index=case_indexes[case_id],
                            case_id=case_id,
                        )
                        checkpoint_row = dict(row)
                        checkpoint_row["prediction_trace_artifact"] = checkpoint_artifact
                        checkpoint_rows[case_id] = checkpoint_row
                        ordered_checkpoint_rows = [
                            checkpoint_rows[item["case_id"]]
                            for item in cases
                            if item["case_id"] in checkpoint_rows
                        ]
                        _checkpoint_write_snapshot(
                            checkpoint_index,
                            checkpoint_bundle,
                            output_path=output_path,
                            run_identity_sha256=run_identity_sha256,
                            rows=ordered_checkpoint_rows,
                        )
                        completed[case_id] = row
                        progress_ok += 1
                        status = "ok"
                    except Exception as exc:
                        progress_failed += 1
                        if isinstance(exc, _CaseEvaluationFailure):
                            case_operational[case["case_id"]] = dict(exc.operational)
                            error_type = exc.original_type
                            error_message = exc.original_message
                        else:
                            error_type = type(exc).__name__
                            error_message = " ".join(str(exc).split())[:1000]
                            case_operational[case["case_id"]] = _case_operational_summary(
                                case=case,
                                status="failed",
                                elapsed_seconds=0.0,
                                usage={
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                    "api_calls": 0,
                                },
                                provider_traces=[],
                                error={"type": error_type, "message": error_message},
                            )
                        progress_error = error_message
                        status = (
                            f"failed:{error_type}:"
                            f"{progress_error[:240]}"
                        )
                        errors.append(
                            (
                                case_indexes[case["case_id"]],
                                f"{case['case_id']}: {error_type}: {error_message}",
                            )
                        )
                    finally:
                        progress_done += 1
                        report_progress(case_id=case["case_id"], status=status)



                        future.cancel()
                        del future
                        submit_next()
        case_execution_seconds = time.monotonic() - case_execution_started
        if errors:
            error_messages = [message for _, message in sorted(errors)]
            ordered_operational = [
                case_operational[case["case_id"]]
                for case in cases
                if case["case_id"] in case_operational
            ]
            failed_receipt = _evaluation_operational_receipt(
                status="failed",
                output_path=output_path,
                test_path=test_path,
                run_identity_sha256=run_identity_sha256,
                started_at_utc=run_started_at_utc,
                finished_at_utc=_utc_now(),
                elapsed_seconds=time.monotonic() - run_started_monotonic,
                initialization_seconds=initialization_seconds,
                case_execution_seconds=case_execution_seconds,
                publication_seconds=0.0,
                workers=workers,
                publication_validation=publication_validation,
                cases=ordered_operational,
                errors=error_messages,
            )
            _atomic_create_json(
                _operational_receipt_path(output_path), failed_receipt
            )
            raise EvaluationError(
                "evaluation failed; no output published: "
                + " | ".join(error_messages)
            )
        if set(completed) != set(cases_by_id):
            raise EvaluationError("evaluation is incomplete; no output published")
        ordered_rows = [completed[case["case_id"]] for case in cases]
        print(
            f"[EVAL] case execution complete: {len(ordered_rows)}/{total_cases}; "
            f"validating ({publication_validation}) and publishing",
            file=sys.stderr,
            flush=True,
        )
        publication_started = time.monotonic()
        try:
            _write_artifact_bundle_manifest(
                staging_bundle,
                run_identity_sha256=run_identity_sha256,
                rows=ordered_rows,
            )
            _validate_existing_rows(
                ordered_rows,
                run_identity_sha256=run_identity_sha256,
                cases_by_id=cases_by_id,
                run_payload=run_payload,
                result_path=output_path,
                artifact_root_override=staging_bundle,
                validate_traces=publication_validation == "full",
            )
            os.replace(staging_bundle, final_bundle)
            bundle_published = True
            _fsync_directory(output_path.parent)



            _validate_existing_rows(
                ordered_rows,
                run_identity_sha256=run_identity_sha256,
                cases_by_id=cases_by_id,
                run_payload=run_payload,
                result_path=output_path,
                validate_traces=publication_validation == "full",
            )
            _atomic_publish_jsonl(output_path, ordered_rows)
            publication_seconds = time.monotonic() - publication_started
            ordered_operational = [
                case_operational[case["case_id"]]
                for case in cases
                if case["case_id"] in case_operational
            ]
            success_receipt = _evaluation_operational_receipt(
                status="success",
                output_path=output_path,
                test_path=test_path,
                run_identity_sha256=run_identity_sha256,
                started_at_utc=run_started_at_utc,
                finished_at_utc=_utc_now(),
                elapsed_seconds=time.monotonic() - run_started_monotonic,
                initialization_seconds=initialization_seconds,
                case_execution_seconds=case_execution_seconds,
                publication_seconds=publication_seconds,
                workers=workers,
                publication_validation=publication_validation,
                cases=ordered_operational,
                result_path=output_path,
                artifact_bundle=final_bundle,
            )
            _atomic_create_json(
                _operational_receipt_path(output_path), success_receipt
            )
            try:
                _remove_checkpoint(checkpoint_index, checkpoint_bundle)
            except EvaluationError as cleanup_error:



                print(
                    f"[EVAL] checkpoint cleanup deferred: {cleanup_error}",
                    file=sys.stderr,
                    flush=True,
                )
            return str(output_path)
        except Exception as exc:
            publication_seconds = time.monotonic() - publication_started
            receipt_path = _operational_receipt_path(output_path)
            if not receipt_path.exists():
                ordered_operational = [
                    case_operational[case["case_id"]]
                    for case in cases
                    if case["case_id"] in case_operational
                ]
                publication_receipt = _evaluation_operational_receipt(
                    status="publication_failed",
                    output_path=output_path,
                    test_path=test_path,
                    run_identity_sha256=run_identity_sha256,
                    started_at_utc=run_started_at_utc,
                    finished_at_utc=_utc_now(),
                    elapsed_seconds=time.monotonic() - run_started_monotonic,
                    initialization_seconds=initialization_seconds,
                    case_execution_seconds=case_execution_seconds,
                    publication_seconds=publication_seconds,
                    workers=workers,
                    publication_validation=publication_validation,
                    cases=ordered_operational,
                    result_path=output_path if output_path.is_file() else None,
                    artifact_bundle=final_bundle if final_bundle.is_dir() else None,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
                _atomic_create_json(receipt_path, publication_receipt)
            print(
                f"[EVAL] case execution complete but publication failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
    finally:
        if not bundle_published:
            shutil.rmtree(staging_bundle, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CHAIN on a structurally complete portable graph"
    )
    parser.add_argument("--test-file", default="dataset_split/test_qa.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse a complete result or durable per-case checkpoint; only "
            "pending cases call the provider"
        ),
    )
    parser.add_argument(
        "--resume-by-case-id",
        action="store_true",
        help=(
            "reuse completed rows from the output (or --reuse-case-file) by "
            "matching case_id; re-seal them into the current run and evaluate "
            "only IDs that are missing"
        ),
    )
    parser.add_argument(
        "--reuse-case-file",
        default=None,
        help=(
            "prior JSONL used by --resume-by-case-id; defaults to --output "
            "for the in-place 128-to-512 workflow"
        ),
    )
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "runtime-only case concurrency override; does not change the "
            "scientific configuration or run identity (compat defaults to "
            "the measured safe cap of 16)"
        ),
    )
    parser.add_argument(
        "--publish-validation",
        choices=("full", "fast"),
        default="full",
        help=(
            "publication/report trace validation mode; fast keeps structural "
            "closure and defers full trace audit"
        ),
    )
    parser.add_argument("--report-only", default=None, metavar="FILE")
    parser.add_argument(
        "--cutoff",
        default=None,
        help=(
            "compat-only fallback cutoff for records that omit both per-case "
            "cutoff and the registered historical date alias; the registered "
            "clinical legacy file also has the historical 03/10/2026 default"
        ),
    )
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--ablation-mode",
        choices=(
            "none",
            "without_ctvf",
            "without_noisy_or",
            "without_adaptive_alpha",
            "infer_tvf",
            "without_tvf",
            "without_all",
        ),
        default=None,
        help="compat-only versioned experiment mode",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("adaptive", "fixed"),
        default=None,
        help="compat-only fusion schedule override",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="compat-only fixed fusion weight in [0,1]",
    )
    return parser


def _default_output_path(test_file: str, model: str, tvf_enabled: bool) -> Path:
    model_tag = model.replace("/", "_").replace(":", "_")
    tvf_tag = "tvf_on" if tvf_enabled else "tvf_off"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "eval" / model_tag / f"{Path(test_file).stem}_{timestamp}_{tvf_tag}.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    profile = "compat"
    if args.report_only:
        if any(
            (
                args.output,
                args.limit is not None,
                args.resume,
                args.resume_by_case_id,
                args.reuse_case_file,
                args.skip_if_done,
                args.workers is not None,
                args.cutoff,
                args.graph_dir,
                args.ablation_mode,
                args.fusion_mode,
                args.fixed_alpha is not None,
            )
        ):
            raise EvaluationError("--report-only conflicts with prediction/write-mode flags")
        report_config = None
        if args.config:
            from chain.config import resolve_config
            report_config = resolve_config(profile, args.config, env=os.environ)
        if args.publish_validation == "full":


            generate_report(args.report_only, profile=profile, config=report_config)
        else:
            generate_report(
                args.report_only,
                profile=profile,
                config=report_config,
                validation_mode=args.publish_validation,
            )
        return 0
    from chain.config import resolve_config

    config_overrides: Dict[str, Any] = {}
    if args.ablation_mode is not None:
        config_overrides["ablation_mode"] = args.ablation_mode
    if args.fusion_mode is not None:
        config_overrides["fusion_mode"] = args.fusion_mode
    if args.fixed_alpha is not None:
        config_overrides["fixed_alpha"] = args.fixed_alpha
    resolved_config = resolve_config(
        profile,
        args.config,
        env=os.environ,
        overrides=config_overrides or None,
    )
    published = run_evaluation(
        args.test_file,
        args.output,
        args.limit,
        args.resume,
        args.cutoff,
        profile=profile,
        graph_dir=args.graph_dir,
        config_path=args.config,
        skip_if_done=args.skip_if_done,
        resume_by_case_id=args.resume_by_case_id,
        reuse_case_file=args.reuse_case_file,
        resolved_config=resolved_config,
        runtime_workers=args.workers,
        publication_validation=args.publish_validation,
    )
    if args.publish_validation == "full":
        generate_report(published, profile=profile, config=resolved_config)
    else:
        generate_report(
            published,
            profile=profile,
            config=resolved_config,
            validation_mode=args.publish_validation,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
