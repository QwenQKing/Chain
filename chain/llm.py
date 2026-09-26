from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import re
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx
from openai import OpenAI

from chain.config import (
    DEFAULT_CONFIG,
    EXTRACTION_RESPONSE_FORMAT_VERSION,
    ResolvedConfig,
    canonical_json,
)


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "You are an expert forecasting analyst. Predict whether the specified "
    "future event will occur using only the supplied evidence."
)
RETRY_BASE_DELAY = 1.0
RATE_LIMIT_BASE_DELAY = 5.0
RETRY_DELAY_CAP = 60.0
RATE_LIMIT_JITTER_RATIO = 0.25
_monotonic = time.monotonic
_random = random.SystemRandom().random

_TEXT_STATUS_RE = re.compile(
    r"\b(?:http(?:[\s_-]+status)?(?:[\s_-]+code)?|"
    r"status(?:[\s_-]+code)?|error[\s_-]+code)"
    r"\s*[:=]?\s*(?P<status>[1-5]\d{2})\b",
    re.IGNORECASE,
)


def _elapsed_since(started: float) -> float:


    elapsed = float(_monotonic()) - float(started)
    return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else 0.0


def _attempts_elapsed_seconds(attempts: Sequence[Mapping[str, Any]]) -> float:


    total = 0.0
    for attempt in attempts:
        raw = attempt.get("elapsed_seconds")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value) and value >= 0.0:
            total += value
    return total


def _provider_calls_elapsed_seconds(calls: Sequence[Mapping[str, Any]]) -> float:


    total = 0.0
    for call in calls:
        operational = call.get("operational")
        if not isinstance(operational, Mapping):
            continue
        raw = operational.get("elapsed_seconds")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value) and value >= 0.0:
            total += value
    return total


class ProviderCallError(RuntimeError):


    def __init__(
        self,
        *,
        stage: str,
        provider: str,
        model: str,
        attempts: Sequence[Mapping[str, Any]],
        message: str,
        model_revision: str = "unspecified",
        response_format: Optional[str] = None,
        response_format_version: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        self.stage = stage
        self.provider = provider
        self.model = model
        self.model_revision = model_revision
        self.response_format = response_format
        self.response_format_version = response_format_version
        self.attempts = [dict(item) for item in attempts]
        if elapsed_seconds is None:
            operational_elapsed = _attempts_elapsed_seconds(self.attempts)
        else:
            operational_elapsed = float(elapsed_seconds)
            if not math.isfinite(operational_elapsed) or operational_elapsed < 0.0:
                raise ValueError("elapsed_seconds must be finite and non-negative")
        self.trace = {
            "trace_schema_version": "chain-provider-trace-v1",
            "stage": stage,
            "provider": provider,
            "model": model,
            "model_revision": model_revision,
            "status": "failed",
            "attempt_count": len(self.attempts),
            "attempts": self.attempts,
            "error": message,
            "scientific": {
                "stage": stage,
                "provider": provider,
                "model": model,
                "model_revision": model_revision,
                "attempt_request_response_hashes": [
                    {
                        key: item[key]
                        for key in ("request_sha256", "response_sha256")
                        if key in item
                    }
                    for item in self.attempts
                ],
            },
            "operational": {
                "status": "failed",
                "attempt_count": len(self.attempts),
                "elapsed_seconds": operational_elapsed,
            },
        }
        if response_format is not None:
            self.trace["response_format"] = response_format
            self.trace["response_format_version"] = response_format_version
            self.trace["scientific"]["response_format"] = response_format
            self.trace["scientific"]["response_format_version"] = response_format_version
        super().__init__(f"{stage} failed after {len(self.attempts)} attempt(s): {message}")


def _select_model(value: Optional[str], default: str, name: str = "model") -> str:
    if value is None:
        selected = default
    else:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} override must be a non-empty string")
        selected = value.strip()
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"resolved {name} must be a non-empty string")
    return selected.strip()


def _select_revision(
    value: Optional[str], default: str, *, model_overridden: bool, is_paper: bool,
) -> str:
    if value is not None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_revision override must be a non-empty string")
        selected = value.strip()
    elif model_overridden:
        selected = "unspecified"
    else:
        selected = str(default).strip()
    if not selected:
        raise ValueError("resolved model_revision must be non-empty")
    if is_paper and selected.casefold() == "unspecified":
        raise ValueError("paper profile requires an explicit model_revision")
    return selected


_usage_local = threading.local()
_provider_trace_local = threading.local()
_client_lock = threading.Lock()
_clients: dict[tuple[str, str, str], OpenAI] = {}


def _require_resolved_config(
    config: Optional[ResolvedConfig], *, boundary: str
) -> ResolvedConfig:


    if config is None:
        raise TypeError(f"{boundary} requires an explicit ResolvedConfig")
    if not isinstance(config, ResolvedConfig):
        raise TypeError(f"{boundary} requires a ResolvedConfig")
    return config


def reset_provider_traces() -> None:


    _provider_trace_local.traces = []


def get_provider_traces() -> List[Dict[str, Any]]:


    return copy.deepcopy(getattr(_provider_trace_local, "traces", []))


def _record_provider_trace(trace: Mapping[str, Any]) -> None:
    if not isinstance(trace, Mapping):
        raise TypeError("provider trace must be an object")
    traces = getattr(_provider_trace_local, "traces", None)
    if traces is None:
        traces = []
        _provider_trace_local.traces = traces
    traces.append(copy.deepcopy(dict(trace)))


def reset_usage() -> None:
    _usage_local.prompt_tokens = 0
    _usage_local.completion_tokens = 0
    _usage_local.total_tokens = 0
    _usage_local.api_calls = 0


    reset_provider_traces()


def get_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": int(getattr(_usage_local, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(_usage_local, "completion_tokens", 0)),
        "total_tokens": int(getattr(_usage_local, "total_tokens", 0)),
        "api_calls": int(getattr(_usage_local, "api_calls", 0)),
    }


def _accumulate_usage(resp_usage: Any = None, *, count_call: bool = True) -> None:


    if count_call:
        _usage_local.api_calls = getattr(_usage_local, "api_calls", 0) + 1
    if resp_usage is None:
        return
    for local_name, response_name in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        current = getattr(_usage_local, local_name, 0)
        value = getattr(resp_usage, response_name, 0) or 0
        setattr(_usage_local, local_name, current + int(value))


def reset_clients() -> None:


    global _clients
    with _client_lock:
        old, _clients = _clients, {}
    for client in old.values():
        try:
            client.close()
        except Exception:
            pass


def _make_http_client(max_connections: int = 256) -> httpx.Client:
    return httpx.Client(
        trust_env=False,
        limits=httpx.Limits(
            max_connections=max_connections + 16,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
    )


def _client_for(
    *,
    kind: str,
    base_url: str,
    api_key: str,
    config: ResolvedConfig,
) -> OpenAI:
    if not api_key:
        revision = (
            config.embed_revision if kind == "embedding"
            else config.build_llm_revision if kind == "build"
            else config.llm_revision
        )
        raise ProviderCallError(
            stage=f"{kind}.provider_init",
            provider=config.api_provider,
            model=(
                config.embed_model if kind == "embedding"
                else config.build_llm_model if kind == "build"
                else config.llm_model
            ),
            attempts=[],
            message="missing API credential",
            model_revision=revision,
        )
    secret_identity = hashlib.sha256(
        (api_key + "\0" + config.alternate_token).encode("utf-8")
    ).hexdigest()
    cache_key = (f"{config.api_provider}:{kind}", base_url, secret_identity)
    with _client_lock:
        client = _clients.get(cache_key)
        if client is not None:
            return client
        headers = (
            {"token": config.alternate_token}
            if kind in {"chat", "build"}
            and config.api_provider == "alternate"
            and config.alternate_token
            else None
        )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=120.0 if kind != "embedding" else 60.0,
            default_headers=headers,
            http_client=_make_http_client(),
        )
        _clients[cache_key] = client
        return client


def get_client(config: Optional[ResolvedConfig] = None) -> OpenAI:
    resolved = _require_resolved_config(config, boundary="get_client")
    return _client_for(
        kind="chat",
        base_url=resolved.llm_base_url,
        api_key=resolved.llm_api_key,
        config=resolved,
    )


def get_build_client(config: Optional[ResolvedConfig] = None) -> OpenAI:
    resolved = _require_resolved_config(config, boundary="get_build_client")
    return _client_for(
        kind="build",
        base_url=resolved.build_llm_base_url,
        api_key=resolved.build_llm_api_key,
        config=resolved,
    )


def get_embed_client(config: Optional[ResolvedConfig] = None) -> OpenAI:
    resolved = _require_resolved_config(config, boundary="get_embed_client")
    return _client_for(
        kind="embedding",
        base_url=resolved.embed_base_url,
        api_key=resolved.embed_api_key,
        config=resolved,
    )


def _safe_error(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()

    text = re.sub(r"(?i)(api[_ -]?key|authorization|bearer|token)\s*[:=]\s*\S+", r"\1=<redacted>", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "<redacted-key>", text)
    text = re.sub(r"(https?://)[^/@\s:]+:[^/@\s]+@", r"\1<redacted>@", text)
    text = re.sub(
        r"(?i)([?&](?:api[_-]?key|key|token|access_token)=)[^&\s]+",
        r"\1<redacted>",
        text,
    )
    return text[:500] or type(exc).__name__


def _usage_values(resp_usage: Any) -> Dict[str, int]:
    if resp_usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(resp_usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(resp_usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(resp_usage, "total_tokens", 0) or 0),
    }


def _status_code(exc: BaseException) -> Optional[int]:


    response = getattr(exc, "response", None)
    for raw_status in (
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
    ):
        if isinstance(raw_status, bool):
            continue
        if isinstance(raw_status, int):
            status = raw_status
        elif isinstance(raw_status, str) and re.fullmatch(r"\d{3}", raw_status.strip()):
            status = int(raw_status)
        else:
            continue
        if 100 <= status <= 599:
            return status
    return None


def _text_status_code(exc: BaseException) -> Optional[int]:
    match = _TEXT_STATUS_RE.search(str(exc))
    return int(match.group("status")) if match is not None else None


def _is_non_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    status = _status_code(exc)
    if status in (401, 403):
        return True
    if status == 429 or (status is not None and 500 <= status <= 599):
        return False
    if any(
        marker in text
        for marker in (
            "context_length", "maximum context", "invalid_request", "authentication",
            "permission",
        )
    ):
        return True

    return status is None and _text_status_code(exc) in (401, 403)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:


    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw_value = getter("retry-after")
    if raw_value is None:
        raw_value = getter("Retry-After")
    if isinstance(raw_value, bool):
        return None
    try:
        delay = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(delay) or delay < 0.0:
        return None
    return min(delay, RETRY_DELAY_CAP)


def _retry_delay(exc: BaseException, attempt_index: int) -> float:


    status = _status_code(exc)
    is_rate_limit = status == 429 or (
        status is None and _text_status_code(exc) == 429
    )
    if not is_rate_limit:
        return RETRY_BASE_DELAY * (2 ** (attempt_index - 1))
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    exponential = RATE_LIMIT_BASE_DELAY * (2 ** (attempt_index - 1))
    jittered = exponential * (1.0 + RATE_LIMIT_JITTER_RATIO * _random())
    return min(jittered, RETRY_DELAY_CAP)


def _request_hash(
    messages: Sequence[Mapping[str, str]],
    *,
    response_format: Optional[str] = None,
    response_format_version: Optional[str] = None,
) -> str:


    payload: Any = list(messages)
    if response_format is not None:
        payload = {
            "messages": list(messages),
            "response_format": {"type": response_format},
            "response_format_version": response_format_version,
        }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def resolved_chat(
    prompt: str,
    *,
    system: str = DEFAULT_SYSTEM,
    config: Optional[ResolvedConfig] = None,
    stage: str = "inference_chat",
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    client: Any = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    role: str = "inference",
    max_attempts: Optional[int] = None,
    response_format: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    resolved = _require_resolved_config(config, boundary="resolved_chat")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system must be a non-empty string")
    if role not in {"inference", "build"}:
        raise ValueError("role must be inference or build")
    if response_format is None:
        selected_response_format = None
    elif not isinstance(response_format, str) or response_format != "json_object":
        raise ValueError("response_format must be None or 'json_object'")
    else:
        selected_response_format = response_format
    if selected_response_format is not None and role != "build":
        raise ValueError("structured response_format is restricted to build-role calls")
    if role == "build" and selected_response_format != "json_object":
        raise ValueError("build-role calls require response_format='json_object'")
    selected_response_format_version = (
        EXTRACTION_RESPONSE_FORMAT_VERSION
        if selected_response_format is not None
        else None
    )
    if (
        selected_response_format is not None
        and getattr(resolved, "extraction_response_format_version", None)
        != selected_response_format_version
    ):
        raise ValueError(
            "resolved config does not attest the sealed extraction response-format protocol"
        )
    if max_attempts is None:
        attempt_limit = resolved.llm_retry_attempts
    else:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if max_attempts > resolved.llm_retry_attempts:
            raise ValueError("max_attempts cannot exceed the resolved retry policy")
        attempt_limit = max_attempts
    role_model = resolved.build_llm_model if role == "build" else resolved.llm_model
    role_revision = resolved.build_llm_revision if role == "build" else resolved.llm_revision
    role_temperature = (
        resolved.extractor_temperature if role == "build" else resolved.llm_temperature
    )
    role_max_tokens = (
        resolved.extractor_max_tokens if role == "build" else resolved.llm_max_tokens
    )
    if role == "inference" and "outcome_mapping" in stage:
        role_max_tokens = resolved.outcome_mapping_max_tokens
    if resolved.is_paper:
        if model is not None and str(model).strip() != role_model:
            raise ValueError("paper profile forbids a stage model override")
        if model_revision is not None and str(model_revision).strip() != role_revision:
            raise ValueError("paper profile forbids a stage model revision override")
        if temperature is not None and float(temperature) != float(role_temperature):
            raise ValueError("paper profile forbids a stage temperature override")
        if max_tokens is not None and max_tokens != role_max_tokens:
            raise ValueError("paper profile forbids a stage max_tokens override")
    selected_model = _select_model(model, role_model)





    model_is_override = model is not None and selected_model != role_model
    selected_revision = _select_revision(
        model_revision,
        role_revision,
        model_overridden=model_is_override,
        is_paper=resolved.is_paper,
    )
    if temperature is None:
        selected_temperature = role_temperature
    else:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a finite number, not bool")
        selected_temperature = float(temperature)
    if max_tokens is None:
        selected_max_tokens = role_max_tokens
    else:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be an integer, not bool")
        selected_max_tokens = max_tokens
    if not math.isfinite(selected_temperature) or not 0.0 <= selected_temperature <= 2.0:
        raise ValueError("temperature must be finite and in [0,2]")
    if selected_max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    provider_client = (
        get_build_client(resolved) if role == "build" else get_client(resolved)
    ) if client is None else client
    attempts: list[dict[str, Any]] = []
    request_sha256 = _request_hash(
        messages,
        response_format=selected_response_format,
        response_format_version=selected_response_format_version,
    )
    response_format_identity = (
        {
            "response_format": selected_response_format,
            "response_format_version": selected_response_format_version,
        }
        if selected_response_format is not None
        else {}
    )
    for attempt_index in range(1, attempt_limit + 1):
        attempt_started = _monotonic()
        response_usage: Optional[Dict[str, int]] = None
        content: Optional[str] = None
        try:
            create_call = provider_client.chat.completions.create
            _accumulate_usage(None)
            create_kwargs: Dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "temperature": selected_temperature,
                "max_tokens": selected_max_tokens,
            }
            if selected_response_format is not None:
                create_kwargs["response_format"] = {
                    "type": selected_response_format
                }
            response = create_call(
                **create_kwargs,
            )
            response_usage = _usage_values(getattr(response, "usage", None))
            _accumulate_usage(getattr(response, "usage", None), count_call=False)
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("provider returned no choices")
            content = getattr(getattr(choices[0], "message", None), "content", None)
            if not isinstance(content, str) or not content.strip():
                raise ValueError("provider returned empty message content")
            attempt_elapsed = _elapsed_since(attempt_started)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "status": "ok",
                    "model": selected_model,
                    "model_revision": selected_revision,
                    **response_format_identity,
                    "request_sha256": request_sha256,
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "response_nbytes": len(content.encode("utf-8")),
                    "raw_response": content,
                    "usage": response_usage,
                    "elapsed_seconds": attempt_elapsed,
                }
            )
            trace = {
                "trace_schema_version": "chain-provider-trace-v1",
                "stage": stage,
                "provider": resolved.api_provider,
                "model": selected_model,
                "model_revision": selected_revision,
                **response_format_identity,
                "status": "ok",
                "request_sha256": request_sha256,
                "temperature": selected_temperature,
                "max_tokens": selected_max_tokens,
                "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "attempt_count": attempt_index,
                "attempts": attempts,
                "usage": response_usage,
                "scientific": {
                    "stage": stage,
                    "provider": resolved.api_provider,
                    "model": selected_model,
                    "model_revision": selected_revision,
                    **response_format_identity,
                    "request_sha256": request_sha256,
                    "temperature": selected_temperature,
                    "max_tokens": selected_max_tokens,
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                },
                "operational": {
                    "status": "ok",
                    "attempt_count": attempt_index,
                    "usage": response_usage,
                    "elapsed_seconds": _attempts_elapsed_seconds(attempts),
                },
            }
            return content, trace
        except Exception as exc:
            attempt_elapsed = _elapsed_since(attempt_started)
            attempt_error: Dict[str, Any] = {
                "attempt": attempt_index,
                "status": "error",
                "model": selected_model,
                "model_revision": selected_revision,
                **response_format_identity,
                "request_sha256": request_sha256,
                "error_type": type(exc).__name__,
                "error": _safe_error(exc),
                "usage": response_usage,
                "elapsed_seconds": attempt_elapsed,
            }
            if isinstance(content, str):
                attempt_error.update(
                    {
                        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "response_nbytes": len(content.encode("utf-8")),
                        "raw_response": content,
                    }
                )
            attempts.append(attempt_error)
            if _is_non_retryable(exc) or attempt_index >= attempt_limit:
                raise ProviderCallError(
                    stage=stage,
                    provider=resolved.api_provider,
                    model=selected_model,
                    model_revision=selected_revision,
                    response_format=selected_response_format,
                    response_format_version=selected_response_format_version,
                    attempts=attempts,
                    message=_safe_error(exc),
                ) from exc
            sleep_fn(_retry_delay(exc, attempt_index))
    raise AssertionError("unreachable")


def chat(
    prompt: str,
    system: str = DEFAULT_SYSTEM,
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    *,
    config: Optional[ResolvedConfig] = None,
    stage: str = "inference_chat",
    client: Any = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    resolved_config = DEFAULT_CONFIG if config is None else config
    content, _ = resolved_chat(
        prompt,
        system=system,
        model=model,
        model_revision=model_revision,
        temperature=temperature,
        max_tokens=max_tokens,
        config=resolved_config,
        stage=stage,
        client=client,
        sleep_fn=sleep_fn,
    )
    return content


def resolved_build_chat(
    prompt: str,
    *,
    system: str = DEFAULT_SYSTEM,
    config: Optional[ResolvedConfig] = None,
    stage: str = "graph_extraction",
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    client: Any = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_attempts: Optional[int] = None,
    response_format: str = "json_object",
) -> Tuple[str, Dict[str, Any]]:


    resolved = _require_resolved_config(config, boundary="resolved_build_chat")
    if response_format != "json_object":
        raise ValueError("resolved_build_chat requires response_format='json_object'")
    selected_model = resolved.build_llm_model if model is None else model
    selected_revision = model_revision
    if selected_revision is None and (model is None or model == resolved.build_llm_model):
        selected_revision = resolved.build_llm_revision
    return resolved_chat(
        prompt,
        system=system,
        config=resolved,
        stage=stage,
        model=selected_model,
        model_revision=selected_revision,
        temperature=resolved.extractor_temperature if temperature is None else temperature,
        max_tokens=resolved.extractor_max_tokens if max_tokens is None else max_tokens,
        client=client,
        sleep_fn=sleep_fn,
        role="build",
        max_attempts=max_attempts,
        response_format=response_format,
    )


def _extract_json(text: str) -> str:


    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return stripped
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(stripped[start:])
    except json.JSONDecodeError:
        return stripped[start:]
    return stripped[start : start + end]


def _repair_json(text: str) -> str:


    value = re.sub(r"//[^\n]*", "", text)
    value = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
    return re.sub(r",(\s*[}\]])", r"\1", value)


_MARKDOWN_JSON_FENCE_NORMALIZATION = "markdown-json-fence-v1"
_MARKDOWN_JSON_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*(?:\r\n|\n|\r)"
    r"(?P<body>.*?)"
    r"(?:\r\n|\n|\r)?```[ \t]*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def _normalize_json_response_envelope(raw: str) -> Tuple[str, Optional[str]]:


    stripped = raw.strip()
    if not stripped.startswith("```"):
        return raw, None
    match = _MARKDOWN_JSON_FENCE_RE.fullmatch(stripped)
    if match is None:
        raise ValueError(
            "response markdown JSON fence must be one complete whole-response envelope"
        )
    return match.group("body"), _MARKDOWN_JSON_FENCE_NORMALIZATION


def _strict_json_object(raw: str) -> Dict[str, Any]:
    def strict_pairs(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not a single valid JSON value: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("response JSON root must be an object")

    canonical_json(parsed)
    return parsed


def resolved_chat_json(
    prompt: str,
    *,
    config: Optional[ResolvedConfig] = None,
    stage: str = "inference_json",
    schema_validator: Optional[Callable[[Dict[str, Any]], Any]] = None,
    correction_attempts: Optional[int] = None,
    client: Any = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    **chat_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = _require_resolved_config(config, boundary="resolved_chat_json")
    if correction_attempts is None:
        max_corrections = resolved.json_correction_attempts
    else:
        if isinstance(correction_attempts, bool) or not isinstance(correction_attempts, int):
            raise ValueError("correction_attempts must be an integer, not bool")
        max_corrections = correction_attempts
    if max_corrections < 0:
        raise ValueError("correction_attempts must be >= 0")
    calls: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    schema_attempts: list[dict[str, Any]] = []
    for correction_index in range(max_corrections + 1):
        instruction = (
            "\n\nReturn exactly one valid JSON object. Do not use markdown fences, "
            "comments, trailing text, NaN, or Infinity."
        )
        if correction_index:
            instruction += " This is a schema-correction retry; satisfy every requested field exactly."
        raw, call_trace = resolved_chat(
            prompt + instruction,
            config=resolved,
            stage=f"{stage}.provider",
            client=client,
            sleep_fn=sleep_fn,
            **chat_kwargs,
        )
        calls.append(call_trace)
        response_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        attempt_audit: dict[str, Any] = {
            "correction": correction_index,
            "request_sha256": call_trace["request_sha256"],
            "response_sha256": response_sha256,
            "response_nbytes": len(raw.encode("utf-8")),
            "raw_response": raw,
        }
        try:
            normalized_raw, response_normalization = _normalize_json_response_envelope(raw)
            if response_normalization is not None:
                attempt_audit["response_normalization"] = response_normalization
            payload = _strict_json_object(normalized_raw)
            if schema_validator is not None:
                validated = schema_validator(payload)
                if validated is not None:
                    payload = validated
                if not isinstance(payload, dict):
                    raise ValueError("schema validator must return a dict or None")
            attempt_audit["status"] = "ok"
            schema_attempts.append(attempt_audit)
            success_trace: dict[str, Any] = {
                "trace_schema_version": "chain-provider-json-trace-v1",
                "stage": stage,
                "status": "ok",
                "correction_count": correction_index,
                "provider_calls": calls,
                "schema_attempts": schema_attempts,
                "parse_errors": parse_errors,
                "response_sha256": response_sha256,
                "scientific": {
                    "stage": stage,
                    "schema_attempts": schema_attempts,
                    "response_sha256": response_sha256,
                },
                "operational": {
                    "status": "ok",
                    "correction_count": correction_index,
                    "provider_call_count": len(calls),
                    "elapsed_seconds": _provider_calls_elapsed_seconds(calls),
                },
            }
            if response_normalization is not None:
                success_trace["response_normalization"] = response_normalization
                success_trace["scientific"]["response_normalization"] = (
                    response_normalization
                )
            return payload, success_trace
        except Exception as exc:
            attempt_audit.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": _safe_error(exc),
                }
            )
            schema_attempts.append(attempt_audit)
            parse_errors.append(dict(attempt_audit))
    failure_model = _select_model(chat_kwargs.get("model"), resolved.llm_model)
    failure_revision = _select_revision(
        chat_kwargs.get("model_revision"),
        resolved.llm_revision,
        model_overridden=chat_kwargs.get("model") is not None,
        is_paper=resolved.is_paper,
    )
    failure = ProviderCallError(
        stage=stage,
        provider=resolved.api_provider,
        model=failure_model,
        model_revision=failure_revision,
        attempts=schema_attempts,
        message=parse_errors[-1]["error"] if parse_errors else "JSON validation failed",
        elapsed_seconds=_provider_calls_elapsed_seconds(calls),
    )




    failure.trace["operational"].update(
        {
            "provider_call_count": len(calls),
            "provider_calls": copy.deepcopy(calls),
        }
    )
    raise failure


def call_json(prompt: str, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    return resolved_chat_json(prompt, **kwargs)


def chat_json(prompt: str, **kwargs: Any) -> Dict[str, Any]:
    if kwargs.get("config") is None:
        kwargs["config"] = DEFAULT_CONFIG
    try:
        payload, trace = resolved_chat_json(prompt, **kwargs)
    except ProviderCallError as exc:
        _record_provider_trace(exc.trace)
        raise
    _record_provider_trace(trace)
    return payload


def _validate_vectors(
    vectors: Sequence[Sequence[Any]],
    *,
    expected_count: int,
    expected_dim: Optional[int],
) -> List[List[float]]:
    if len(vectors) != expected_count:
        raise ValueError(
            f"embedding count mismatch: expected {expected_count}, received {len(vectors)}"
        )
    converted: list[list[float]] = []
    observed_dim: Optional[int] = None
    for row_index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"embedding {row_index} is empty or not an array")
        row: list[float] = []
        for value in vector:



            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"embedding {row_index} contains a non-numeric value")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"embedding {row_index} contains non-finite value")
            try:
                number = struct.unpack("<f", struct.pack("<f", number))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(f"embedding {row_index} is not representable as float32") from exc
            row.append(float(number))
        norm = math.sqrt(sum(number * number for number in row))
        if not math.isfinite(norm) or norm == 0.0:
            if not math.isfinite(norm):
                raise ValueError(f"embedding {row_index} has a non-finite L2 norm")
            raise ValueError(f"embedding {row_index} is an all-zero vector")



        row = [float(number / norm) for number in row]
        if observed_dim is None:
            observed_dim = len(row)
        elif len(row) != observed_dim:
            raise ValueError("embedding vectors have inconsistent dimensions")
        converted.append(row)
    if expected_dim is not None and observed_dim != expected_dim:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dim}, received {observed_dim}"
        )
    return converted


def embed_texts_strict(
    texts: Sequence[str],
    *,
    config: Optional[ResolvedConfig] = None,
    stage: str = "embedding",
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    expected_dim: Optional[int] = None,
    client: Any = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    resolved = _require_resolved_config(config, boundary="embed_texts_strict")
    selected_model = _select_model(model, resolved.embed_model, "embedding model")
    selected_revision = _select_revision(
        model_revision,
        resolved.embed_revision,
        model_overridden=model is not None,
        is_paper=resolved.is_paper,
    )
    if resolved.is_paper:
        if model is not None and selected_model != resolved.embed_model:
            raise ValueError("paper profile forbids an embedding model override")
        if model_revision is not None and selected_revision != resolved.embed_revision:
            raise ValueError("paper profile forbids an embedding revision override")
    if expected_dim is None:
        selected_dim = resolved.embed_dim
    else:
        if isinstance(expected_dim, bool) or not isinstance(expected_dim, int) or expected_dim <= 0:
            raise ValueError("expected_dim must be a positive integer, not bool")
        selected_dim = expected_dim
    if resolved.is_paper and selected_dim != resolved.embed_dim:
        raise ValueError("paper profile forbids an embedding dimension override")
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("embedding texts must be a sequence")
    if not texts:
        return [], {
            "trace_schema_version": "chain-provider-trace-v1",
            "stage": stage,
            "status": "ok",
            "attempt_count": 0,
            "input_count": 0,
            "vectors": 0,
            "model": selected_model,
            "model_revision": selected_revision,
            "scientific": {
                "stage": stage,
                "provider": resolved.api_provider,
                "model": selected_model,
                "model_revision": selected_revision,
                "input_sha256": hashlib.sha256(canonical_json([]).encode("utf-8")).hexdigest(),
                "dimension": selected_dim,
            },
            "operational": {
                "status": "ok",
                "attempt_count": 0,
                "elapsed_seconds": 0.0,
            },
        }
    normalized_texts: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"embedding input {index} must be a non-empty string")
        normalized_texts.append(text)
    if resolved.alternate_embed_enabled:
        raise ProviderCallError(
            stage=stage,
            provider=resolved.api_provider,
            model=selected_model,
            model_revision=selected_revision,
            attempts=[],
            message="alternate embedding transport is not a sealed supported path",
        )
    provider_client = get_embed_client(resolved) if client is None else client
    attempts: list[dict[str, Any]] = []
    input_sha256 = hashlib.sha256(canonical_json(normalized_texts).encode("utf-8")).hexdigest()
    for attempt_index in range(1, resolved.embedding_attempts + 1):
        attempt_started = _monotonic()
        response_usage: Optional[Dict[str, int]] = None
        try:
            create_call = provider_client.embeddings.create
            _accumulate_usage(None)
            response = create_call(
                model=selected_model,
                input=normalized_texts,
            )
            response_usage = _usage_values(getattr(response, "usage", None))
            _accumulate_usage(getattr(response, "usage", None), count_call=False)
            data = sorted(getattr(response, "data", []), key=lambda item: item.index)
            vectors = _validate_vectors(
                [item.embedding for item in data],
                expected_count=len(normalized_texts),
                expected_dim=selected_dim,
            )
            attempt_elapsed = _elapsed_since(attempt_started)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "status": "ok",
                    "input_sha256": input_sha256,
                    "usage": response_usage,
                    "elapsed_seconds": attempt_elapsed,
                }
            )
            return vectors, {
                "trace_schema_version": "chain-provider-trace-v1",
                "stage": stage,
                "provider": resolved.api_provider,
                "model": selected_model,
                "model_revision": selected_revision,
                "status": "ok",
                "attempt_count": attempt_index,
                "attempts": attempts,
                "input_count": len(normalized_texts),
                "dimension": selected_dim,
                "input_sha256": input_sha256,
                "usage": response_usage,
                "scientific": {
                    "stage": stage,
                    "provider": resolved.api_provider,
                    "model": selected_model,
                    "model_revision": selected_revision,
                    "input_sha256": input_sha256,
                    "dimension": selected_dim,
                },
                "operational": {
                    "status": "ok",
                    "attempt_count": attempt_index,
                    "usage": response_usage,
                    "elapsed_seconds": _attempts_elapsed_seconds(attempts),
                },
            }
        except Exception as exc:
            attempt_elapsed = _elapsed_since(attempt_started)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "status": "error",
                    "input_sha256": input_sha256,
                    "error_type": type(exc).__name__,
                    "error": _safe_error(exc),
                    "elapsed_seconds": attempt_elapsed,
                    **({"usage": response_usage} if response_usage is not None else {}),
                }
            )
            if _is_non_retryable(exc) or attempt_index >= resolved.embedding_attempts:
                raise ProviderCallError(
                    stage=stage,
                    provider=resolved.api_provider,
                    model=selected_model,
                    model_revision=selected_revision,
                    attempts=attempts,
                    message=_safe_error(exc),
                ) from exc
            sleep_fn(_retry_delay(exc, attempt_index))
    raise AssertionError("unreachable")


def embed_texts(
    texts: List[str],
    *,
    config: Optional[ResolvedConfig] = None,
    client: Any = None,
    expected_dim: Optional[int] = None,
    model: Optional[str] = None,
    model_revision: Optional[str] = None,
    stage: str = "embedding",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[List[float]]:
    resolved_config = DEFAULT_CONFIG if config is None else config
    try:
        vectors, trace = embed_texts_strict(
            texts,
            config=resolved_config,
            client=client,
            expected_dim=expected_dim,
            model=model,
            model_revision=model_revision,
            stage=stage,
            sleep_fn=sleep_fn,
        )
    except ProviderCallError as exc:
        _record_provider_trace(exc.trace)
        raise
    _record_provider_trace(trace)
    return vectors


_COSINE_BOUND_TOLERANCE = 1e-6


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        raise ValueError("cosine vectors must be non-empty and have equal dimensions")
    va = [float(value) for value in a]
    vb = [float(value) for value in b]
    if not all(math.isfinite(value) for value in va + vb):
        raise ValueError("cosine vectors must be finite")
    norm_a = math.sqrt(sum(value * value for value in va))
    norm_b = math.sqrt(sum(value * value for value in vb))
    if (
        not math.isfinite(norm_a)
        or not math.isfinite(norm_b)
        or norm_a <= 0.0
        or norm_b <= 0.0
    ):
        raise ValueError("cosine vectors must have finite non-zero norm")
    score = sum(x * y for x, y in zip(va, vb)) / (norm_a * norm_b)
    if not math.isfinite(score):
        raise ValueError("cosine similarity must be finite")
    if (
        score < -1.0 - _COSINE_BOUND_TOLERANCE
        or score > 1.0 + _COSINE_BOUND_TOLERANCE
    ):
        raise ValueError("cosine similarity is materially outside [-1,1]")
    return min(1.0, max(-1.0, score))
