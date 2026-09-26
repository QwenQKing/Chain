from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit


BASE_DIR = Path(".")

COMPAT_PROFILE = "compat"
PAPER_PROFILE = "paper"
COMPAT_PROFILE_ID = "compat"
PAPER_PROFILE_ID = "paper"
CONFIG_SCHEMA_VERSION = "chain-inference-config-v1"
CONSTRUCTION_SCHEMA_VERSION = "chain-construction-config-v2"
BINARY_AXIS_SCHEMA_VERSION = "chain-binary-axis-v1"
LEGACY_AXIS_ADAPTER_ID = "chain-legacy-native-yes-no-v1"
REASONING_PROMPT_VERSION = "chain-reasoning-prompt-v3"
EXTRACTION_RESPONSE_FORMAT_VERSION = "chain-extraction-response-format-json-object-v1"





PAPER_ABLATION_MODES = frozenset(
    {
        "none",
        "without_ctvf",
        "without_noisy_or",
        "without_adaptive_alpha",
    }
)
COMPAT_ONLY_ABLATION_MODES = frozenset(
    {
        "infer_tvf",
        "without_tvf",
        "without_all",
    }
)
ABLATION_MODES = PAPER_ABLATION_MODES | COMPAT_ONLY_ABLATION_MODES




_ABLATION_TVF_FLAGS = {
    "none": (True, True),
    "without_ctvf": (True, True),
    "without_noisy_or": (True, True),
    "without_adaptive_alpha": (True, True),
    "infer_tvf": (True, False),
    "without_tvf": (False, False),
    "without_all": (False, False),
}


class ConfigurationError(ValueError):
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
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalised_key = unicodedata.normalize("NFC", key)
            if normalised_key in result:
                raise ValueError(f"duplicate canonical JSON key: {normalised_key!r}")
            result[normalised_key] = _normalise_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:


    return json.dumps(
        _normalise_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise ConfigurationError(f"{name} must be one of true/false/1/0/yes/no")


def _strict_int(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ConfigurationError(f"{name} must be an integer")
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return parsed


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a finite number, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ConfigurationError(f"{name} must be finite")
    return parsed


def _optional_positive_int(value: Any, name: str) -> Optional[int]:
    if value is None or (
        isinstance(value, str)
        and value.strip().lower() in {"", "none", "null", "unlimited"}
    ):
        return None
    return _strict_int(value, name, minimum=1)


def _validate_endpoint(value: Any, name: str) -> str:
    raw = str(value).strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ConfigurationError(f"{name} must not contain userinfo credentials")
    if parsed.query or parsed.fragment:
        raise ConfigurationError(f"{name} must not contain query or fragment data")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def endpoint_identity(value: str) -> dict[str, str]:


    normalized = _validate_endpoint(value, "endpoint")
    parsed = urlsplit(normalized)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname or "",
        "port": str(parsed.port or ""),
        "path_sha256": hashlib.sha256(parsed.path.encode("utf-8")).hexdigest(),
    }


def _normalise_profile(profile: str) -> str:
    raw = str(profile or COMPAT_PROFILE).strip().lower()
    aliases = {
        COMPAT_PROFILE: COMPAT_PROFILE,
        COMPAT_PROFILE_ID: COMPAT_PROFILE,
        PAPER_PROFILE: PAPER_PROFILE,
        PAPER_PROFILE_ID: PAPER_PROFILE,
    }
    if raw not in aliases:
        raise ConfigurationError("profile must be compat or paper")
    return aliases[raw]


def _read_secret_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    except OSError:
        return ""


_DEFAULT_URL = "https://api.openai.com/v1"
_ALTERNATE_BASE = "https://placeholder.example.com/v1"


def alternate_chat_url_for(model: str) -> str:
    model_name = (model or "").lower()
    if model_name.startswith(("gpt", "o1", "o3", "o4")):
        return f"{_ALTERNATE_BASE}/chatgpt/v3"
    if model_name.startswith(("deepseek-r1", "deepseek-v3")):
        return f"{_ALTERNATE_BASE}/qianwen/v1"
    if model_name.startswith("deepseek"):
        return f"{_ALTERNATE_BASE}/deepseek/v1"
    if model_name.startswith(("gemini", "google/")):
        return f"{_ALTERNATE_BASE}/new/gemini/openapi"
    if model_name.startswith(("moonshot", "kimi")):
        return f"{_ALTERNATE_BASE}/kimi_moonshot/v2"
    if model_name.startswith(("doubao", "ep-")):
        return f"{_ALTERNATE_BASE}/doubao/v3"
    return f"{_ALTERNATE_BASE}/chatgpt/v3"


def _provider_defaults(env: Mapping[str, str]) -> dict[str, str]:
    provider = str(env.get("API_PROVIDER", "default")).strip().lower() or "default"
    llm_model = str(env.get("LLM_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"
    build_model = (
        str(env.get("BUILD_LLM_MODEL", "gpt-4o-mini")).strip()
        or "gpt-4o-mini"
    )
    default_key = next(
        (
            secret
            for secret in (
                _read_secret_file(BASE_DIR / "api_key.txt"),
            )
            if secret
        ),
        env.get("OPENAI_API_KEY", ""),
    )
    if provider == "alternate":
        alternate_token = next(
            (
                secret
                for secret in (
                    _read_secret_file(BASE_DIR / "alternate_token.txt"),
                    _read_secret_file(Path("..") / "alternate_token.txt"),
                )
                if secret
            ),
            "",
        )
        build_chat_base = alternate_chat_url_for(build_model)
        inference_chat_base = alternate_chat_url_for(llm_model)
        embed_base = _DEFAULT_URL
    else:
        alternate_token = ""
        build_chat_base = _DEFAULT_URL
        inference_chat_base = _DEFAULT_URL
        embed_base = _DEFAULT_URL
    openai_key = env.get("OPENAI_API_KEY", default_key)
    openai_base = env.get("OPENAI_BASE_URL", inference_chat_base)
    return {
        "api_provider": provider,
        "alternate_token": alternate_token,
        "build_llm_api_key": env.get("BUILD_LLM_API_KEY", openai_key),
        "build_llm_base_url": env.get(
            "BUILD_LLM_BASE_URL",
            env.get("OPENAI_BASE_URL", build_chat_base),
        ),
        "embed_api_key": env.get("EMBED_API_KEY", openai_key),
        "embed_base_url": env.get("EMBED_BASE_URL", embed_base),
        "llm_api_key": env.get("LLM_API_KEY", openai_key),
        "llm_base_url": env.get("LLM_BASE_URL", openai_base),
    }


_HASHED_FIELDS = (
    "schema_version", "construction_schema_version", "profile_id",
    "api_provider", "build_llm_base_url", "build_llm_model", "build_llm_revision",
    "embed_base_url", "embed_model", "embed_revision", "embed_dim",
    "embedding_preprocess_version", "embedding_normalization",
    "embedding_distance", "embedding_dtype", "embedding_serialization_precision",
    "llm_base_url", "llm_model", "llm_revision",
    "chunk_max_tokens", "chunk_overlap_tokens", "tokenizer_model",
    "extraction_batch_size", "extractor_temperature", "extractor_max_tokens",
    "extractor_schema_retries", "extraction_workers", "embedding_workers",
    "embedding_batch_size", "embedding_attempts", "d_max", "B_pi", "F_max",
    "tau_phi", "theta", "eta", "k_sat", "beta_0", "Z", "alpha_b",
    "alpha_0", "Omega_0", "zeta", "epsilon", "fusion_mode", "fixed_alpha",
    "ablation_mode", "tvf_enabled", "ctvf_enabled", "max_outer_loops",
    "collect_rounds_per_phase", "reason_rounds_per_phase", "confidence_threshold",
    "minimum_new_results", "stability_window", "stability_stdev",
    "stability_mean", "target_top_k", "target_candidate_cap",
    "target_min_similarity", "target_min_margin", "direction_entity_cap",
    "direction_char_cap", "reasoning_hub_cap", "reasoning_hub_multiplier",
    "timeline_per_entity_cap", "trajectory_char_cap", "causal_link_working_cap",
    "causal_link_returned_cap", "causal_link_prompt_cap", "chain_start_cap",
    "chain_working_cap", "chain_returned_cap", "chain_prompt_cap",
    "hyperedge_working_cap", "hyperedge_prompt_cap", "contradiction_returned_cap",
    "contradiction_prompt_cap", "prior_outcome_cap", "prior_description_char_cap",
    "prior_serialized_char_cap", "attribution_top_k", "counterfactual_top_k",
    "displayed_top_chain_cap", "max_prompt_chars", "speculative_merge_threshold",
    "speculative_merge_min_tokens", "derivation_similarity_threshold",
    "prune_threshold", "serialized_path_cap", "llm_retry_attempts",
    "llm_temperature", "llm_max_tokens", "json_correction_attempts",
    "outcome_mapping_max_tokens", "outcome_sum_tolerance", "chain_eval_workers",
    "chain_embed_concurrency", "chain_skip_viz_embed", "alternate_embed_enabled",
    "ece_bins", "ace_bins", "reliability_bins", "mce_min_bin_size",
    "oc_uc_confidence_gap", "nll_epsilon", "bootstrap_draws", "bootstrap_seed",
    "canonical_float_format_version", "extraction_prompt_version",
    "extraction_schema_version", "relation_admissibility_policy_version",
    "grounding_policy_version", "proposition_schema_projection_policy_version",
    "extraction_response_format_version",
    "reasoning_prompt_version",
    "outcome_mapping_schema_version", "binary_axis_schema_version",
    "admissibility_policy_version", "chunker_version", "record_layout_version",
    "normalization_version", "canonical_writer_version",
)






_CONSTRUCTION_HASHED_FIELDS = (
    "construction_schema_version",
    "profile_id",
    "api_provider",
    "build_llm_base_url",
    "build_llm_model",
    "build_llm_revision",
    "embed_base_url",
    "embed_model",
    "embed_revision",
    "embed_dim",
    "embedding_preprocess_version",
    "embedding_normalization",
    "embedding_distance",
    "embedding_dtype",
    "embedding_serialization_precision",
    "chunk_max_tokens",
    "chunk_overlap_tokens",
    "tokenizer_model",
    "extraction_batch_size",
    "extractor_temperature",
    "extractor_max_tokens",
    "extractor_schema_retries",
    "extraction_workers",
    "embedding_workers",
    "embedding_batch_size",
    "embedding_attempts",
    "alternate_embed_enabled",
    "extraction_prompt_version",
    "extraction_schema_version",
    "relation_admissibility_policy_version",
    "grounding_policy_version", "proposition_schema_projection_policy_version",
    "extraction_response_format_version",
    "chunker_version",
    "record_layout_version",
    "normalization_version",
    "canonical_writer_version",
)


@dataclass(frozen=True)
class ResolvedConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    construction_schema_version: str = CONSTRUCTION_SCHEMA_VERSION
    profile: str = COMPAT_PROFILE
    profile_id: str = COMPAT_PROFILE_ID

    api_provider: str = "default"
    build_llm_base_url: str = _DEFAULT_URL
    build_llm_api_key: str = field(default="", repr=False, compare=False)
    build_llm_model: str = "gpt-4o-mini"
    build_llm_revision: str = "unspecified"
    embed_base_url: str = _DEFAULT_URL
    embed_api_key: str = field(default="", repr=False, compare=False)
    embed_model: str = "text-embedding-3-small"
    embed_revision: str = "unspecified"
    embed_dim: int = 1536
    embedding_preprocess_version: str = "chain-canonical-entity-key-v1"
    embedding_normalization: str = "l2"
    embedding_distance: str = "cosine"
    embedding_dtype: str = "float32"
    embedding_serialization_precision: str = "float32-le"
    llm_base_url: str = _DEFAULT_URL
    llm_api_key: str = field(default="", repr=False, compare=False)
    llm_model: str = "gpt-4o-mini"
    llm_revision: str = "unspecified"
    alternate_token: str = field(default="", repr=False, compare=False)

    hyperskill_dir: str = str(BASE_DIR / "HyperSkill")
    hypergraph_dir: str = str(BASE_DIR / "datasets" / "KG" / "chain")
    event_dir: str = str(BASE_DIR / "datasets" / "events")
    knowledge_files_dir: str = str(BASE_DIR / "knowledge_files")
    results_dir: str = str(BASE_DIR / "results")
    dataset_dir: str = str(BASE_DIR / "dataset")

    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    tokenizer_model: str = "gpt-4o"
    extraction_batch_size: int = 5
    extractor_temperature: float = 0.0
    extractor_max_tokens: int = 4096
    extractor_schema_retries: int = 2
    extraction_workers: int = 256
    embedding_workers: int = 256
    embedding_batch_size: int = 100
    embedding_attempts: int = 3

    d_max: int = 4
    B_pi: int = 200
    F_max: Optional[int] = None
    tau_phi: float = 0.005
    theta: float = 0.5
    eta: float = 5.0
    k_sat: int = 10
    beta_0: float = 0.3
    Z: float = 5.0
    alpha_b: float = 0.5
    alpha_0: float = 0.6
    Omega_0: float = 0.3
    zeta: float = 3.0
    epsilon: float = 1e-6
    fusion_mode: str = "adaptive"
    fixed_alpha: Optional[float] = None
    ablation_mode: str = "none"
    tvf_enabled: bool = True
    ctvf_enabled: bool = True

    max_outer_loops: int = 3
    collect_rounds_per_phase: int = 3
    reason_rounds_per_phase: int = 1
    confidence_threshold: float = 0.75
    minimum_new_results: int = 2
    stability_window: int = 3
    stability_stdev: float = 0.02
    stability_mean: float = 0.45

    target_top_k: int = 5
    target_candidate_cap: int = 100
    target_min_similarity: float = 0.45
    target_min_margin: float = 0.03
    direction_entity_cap: int = 80
    direction_char_cap: int = 6000
    reasoning_hub_cap: int = 8
    reasoning_hub_multiplier: int = 2
    timeline_per_entity_cap: int = 5
    trajectory_char_cap: int = 2000
    causal_link_working_cap: int = 50
    causal_link_returned_cap: int = 20
    causal_link_prompt_cap: int = 12
    chain_start_cap: int = 20
    chain_working_cap: int = 50
    chain_returned_cap: int = 10
    chain_prompt_cap: int = 5
    hyperedge_working_cap: int = 60
    hyperedge_prompt_cap: int = 15
    contradiction_returned_cap: int = 4
    contradiction_prompt_cap: int = 3
    prior_outcome_cap: int = 6
    prior_description_char_cap: int = 100
    prior_serialized_char_cap: int = 600
    attribution_top_k: int = 5
    counterfactual_top_k: int = 3
    displayed_top_chain_cap: int = 8
    max_prompt_chars: int = 380000
    speculative_merge_threshold: float = 0.35
    speculative_merge_min_tokens: int = 2
    derivation_similarity_threshold: float = 0.55
    prune_threshold: float = 0.04
    serialized_path_cap: int = 10

    llm_retry_attempts: int = 5
    llm_temperature: float = 0.3
    llm_max_tokens: int = 8192
    json_correction_attempts: int = 2
    outcome_mapping_max_tokens: int = 1024
    outcome_sum_tolerance: float = 1e-6

    chain_eval_workers: int = 32
    chain_embed_concurrency: int = 8
    chain_skip_viz_embed: bool = False
    alternate_embed_enabled: bool = False

    ece_bins: int = 10

    ace_bins: int = 10
    reliability_bins: int = 10
    mce_min_bin_size: int = 5
    oc_uc_confidence_gap: float = 0.1
    nll_epsilon: float = 1e-7
    bootstrap_draws: int = 1000
    bootstrap_seed: int = 42
    canonical_float_format_version: str = "chain-float-json-v1"
    extraction_prompt_version: str = "chain-extraction-prompt-v17"
    extraction_schema_version: str = "chain-extraction-schema-v6"
    relation_admissibility_policy_version: str = (
        "chain-final-retry-relation-admissibility-v2"
    )
    grounding_policy_version: str = "chain-source-grounding-proposition-omission-v3"
    proposition_schema_projection_policy_version: str = (
        "chain-final-retry-empty-or-duplicate-canonical-entity-proposition-omission-v2"
    )
    extraction_response_format_version: str = EXTRACTION_RESPONSE_FORMAT_VERSION
    reasoning_prompt_version: str = REASONING_PROMPT_VERSION
    outcome_mapping_schema_version: str = "chain-outcome-mapping-v1"
    binary_axis_schema_version: str = BINARY_AXIS_SCHEMA_VERSION
    admissibility_policy_version: str = "chain-cutoff-admissibility-v1"
    chunker_version: str = "chain-token-chunker-v1"
    record_layout_version: str = "chain-record-layout-v1"
    normalization_version: str = "chain-unicode-normalization-v1"
    canonical_writer_version: str = "chain-canonical-writer-v1"

    deprecation_warnings: tuple[str, ...] = ()
    deprecated_max_reasoning_rounds: Optional[int] = None

    construction_config_sha256: str = field(init=False)
    scientific_config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(
            self,
            "construction_config_sha256",
            hash_payload(self.construction_payload()),
        )
        object.__setattr__(
            self,
            "scientific_config_sha256",
            hash_payload(self.scientific_payload()),
        )

    @property
    def is_paper(self) -> bool:
        return self.profile == PAPER_PROFILE

    @property
    def publishable(self) -> bool:
        return self.is_paper

    @property
    def out_of_paper_protocol(self) -> bool:
        return not self.is_paper

    @property
    def coverage_normalizer(self) -> float:
        return self.Z

    @property
    def max_causal_depth(self) -> int:
        return self.d_max

    @property
    def alpha_base(self) -> float:
        return self.alpha_b

    @property
    def alpha_adaptive(self) -> bool:
        return self.fusion_mode == "adaptive"

    @property
    def embedding_signature(self) -> dict[str, Any]:
        return {
            "provider": self.api_provider,
            "endpoint_identity": endpoint_identity(self.embed_base_url),
            "model": self.embed_model,
            "revision": self.embed_revision,
            "dimension": self.embed_dim,
            "dtype": self.embedding_dtype,
            "serialization_precision": self.embedding_serialization_precision,
            "preprocess_version": self.embedding_preprocess_version,
            "normalization": self.embedding_normalization,
            "distance": self.embedding_distance,
        }

    def scientific_payload(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in _HASHED_FIELDS}
        for name in ("build_llm_base_url", "embed_base_url", "llm_base_url"):
            payload[name.replace("base_url", "endpoint_identity")] = endpoint_identity(
                payload.pop(name)
            )
        return payload

    def construction_payload(self) -> dict[str, Any]:


        payload = {
            name: getattr(self, name) for name in _CONSTRUCTION_HASHED_FIELDS
        }
        for name in ("build_llm_base_url", "embed_base_url"):
            payload[name.replace("base_url", "endpoint_identity")] = endpoint_identity(
                payload.pop(name)
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self.scientific_payload()
        payload.update(
            {
                "profile": self.profile,
                "publishable": self.publishable,
                "out_of_paper_protocol": self.out_of_paper_protocol,
                "construction_config_sha256": self.construction_config_sha256,
                "scientific_config_sha256": self.scientific_config_sha256,
                "deprecation_warnings": list(self.deprecation_warnings),
                "deprecated_max_reasoning_rounds": self.deprecated_max_reasoning_rounds,
            }
        )
        return payload

    def _validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(f"unsupported config schema: {self.schema_version}")
        if self.construction_schema_version != CONSTRUCTION_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported construction schema: {self.construction_schema_version}"
            )
        if self.profile not in {COMPAT_PROFILE, PAPER_PROFILE}:
            raise ConfigurationError("profile must be compat or paper")
        expected_profile_id = PAPER_PROFILE_ID if self.is_paper else COMPAT_PROFILE_ID
        if self.profile_id != expected_profile_id:
            raise ConfigurationError(
                f"profile_id {self.profile_id!r} does not match profile {self.profile!r}"
            )
        positive_ints = (
            "embed_dim", "chunk_max_tokens", "extraction_batch_size", "extractor_max_tokens",
            "extraction_workers", "embedding_workers", "embedding_batch_size",
            "embedding_attempts", "d_max", "B_pi", "k_sat", "max_outer_loops",
            "collect_rounds_per_phase", "reason_rounds_per_phase", "minimum_new_results",
            "stability_window", "target_top_k", "target_candidate_cap",
            "direction_entity_cap", "direction_char_cap", "reasoning_hub_cap",
            "reasoning_hub_multiplier", "timeline_per_entity_cap", "trajectory_char_cap",
            "causal_link_working_cap", "causal_link_returned_cap", "causal_link_prompt_cap",
            "chain_start_cap", "chain_working_cap", "chain_returned_cap", "chain_prompt_cap",
            "hyperedge_working_cap", "hyperedge_prompt_cap", "contradiction_returned_cap",
            "contradiction_prompt_cap", "prior_outcome_cap", "prior_description_char_cap",
            "prior_serialized_char_cap", "attribution_top_k", "counterfactual_top_k",
            "displayed_top_chain_cap", "max_prompt_chars", "speculative_merge_min_tokens",
            "serialized_path_cap", "llm_retry_attempts", "llm_max_tokens",
            "outcome_mapping_max_tokens", "chain_eval_workers",
            "chain_embed_concurrency", "ece_bins", "ace_bins", "reliability_bins",
            "mce_min_bin_size", "bootstrap_draws",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if (
            isinstance(self.chunk_overlap_tokens, bool)
            or not isinstance(self.chunk_overlap_tokens, int)
            or self.chunk_overlap_tokens < 0
            or self.chunk_overlap_tokens >= self.chunk_max_tokens
        ):
            raise ConfigurationError("chunk_overlap_tokens must be in [0, chunk_max_tokens)")
        if (
            isinstance(self.extractor_schema_retries, bool)
            or not isinstance(self.extractor_schema_retries, int)
            or self.extractor_schema_retries < 0
        ):
            raise ConfigurationError("extractor_schema_retries must be >= 0")
        if (
            isinstance(self.json_correction_attempts, bool)
            or not isinstance(self.json_correction_attempts, int)
            or self.json_correction_attempts < 0
        ):
            raise ConfigurationError("json_correction_attempts must be >= 0")
        if self.F_max is not None and (
            isinstance(self.F_max, bool) or not isinstance(self.F_max, int) or self.F_max <= 0
        ):
            raise ConfigurationError("F_max must be None or a positive integer")
        intervals = (
            ("tau_phi", 0.0, 1.0, False, False),
            ("theta", 0.0, 1.0, True, True),
            ("beta_0", 0.0, 1.0, True, True),
            ("alpha_0", 0.0, 1.0, True, False),
            ("Omega_0", 0.0, 1.0, False, False),
            ("confidence_threshold", 0.0, 1.0, False, False),
            ("stability_mean", 0.0, 1.0, False, False),
            ("target_min_similarity", -1.0, 1.0, False, False),
            ("target_min_margin", 0.0, 2.0, False, False),
            ("speculative_merge_threshold", 0.0, 1.0, False, False),
            ("derivation_similarity_threshold", 0.0, 1.0, False, False),
            ("prune_threshold", 0.0, 1.0, False, False),
        )
        for name, lower, upper, lower_open, upper_open in intervals:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigurationError(f"{name} must be a finite number, not bool")
            lower_ok = value > lower if lower_open else value >= lower
            upper_ok = value < upper if upper_open else value <= upper
            if not (lower_ok and upper_ok):
                raise ConfigurationError(f"{name} is outside its allowed interval")
        for name in (
            "eta", "Z", "alpha_b", "zeta", "epsilon", "stability_stdev",
            "outcome_sum_tolerance", "nll_epsilon",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigurationError(f"{name} must be finite and > 0")
        if self.is_paper and not self.alpha_0 < 2 * self.alpha_b:
            raise ConfigurationError("paper adaptive domain requires alpha_0 < 2 * alpha_b")
        if self.fusion_mode not in {"adaptive", "fixed"}:
            raise ConfigurationError("fusion_mode must be adaptive or fixed")
        if self.fusion_mode == "fixed":
            if (
                self.fixed_alpha is None
                or isinstance(self.fixed_alpha, bool)
                or not isinstance(self.fixed_alpha, (int, float))
                or not math.isfinite(self.fixed_alpha)
                or not 0.0 <= self.fixed_alpha <= 1.0
            ):
                raise ConfigurationError("fixed fusion requires fixed_alpha in [0,1]")
        elif self.fixed_alpha is not None:
            raise ConfigurationError("fixed_alpha is only valid with fusion_mode=fixed")
        if self.ablation_mode not in ABLATION_MODES:
            raise ConfigurationError("invalid ablation_mode")
        if self.is_paper and self.ablation_mode in COMPAT_ONLY_ABLATION_MODES:
            raise ConfigurationError(
                f"paper profile rejects compat-only ablation_mode={self.ablation_mode!r}"
            )
        expected_tvf, expected_ctvf = _ABLATION_TVF_FLAGS[self.ablation_mode]
        if (self.tvf_enabled, self.ctvf_enabled) != (expected_tvf, expected_ctvf):
            raise ConfigurationError(
                "tvf_enabled/ctvf_enabled do not match versioned ablation_mode; "
                "select infer_tvf, without_tvf, or without_all explicitly instead "
                "of using legacy TVF/CTVF switches"
            )
        for name in (
            "tvf_enabled", "ctvf_enabled", "chain_skip_viz_embed",
            "alternate_embed_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"{name} must be bool")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int):
            raise ConfigurationError("bootstrap_seed must be an integer, not bool")
        if self.epsilon != 1e-6:
            raise ConfigurationError("all current profiles fix epsilon at 1e-6")
        for name in (
            "api_provider", "build_llm_model", "build_llm_revision", "embed_model", "embed_revision",
            "embedding_preprocess_version", "embedding_normalization",
            "embedding_distance", "embedding_dtype", "embedding_serialization_precision",
            "llm_model", "llm_revision", "extraction_prompt_version",
            "extraction_schema_version", "relation_admissibility_policy_version",
            "grounding_policy_version", "proposition_schema_projection_policy_version",
            "extraction_response_format_version",
            "reasoning_prompt_version", "outcome_mapping_schema_version",
            "binary_axis_schema_version", "admissibility_policy_version",
            "chunker_version", "record_layout_version", "normalization_version",
            "canonical_writer_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ConfigurationError(f"{name} must be non-empty")
        if self.extraction_response_format_version != EXTRACTION_RESPONSE_FORMAT_VERSION:
            raise ConfigurationError(
                "unsupported extraction_response_format_version: "
                f"{self.extraction_response_format_version!r}"
            )
        if self.is_paper and any(
            getattr(self, name).strip().casefold() == "unspecified"
            for name in ("build_llm_revision", "embed_revision", "llm_revision")
        ):
            raise ConfigurationError("paper profile requires explicit, non-placeholder model revisions")
        for name in ("build_llm_base_url", "embed_base_url", "llm_base_url"):
            _validate_endpoint(getattr(self, name), name)
        if self.embedding_dtype != "float32":
            raise ConfigurationError("embedding_dtype must be float32")
        if self.embedding_normalization != "l2":
            raise ConfigurationError("embedding_normalization must be l2")
        if self.embedding_serialization_precision != "float32-le":
            raise ConfigurationError("embedding_serialization_precision must be float32-le")
        if self.alternate_embed_enabled:
            raise ConfigurationError(
                "CHAIN_USE_ALTERNATE_EMBED is unsupported: no sealed compatible embedding path"
            )
        if isinstance(self.llm_temperature, bool) or not isinstance(self.llm_temperature, (int, float)):
            raise ConfigurationError("llm_temperature must be a finite number, not bool")
        if not math.isfinite(self.llm_temperature) or not 0.0 <= self.llm_temperature <= 2.0:
            raise ConfigurationError("llm_temperature must be in [0,2]")
        if isinstance(self.extractor_temperature, bool) or not isinstance(self.extractor_temperature, (int, float)):
            raise ConfigurationError("extractor_temperature must be a finite number, not bool")
        if not math.isfinite(self.extractor_temperature) or not 0.0 <= self.extractor_temperature <= 2.0:
            raise ConfigurationError("extractor_temperature must be in [0,2]")
        if (
            isinstance(self.oc_uc_confidence_gap, bool)
            or not isinstance(self.oc_uc_confidence_gap, (int, float))
            or not math.isfinite(float(self.oc_uc_confidence_gap))
            or not 0.0 <= float(self.oc_uc_confidence_gap) <= 1.0
        ):
            raise ConfigurationError("oc_uc_confidence_gap must be finite and in [0,1]")
def load_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigurationError(f"duplicate JSON key in config: {key!r}")
            result[key] = value
        return result
    try:
        raw = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ConfigurationError(f"non-finite JSON constant in config: {value}")
            ),
        )
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"config file is not valid JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("config file root must be a JSON object")
    return raw


_NESTED_GROUPS = {
    "construction", "inference", "orchestration", "provider", "metrics", "paths", "environment",
}


def _flatten_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _NESTED_GROUPS:
            if not isinstance(value, Mapping):
                raise ConfigurationError(f"config group {key!r} must be an object")
            for nested_key, nested_value in value.items():
                if nested_key in flat:
                    raise ConfigurationError(f"duplicate config field: {nested_key}")
                flat[nested_key] = nested_value
        else:
            if key in flat:
                raise ConfigurationError(f"duplicate config field: {key}")
            flat[key] = value
    if "coverage_normalizer" in flat:
        if "Z" in flat:
            raise ConfigurationError("use only one of Z or coverage_normalizer")
        flat["Z"] = flat.pop("coverage_normalizer")
    return flat


_ENV_FIELD_SPECS = {
    "API_PROVIDER": ("api_provider", str),
    "BUILD_LLM_BASE_URL": ("build_llm_base_url", str),
    "BUILD_LLM_API_KEY": ("build_llm_api_key", str),
    "BUILD_LLM_MODEL": ("build_llm_model", str),
    "BUILD_LLM_REVISION": ("build_llm_revision", str),
    "EMBED_BASE_URL": ("embed_base_url", str),
    "EMBED_API_KEY": ("embed_api_key", str),
    "EMBED_MODEL": ("embed_model", str),
    "EMBED_REVISION": ("embed_revision", str),
    "EMBED_DIM": ("embed_dim", lambda v: _strict_int(v, "EMBED_DIM", minimum=1)),
    "LLM_BASE_URL": ("llm_base_url", str),
    "LLM_API_KEY": ("llm_api_key", str),
    "LLM_MODEL": ("llm_model", str),
    "LLM_REVISION": ("llm_revision", str),
    "HYPERSKILL_DIR": ("hyperskill_dir", str),
    "HYPERGRAPH_DIR": ("hypergraph_dir", str),
    "EVENT_DIR": ("event_dir", str),
    "KNOWLEDGE_FILES_DIR": ("knowledge_files_dir", str),
    "RESULTS_DIR": ("results_dir", str),
    "DATASET_DIR": ("dataset_dir", str),
    "MAX_OUTER_LOOPS": ("max_outer_loops", lambda v: _strict_int(v, "MAX_OUTER_LOOPS", minimum=1)),
    "COLLECT_ROUNDS_PER_PHASE": ("collect_rounds_per_phase", lambda v: _strict_int(v, "COLLECT_ROUNDS_PER_PHASE", minimum=1)),
    "REASON_ROUNDS_PER_PHASE": ("reason_rounds_per_phase", lambda v: _strict_int(v, "REASON_ROUNDS_PER_PHASE", minimum=1)),
    "CONFIDENCE_THRESHOLD": ("confidence_threshold", lambda v: _strict_float(v, "CONFIDENCE_THRESHOLD")),
    "TVF_ENABLED": ("tvf_enabled", lambda v: _strict_bool(v, "TVF_ENABLED")),
    "TVF_TAU": ("eta", lambda v: _strict_float(v, "TVF_TAU")),
    "CTVF_ENABLED": ("ctvf_enabled", lambda v: _strict_bool(v, "CTVF_ENABLED")),
    "MAX_CAUSAL_DEPTH": ("d_max", lambda v: _strict_int(v, "MAX_CAUSAL_DEPTH", minimum=1)),
    "ALPHA_BASE": ("alpha_b", lambda v: _strict_float(v, "ALPHA_BASE")),
    "CHAIN_EVAL_WORKERS": ("chain_eval_workers", lambda v: _strict_int(v, "CHAIN_EVAL_WORKERS", minimum=1)),
    "CHAIN_EMBED_CONCURRENCY": ("chain_embed_concurrency", lambda v: _strict_int(v, "CHAIN_EMBED_CONCURRENCY", minimum=1)),
    "CHAIN_SKIP_VIZ_EMBED": ("chain_skip_viz_embed", lambda v: _strict_bool(v, "CHAIN_SKIP_VIZ_EMBED")),
    "CHAIN_USE_ALTERNATE_EMBED": ("alternate_embed_enabled", lambda v: _strict_bool(v, "CHAIN_USE_ALTERNATE_EMBED")),
}


_PAPER_SCIENTIFIC_ENV = {
    "API_PROVIDER", "BUILD_LLM_BASE_URL", "BUILD_LLM_MODEL", "BUILD_LLM_REVISION",
    "EMBED_BASE_URL", "EMBED_MODEL", "EMBED_REVISION", "EMBED_DIM",
    "LLM_BASE_URL", "LLM_MODEL", "LLM_REVISION", "MAX_OUTER_LOOPS",
    "COLLECT_ROUNDS_PER_PHASE", "REASON_ROUNDS_PER_PHASE", "CONFIDENCE_THRESHOLD",
    "TVF_ENABLED", "TVF_TAU", "CTVF_ENABLED", "MAX_CAUSAL_DEPTH", "ALPHA_BASE",
    "ALPHA_ADAPTIVE", "CHAIN_USE_ALTERNATE_EMBED",
}


def _coerce_field(name: str, value: Any) -> Any:
    int_fields = {
        "embed_dim", "chunk_max_tokens", "chunk_overlap_tokens", "extraction_batch_size",
        "extractor_max_tokens", "extractor_schema_retries", "extraction_workers",
        "embedding_workers", "embedding_batch_size", "embedding_attempts", "d_max",
        "B_pi", "k_sat", "max_outer_loops", "collect_rounds_per_phase",
        "reason_rounds_per_phase", "minimum_new_results", "stability_window",
        "target_top_k", "target_candidate_cap", "direction_entity_cap",
        "direction_char_cap", "reasoning_hub_cap", "reasoning_hub_multiplier",
        "timeline_per_entity_cap", "trajectory_char_cap", "causal_link_working_cap",
        "causal_link_returned_cap", "causal_link_prompt_cap", "chain_start_cap",
        "chain_working_cap", "chain_returned_cap", "chain_prompt_cap",
        "hyperedge_working_cap", "hyperedge_prompt_cap", "contradiction_returned_cap",
        "contradiction_prompt_cap", "prior_outcome_cap", "prior_description_char_cap",
        "prior_serialized_char_cap", "attribution_top_k", "counterfactual_top_k",
        "displayed_top_chain_cap", "max_prompt_chars", "speculative_merge_min_tokens",
        "serialized_path_cap", "llm_retry_attempts", "llm_max_tokens",
        "json_correction_attempts", "outcome_mapping_max_tokens", "chain_eval_workers",
        "chain_embed_concurrency", "ece_bins", "ace_bins", "reliability_bins",
        "mce_min_bin_size", "bootstrap_draws", "bootstrap_seed",
    }
    float_fields = {
        "extractor_temperature", "tau_phi", "theta", "eta", "beta_0", "Z",
        "alpha_b", "alpha_0", "Omega_0", "zeta", "epsilon", "fixed_alpha",
        "confidence_threshold", "stability_stdev", "stability_mean",
        "target_min_similarity", "target_min_margin", "speculative_merge_threshold",
        "derivation_similarity_threshold", "prune_threshold", "llm_temperature",
        "outcome_sum_tolerance", "oc_uc_confidence_gap", "nll_epsilon",
    }
    bool_fields = {
        "tvf_enabled", "ctvf_enabled", "chain_skip_viz_embed", "alternate_embed_enabled",
    }
    if name == "F_max":
        return _optional_positive_int(value, name)
    if name == "api_provider":
        parsed = str(value).strip().lower()
        if not parsed:
            raise ConfigurationError("api_provider must be non-empty")
        return parsed
    if name in {
        "build_llm_model", "build_llm_revision", "embed_model", "embed_revision",
        "llm_model", "llm_revision",
        "embedding_preprocess_version", "embedding_normalization",
        "embedding_distance", "embedding_dtype", "embedding_serialization_precision",
        "extraction_prompt_version", "extraction_schema_version",
        "relation_admissibility_policy_version", "grounding_policy_version",
        "proposition_schema_projection_policy_version",
        "extraction_response_format_version",
    }:
        parsed = str(value).strip()
        if not parsed:
            raise ConfigurationError(f"{name} must be non-empty")
        return parsed
    if name in {"build_llm_base_url", "embed_base_url", "llm_base_url"}:
        return _validate_endpoint(value, name)
    if name in int_fields:
        minimum = 0 if name in {
            "chunk_overlap_tokens", "extractor_schema_retries", "json_correction_attempts",
        } else None
        return _strict_int(value, name, minimum=minimum)
    if name in float_fields:
        if value is None and name == "fixed_alpha":
            return None
        return _strict_float(value, name)
    if name in bool_fields:
        return _strict_bool(value, name)
    if name == "deprecation_warnings":
        if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
            raise ConfigurationError("deprecation_warnings must be a string list")
        return tuple(value)
    return value


def _reject_paper_ambient_overrides(
    candidate: ResolvedConfig, environ: Mapping[str, str]
) -> None:


    for env_name in sorted(_PAPER_SCIENTIFIC_ENV):
        if env_name not in environ:
            continue
        if env_name == "ALPHA_ADAPTIVE":
            expected = candidate.fusion_mode == "adaptive"
            actual = _strict_bool(environ[env_name], env_name)
        else:
            field_name, parser = _ENV_FIELD_SPECS[env_name]
            parsed_env = (
                parser(environ[env_name]) if parser is not str else str(environ[env_name])
            )
            actual = _coerce_field(field_name, parsed_env)
            expected = getattr(candidate, field_name)
        if actual != expected:
            raise ConfigurationError(
                f"paper profile rejects ambient override {env_name}={environ[env_name]!r}"
            )
    if "OPENAI_BASE_URL" in environ:
        raise ConfigurationError(
            "paper profile rejects ambient OPENAI_BASE_URL; use explicit role endpoints"
        )


def _canonicalise_ablation_flags(
    values: dict[str, Any],
    *,
    environ: Mapping[str, str],
    raw_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> None:


    mode = values.get("ablation_mode", "none")
    if mode not in _ABLATION_TVF_FLAGS:

        return
    expected_tvf, expected_ctvf = _ABLATION_TVF_FLAGS[mode]
    expectations = {
        "tvf_enabled": ("TVF_ENABLED", expected_tvf),
        "ctvf_enabled": ("CTVF_ENABLED", expected_ctvf),
    }
    for field_name, (env_name, expected) in expectations.items():
        supplied: list[tuple[str, bool]] = []
        if env_name in environ:
            supplied.append(
                (env_name, _strict_bool(environ[env_name], env_name))
            )
        if field_name in raw_config:
            supplied.append(
                (
                    f"config.{field_name}",
                    _strict_bool(raw_config[field_name], field_name),
                )
            )
        if field_name in overrides:
            supplied.append(
                (
                    f"override.{field_name}",
                    _strict_bool(overrides[field_name], field_name),
                )
            )
        for source, actual in supplied:
            if actual != expected:
                raise ConfigurationError(
                    f"{source}={actual!r} conflicts with ablation_mode={mode!r}; "
                    "use the explicit versioned ablation mode without contradictory "
                    "legacy TVF/CTVF switches"
                )
    values["tvf_enabled"] = expected_tvf
    values["ctvf_enabled"] = expected_ctvf


def resolve_config(
    profile: str = COMPAT_PROFILE,
    config: Optional[ResolvedConfig | Mapping[str, Any] | str | os.PathLike[str]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ResolvedConfig:


    environ: Mapping[str, str] = os.environ if env is None else env
    profile_name = _normalise_profile(profile)
    if isinstance(config, ResolvedConfig):
        if overrides:
            raise ConfigurationError("cannot override an already resolved config")
        if config.profile != profile_name:
            raise ConfigurationError("resolved config profile mismatch")
        if config.is_paper:
            _reject_paper_ambient_overrides(config, environ)
        return config

    profile_id = PAPER_PROFILE_ID if profile_name == PAPER_PROFILE else COMPAT_PROFILE_ID
    values: dict[str, Any] = {
        "profile": profile_name,
        "profile_id": profile_id,
        **_provider_defaults({}),
    }

    raw_config: dict[str, Any] = {}
    if config is not None:
        if isinstance(config, (str, os.PathLike)):
            raw_config = load_config_file(config)
        elif isinstance(config, Mapping):
            raw_config = dict(config)
        else:
            raise ConfigurationError("config must be a mapping, path, or ResolvedConfig")
        raw_config = _flatten_config(raw_config)

    valid_init_fields = {item.name for item in fields(ResolvedConfig) if item.init}
    allowed_metadata = {
        "publishable", "out_of_paper_protocol", "construction_config_sha256",
        "scientific_config_sha256",
    }
    unknown = set(raw_config) - valid_init_fields - allowed_metadata
    if unknown:
        raise ConfigurationError(f"unknown config field(s): {', '.join(sorted(unknown))}")
    supplied_metadata = {
        key: raw_config.pop(key) for key in tuple(allowed_metadata) if key in raw_config
    }
    if "profile" in raw_config and _normalise_profile(raw_config["profile"]) != profile_name:
        raise ConfigurationError("config profile does not match the selected run mode")
    if "profile_id" in raw_config and raw_config["profile_id"] != profile_id:
        raise ConfigurationError("config profile_id does not match the selected run mode")
    raw_config["profile"] = profile_name
    raw_config["profile_id"] = profile_id
    deprecations: list[str] = []
    if profile_name == COMPAT_PROFILE:



        values.update(_provider_defaults(environ))
        for env_name, (field_name, parser) in _ENV_FIELD_SPECS.items():
            if env_name in environ:
                raw_value = environ[env_name]
                parsed = parser(raw_value) if parser is not str else str(raw_value)
                values[field_name] = _coerce_field(field_name, parsed)
        if "ALPHA_ADAPTIVE" in environ:
            adaptive = _strict_bool(environ["ALPHA_ADAPTIVE"], "ALPHA_ADAPTIVE")
            values["fusion_mode"] = "adaptive" if adaptive else "fixed"
            if not adaptive:
                values["fixed_alpha"] = values.get("alpha_b", 0.5)
        if "MAX_REASONING_ROUNDS" in environ:
            deprecated_value = _strict_int(
                environ["MAX_REASONING_ROUNDS"], "MAX_REASONING_ROUNDS", minimum=1
            )
            values["deprecated_max_reasoning_rounds"] = deprecated_value
            deprecations.append("MAX_REASONING_ROUNDS is deprecated and has no effect")
            warnings.warn(deprecations[-1], DeprecationWarning, stacklevel=2)
    else:
        required_explicit = {
            "api_provider", "build_llm_base_url", "build_llm_model",
            "build_llm_revision", "embed_base_url", "embed_model", "embed_revision",
            "embed_dim", "llm_base_url", "llm_model", "llm_revision",
        }
        missing_explicit = required_explicit - set(raw_config)
        if missing_explicit:
            raise ConfigurationError(
                "paper profile requires explicit provider/base/model fields: "
                + ", ".join(sorted(missing_explicit))
            )


        openai_key = environ.get("OPENAI_API_KEY", values.get("llm_api_key", ""))
        values["build_llm_api_key"] = environ.get("BUILD_LLM_API_KEY", openai_key)
        values["embed_api_key"] = environ.get("EMBED_API_KEY", openai_key)
        values["llm_api_key"] = environ.get("LLM_API_KEY", openai_key)
        if "MAX_REASONING_ROUNDS" in environ:
            raise ConfigurationError("paper profile rejects deprecated MAX_REASONING_ROUNDS")



    for key, value in raw_config.items():
        values[key] = _coerce_field(key, value)

    override_flat: dict[str, Any] = {}
    if overrides:
        override_flat = _flatten_config(overrides)
        unknown_overrides = set(override_flat) - valid_init_fields
        if unknown_overrides:
            raise ConfigurationError(
                f"unknown override field(s): {', '.join(sorted(unknown_overrides))}"
            )
        for key, value in override_flat.items():
            if key not in {"profile", "profile_id"}:
                values[key] = _coerce_field(key, value)

    _canonicalise_ablation_flags(
        values,
        environ=environ,
        raw_config=raw_config,
        overrides=override_flat,
    )

    if profile_name == PAPER_PROFILE:
        candidate = ResolvedConfig(**values)
        _reject_paper_ambient_overrides(candidate, environ)

    values["deprecation_warnings"] = tuple(sorted(set(deprecations)))
    resolved = ResolvedConfig(**values)
    checks = {
        "construction_config_sha256": resolved.construction_config_sha256,
        "scientific_config_sha256": resolved.scientific_config_sha256,
        "publishable": resolved.publishable,
        "out_of_paper_protocol": resolved.out_of_paper_protocol,
    }
    for key, expected in supplied_metadata.items():
        if expected != checks[key]:
            raise ConfigurationError(f"supplied {key} does not match resolved value")
    return resolved


def validate_binary_axis(value: Mapping[str, Any]) -> dict[str, Any]:


    if not isinstance(value, Mapping):
        raise ConfigurationError("binary_axis must be an object")
    allowed = {
        "schema_version", "positive", "negative", "source", "adapter_id",
        "binary_axis_sha256",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"unknown binary_axis field(s): {', '.join(sorted(unknown))}")
    schema = value.get("schema_version", BINARY_AXIS_SCHEMA_VERSION)
    if schema != BINARY_AXIS_SCHEMA_VERSION:
        raise ConfigurationError(f"unsupported binary axis schema: {schema}")

    def side(name: str) -> dict[str, Any]:
        raw = value.get(name)
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"binary_axis.{name} must be an object")
        unknown_side = set(raw) - {"id", "label", "semantics", "aliases"}
        if unknown_side:
            raise ConfigurationError(
                f"unknown binary_axis.{name} field(s): {', '.join(sorted(unknown_side))}"
            )
        side_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", "")).strip()
        semantics = str(raw.get("semantics", label)).strip()
        aliases_raw = raw.get("aliases", [])
        if not side_id or not label or not semantics:
            raise ConfigurationError(f"binary_axis.{name} requires id, label, and semantics")
        if not isinstance(aliases_raw, (list, tuple)) or not all(
            isinstance(item, str) for item in aliases_raw
        ):
            raise ConfigurationError(f"binary_axis.{name}.aliases must be a string list")
        return {
            "id": side_id,
            "label": label,
            "semantics": semantics,
            "aliases": sorted({item.strip() for item in aliases_raw if item.strip()}),
        }

    positive = side("positive")
    negative = side("negative")

    def axis_token(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    if axis_token(positive["id"]) == axis_token(negative["id"]):
        raise ConfigurationError("binary axis sides must have distinct IDs")
    positive_tokens = {
        axis_token(item)
        for item in (
            positive["id"], positive["label"], positive["semantics"],
            *positive["aliases"],
        )
        if axis_token(item)
    }
    negative_tokens = {
        axis_token(item)
        for item in (
            negative["id"], negative["label"], negative["semantics"],
            *negative["aliases"],
        )
        if axis_token(item)
    }
    overlap = positive_tokens & negative_tokens
    if overlap:
        raise ConfigurationError(
            "binary axis sides have overlapping normalized IDs/labels/semantics/aliases: "
            f"{sorted(overlap)}"
        )
    payload = {
        "schema_version": schema,
        "positive": positive,
        "negative": negative,
        "source": str(value.get("source", "explicit")).strip() or "explicit",
        "adapter_id": str(value.get("adapter_id", "")).strip(),
    }
    payload["binary_axis_sha256"] = hash_payload(payload)
    supplied_hash = str(value.get("binary_axis_sha256", "")).strip()
    if supplied_hash and supplied_hash != payload["binary_axis_sha256"]:
        raise ConfigurationError("binary_axis_sha256 mismatch")
    return payload


def legacy_yes_no_axis() -> dict[str, Any]:
    return validate_binary_axis(
        {
            "schema_version": BINARY_AXIS_SCHEMA_VERSION,
            "positive": {
                "id": "yes",
                "label": "Yes",
                "semantics": "event occurs",
                "aliases": ["yes", "true", "1"],
            },
            "negative": {
                "id": "no",
                "label": "No",
                "semantics": "event does not occur",
                "aliases": ["no", "false", "0"],
            },
            "source": "legacy_adapter",
            "adapter_id": LEGACY_AXIS_ADAPTER_ID,
        }
    )



DEFAULT_CONFIG = resolve_config(COMPAT_PROFILE)

API_PROVIDER = DEFAULT_CONFIG.api_provider
ALTERNATE_TOKEN = DEFAULT_CONFIG.alternate_token
OPENAI_API_KEY = DEFAULT_CONFIG.llm_api_key
OPENAI_BASE_URL = DEFAULT_CONFIG.llm_base_url
BUILD_LLM_BASE_URL = DEFAULT_CONFIG.build_llm_base_url
BUILD_LLM_API_KEY = DEFAULT_CONFIG.build_llm_api_key
BUILD_LLM_MODEL = DEFAULT_CONFIG.build_llm_model
BUILD_LLM_REVISION = DEFAULT_CONFIG.build_llm_revision
EMBED_BASE_URL = DEFAULT_CONFIG.embed_base_url
EMBED_API_KEY = DEFAULT_CONFIG.embed_api_key
EMBED_MODEL = DEFAULT_CONFIG.embed_model
EMBED_REVISION = DEFAULT_CONFIG.embed_revision
EMBED_DIM = DEFAULT_CONFIG.embed_dim
LLM_BASE_URL = DEFAULT_CONFIG.llm_base_url
LLM_API_KEY = DEFAULT_CONFIG.llm_api_key
LLM_MODEL = DEFAULT_CONFIG.llm_model
LLM_REVISION = DEFAULT_CONFIG.llm_revision

HYPERSKILL_DIR = DEFAULT_CONFIG.hyperskill_dir
DATA_DIR = str(BASE_DIR / "datasets")
HYPERGRAPH_DIR = DEFAULT_CONFIG.hypergraph_dir
EVENT_DIR = DEFAULT_CONFIG.event_dir
KNOWLEDGE_FILES_DIR = DEFAULT_CONFIG.knowledge_files_dir
RESULTS_DIR = DEFAULT_CONFIG.results_dir
DATASET_DIR = DEFAULT_CONFIG.dataset_dir

MAX_REASONING_ROUNDS = None
CONFIDENCE_THRESHOLD = DEFAULT_CONFIG.confidence_threshold
MAX_OUTER_LOOPS = DEFAULT_CONFIG.max_outer_loops
COLLECT_ROUNDS_PER_PHASE = DEFAULT_CONFIG.collect_rounds_per_phase
REASON_ROUNDS_PER_PHASE = DEFAULT_CONFIG.reason_rounds_per_phase
TVF_ENABLED = DEFAULT_CONFIG.tvf_enabled
TVF_TAU = DEFAULT_CONFIG.eta
CTVF_ENABLED = DEFAULT_CONFIG.ctvf_enabled
MAX_CAUSAL_DEPTH = DEFAULT_CONFIG.d_max
ALPHA_BASE = DEFAULT_CONFIG.alpha_b
ALPHA_ADAPTIVE = DEFAULT_CONFIG.alpha_adaptive
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = _strict_int(os.getenv("WEB_PORT", "8765"), "WEB_PORT", minimum=1)
