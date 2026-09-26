from __future__ import annotations

import math
from numbers import Real
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ADMISSIBLE_PRE_CUTOFF = "admissible_pre_cutoff"
ADMISSIBLE_FALLBACK = "admissible_fallback"
INADMISSIBLE = "inadmissible"

_DATE_ONLY_PATTERNS = (
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), "%m/%d/%Y"),
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validated_similarity(value: Any) -> float:


    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("stored recurrence similarity must be a real number in [0, 1]")
    similarity = float(value)
    if not math.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
        raise ValueError("stored recurrence similarity must be finite and in [0, 1]")
    return similarity


@dataclass(frozen=True)
class ParsedTemporal:


    value: Any
    precision: str
    timezone_explicit: bool
    canonical: str
    raw: str


@dataclass(frozen=True)
class AdmissibilityDecision:
    status: str
    reason: str
    cutoff: str
    evidence_time: str = ""
    availability_upper_bound: str = ""
    delta_days: Optional[float] = None
    fallback_score: Optional[float] = None

    @property
    def admitted(self) -> bool:
        return self.status in {ADMISSIBLE_PRE_CUTOFF, ADMISSIBLE_FALLBACK}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "cutoff": self.cutoff,
            "evidence_time": self.evidence_time,
            "availability_upper_bound": self.availability_upper_bound,
            "delta_days": self.delta_days,
            "fallback_score": self.fallback_score,
        }


def canonical_proposition_key(value: Any) -> str:


    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.strip().split())


def parse_temporal(value: Any) -> Optional[ParsedTemporal]:


    if value is None:
        return None
    if isinstance(value, datetime):
        raw = value.isoformat()
        explicit_tz = value.tzinfo is not None and value.utcoffset() is not None
        parsed = value.astimezone(timezone.utc) if explicit_tz else value
        return ParsedTemporal(
            parsed, "instant", explicit_tz, parsed.isoformat(), raw
        )
    if isinstance(value, date):
        raw = value.isoformat()
        return ParsedTemporal(value, "date", False, raw, raw)
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None
    for pattern, fmt in _DATE_ONLY_PATTERNS:
        if pattern.fullmatch(raw):
            try:
                parsed_date = datetime.strptime(raw, fmt).date()
            except ValueError:
                return None
            return ParsedTemporal(
                parsed_date, "date", False, parsed_date.isoformat(), raw
            )

    iso_raw = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed_dt = datetime.fromisoformat(iso_raw)
    except ValueError:
        parsed_dt = None
    if parsed_dt is None:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                parsed_dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    if parsed_dt is None:
        return None
    explicit_tz = parsed_dt.tzinfo is not None and parsed_dt.utcoffset() is not None
    normalized_dt = parsed_dt.astimezone(timezone.utc) if explicit_tz else parsed_dt
    return ParsedTemporal(
        normalized_dt,
        "instant",
        explicit_tz,
        normalized_dt.isoformat(),
        raw,
    )


def parse_cutoff(value: Any) -> ParsedTemporal:
    parsed = parse_temporal(value)
    if parsed is None:
        raise ValueError(f"prediction cutoff is missing or unparseable: {value!r}")
    return parsed




def _parse_date(raw: Any) -> Optional[datetime]:
    parsed = parse_temporal(raw)
    if parsed is None:
        return None
    if parsed.precision == "date":
        return datetime.combine(parsed.value, datetime.min.time())
    return parsed.value


def _parse_cutoff(cutoff_str: Any) -> Optional[datetime]:
    try:
        parsed = parse_cutoff(cutoff_str)
    except ValueError:
        return None
    if parsed.precision == "date":
        return datetime.combine(parsed.value, datetime.min.time())
    return parsed.value


def _compare_before(
    earlier: ParsedTemporal,
    later: ParsedTemporal,
) -> Tuple[Optional[bool], Optional[float], str]:
    if earlier.precision == "date" or later.precision == "date":
        earlier_date = (
            earlier.value if earlier.precision == "date" else earlier.value.date()
        )
        later_date = later.value if later.precision == "date" else later.value.date()
        delta = float((later_date - earlier_date).days)
        return earlier_date < later_date, delta, "calendar_date"
    if earlier.timezone_explicit != later.timezone_explicit:
        return None, None, "mixed_timezone_precision"
    delta_seconds = (later.value - earlier.value).total_seconds()
    return earlier.value < later.value, delta_seconds / 86400.0, "instant"


def _first_parsed(
    evidence: Mapping[str, Any],
    keys: Sequence[str],
) -> Tuple[Optional[ParsedTemporal], bool]:
    saw_nonempty = False
    for key in keys:
        if key not in evidence:
            continue
        raw = evidence.get(key)
        if raw is None or not str(raw).strip():
            continue
        saw_nonempty = True
        parsed = parse_temporal(raw)


        return parsed, True
    return None, saw_nonempty


def _availability_provenance_valid(evidence: Mapping[str, Any]) -> bool:
    kind = str(evidence.get("availability_bound_kind", "")).strip()
    source = str(evidence.get("availability_bound_source", "")).strip()
    source_hash = str(
        evidence.get("availability_bound_source_sha256", "")
    ).strip()
    return bool(kind and source and _SHA256_RE.fullmatch(source_hash))


def assess_evidence_admissibility(
    evidence: Mapping[str, Any],
    cutoff: Any,
    *,
    fallback_score: float = 0.25,
) -> AdmissibilityDecision:


    cutoff_parsed = parse_cutoff(cutoff)
    if not 0.0 <= float(fallback_score) <= 1.0:
        raise ValueError("fallback_score must be in [0, 1]")

    event_time, saw_event_value = _first_parsed(
        evidence,
        (
            "timestamp_iso",
            "event_timestamp_iso",
            "event_time_iso",
            "timestamp",
            "event_time",
            "date",
            "event_date",
            "timestamp_raw",
        ),
    )
    if event_time is not None:
        is_before, delta_days, comparison = _compare_before(
            event_time, cutoff_parsed
        )
        if is_before is None:
            return AdmissibilityDecision(
                INADMISSIBLE,
                f"event_time_{comparison}",
                cutoff_parsed.canonical,
                evidence_time=event_time.canonical,
            )
        if is_before:
            return AdmissibilityDecision(
                ADMISSIBLE_PRE_CUTOFF,
                f"known_event_time_pre_cutoff_{comparison}",
                cutoff_parsed.canonical,
                evidence_time=event_time.canonical,
                delta_days=max(1.0, float(delta_days or 0.0)),
            )
        return AdmissibilityDecision(
            INADMISSIBLE,
            f"known_event_time_equal_or_post_cutoff_{comparison}",
            cutoff_parsed.canonical,
            evidence_time=event_time.canonical,
        )

    availability, saw_availability_value = _first_parsed(
        evidence,
        (
            "availability_upper_bound_iso",
            "availability_upper_bound_date",
            "availability_upper_bound",
        ),
    )
    if availability is None:
        event_reason = "unparseable" if saw_event_value else "missing"
        bound_reason = (
            "unparseable_availability_bound"
            if saw_availability_value
            else "missing_availability_bound"
        )
        return AdmissibilityDecision(
            INADMISSIBLE,
            f"event_time_{event_reason}_{bound_reason}",
            cutoff_parsed.canonical,
        )
    if not _availability_provenance_valid(evidence):
        return AdmissibilityDecision(
            INADMISSIBLE,
            "availability_bound_missing_verifiable_provenance",
            cutoff_parsed.canonical,
            availability_upper_bound=availability.canonical,
        )

    is_before, _, comparison = _compare_before(availability, cutoff_parsed)
    if is_before is None:
        return AdmissibilityDecision(
            INADMISSIBLE,
            f"availability_bound_{comparison}",
            cutoff_parsed.canonical,
            availability_upper_bound=availability.canonical,
        )
    if not is_before:
        return AdmissibilityDecision(
            INADMISSIBLE,
            f"availability_bound_equal_or_post_cutoff_{comparison}",
            cutoff_parsed.canonical,
            availability_upper_bound=availability.canonical,
        )
    return AdmissibilityDecision(
        ADMISSIBLE_FALLBACK,
        f"missing_or_unparseable_event_time_safe_bound_{comparison}",
        cutoff_parsed.canonical,
        availability_upper_bound=availability.canonical,
        fallback_score=float(fallback_score),
    )


def is_admitted(decision: AdmissibilityDecision) -> bool:
    return decision.admitted


def _record_id(data: Mapping[str, Any], fallback: str, strict: bool) -> str:
    for key in ("record_id", "source_record_id"):
        value = str(data.get(key, "")).strip()
        if value:
            return value
    provenance = data.get("provenance")
    if isinstance(provenance, Mapping):
        value = str(provenance.get("record_id", "")).strip()
        if value:
            return value
    if strict:
        raise ValueError("admitted occurrence is missing record_id provenance")
    return fallback


def _proposition_key(data: Mapping[str, Any]) -> str:
    explicit = canonical_proposition_key(data.get("proposition_key"))
    if explicit:
        return explicit
    return canonical_proposition_key(
        data.get("name") or data.get("text") or data.get("sentence")
    )


class TemporalValidityScorer:


    def __init__(
        self,
        eta: float = 5.0,
        d_max: int = 4,
        *,
        tau: Optional[float] = None,
    ):
        if tau is not None:
            eta = tau
        if isinstance(d_max, bool) or not isinstance(d_max, int) or d_max <= 0:
            raise ValueError("d_max must be a positive integer")
        if not math.isfinite(float(eta)) or float(eta) <= 0.0:
            raise ValueError("eta must be finite and > 0")
        self.eta = float(eta)
        self.d_max = d_max
        self.tau = self.eta

    def _lambda(self, frequency: int) -> float:
        if isinstance(frequency, bool) or int(frequency) < 0:
            raise ValueError("N_q must be a non-negative integer")
        sqrt_f = math.sqrt(int(frequency))
        return 1.0 / (1.0 + math.exp(-sqrt_f / self.eta)) - 0.5

    def score(
        self,
        tc: datetime,
        tlast: datetime,
        frequency: int = 1,
        embedding_sim: float = 0.5,
        causal_decay: float = 1.0,
        *,
        span_days: Optional[float] = None,
    ) -> float:


        if span_days is None:
            span_days = max(1.0, (tc - tlast).total_seconds() / 86400.0)
        span_days = max(1.0, float(span_days))
        gamma = max(0.0, min(1.0, float(causal_decay)))
        similarity = _validated_similarity(embedding_sim)
        frequency = int(frequency)
        temporal_recency = math.exp(-0.5 * gamma * math.log(span_days))
        recurrence_salience = (
            self._lambda(frequency) * math.sqrt(frequency) * similarity
        )
        return min(temporal_recency + recurrence_salience, 1.0)

    def _gamma_from_distance(
        self,
        distance: Optional[float],
        max_depth: Optional[int] = None,
    ) -> float:
        depth = self.d_max if max_depth is None else max_depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        if distance is None or not math.isfinite(float(distance)):
            return 1.0
        return min(max(float(distance), 0.0) / float(depth), 1.0)

    def score_admitted_node(
        self,
        node_data: Mapping[str, Any],
        cutoff: Any,
        *,
        n_q: int,
        causal_decay: float = 1.0,
    ) -> Tuple[AdmissibilityDecision, Optional[float]]:
        decision = assess_evidence_admissibility(node_data, cutoff)
        if decision.status == INADMISSIBLE:
            return decision, None
        if decision.status == ADMISSIBLE_FALLBACK:
            return decision, float(decision.fallback_score)
        if "similarity" in node_data:
            similarity = node_data["similarity"]
        elif "s" in node_data:
            similarity = node_data["s"]
        else:


            similarity = 0.5
        score = self.score(
            datetime(2000, 1, 2),
            datetime(2000, 1, 1),
            n_q,
            similarity,
            causal_decay,
            span_days=decision.delta_days,
        )
        return decision, score

    def score_node(
        self,
        node_data: Dict[str, Any],
        tc: datetime,
        frequency_table: Dict[str, int],
        causal_decay: float = 1.0,
    ) -> float:
        key = _proposition_key(node_data)
        decision, value = self.score_admitted_node(
            node_data,
            tc,
            n_q=frequency_table.get(key, 1),
            causal_decay=causal_decay,
        )
        return 0.0 if decision.status == INADMISSIBLE else float(value)

    def score_nodes_with_decisions(
        self,
        nodes: Sequence[Tuple[str, Mapping[str, Any]]],
        cutoff_date: Any,
        causal_distances: Optional[Mapping[str, int]] = None,
        max_depth: Optional[int] = None,
        *,
        strict_support_ledger: bool = False,
    ) -> Tuple[
        List[Tuple[str, Mapping[str, Any], float]],
        Dict[str, AdmissibilityDecision],
        Dict[str, int],
    ]:
        parse_cutoff(cutoff_date)
        depth = self.d_max if max_depth is None else max_depth
        decisions: Dict[str, AdmissibilityDecision] = {}
        admitted: List[Tuple[str, Mapping[str, Any]]] = []
        support_sets: Dict[str, set] = {}
        for node_id, node_data in sorted(nodes, key=lambda item: str(item[0])):
            decision = assess_evidence_admissibility(node_data, cutoff_date)
            decisions[str(node_id)] = decision
            if not decision.admitted:
                continue
            proposition_key = _proposition_key(node_data)
            if not proposition_key and strict_support_ledger:
                raise ValueError(
                    f"admitted occurrence {node_id!r} is missing proposition_key"
                )
            proposition_key = proposition_key or canonical_proposition_key(node_id)
            record_id = _record_id(
                node_data, fallback=str(node_id), strict=strict_support_ledger
            )
            support_sets.setdefault(proposition_key, set()).add(record_id)
            admitted.append((str(node_id), node_data))

        n_q_by_proposition = {
            key: len(record_ids) for key, record_ids in sorted(support_sets.items())
        }
        scored: List[Tuple[str, Mapping[str, Any], float]] = []
        for node_id, node_data in admitted:
            proposition_key = _proposition_key(node_data) or canonical_proposition_key(
                node_id
            )
            gamma = self._gamma_from_distance(
                causal_distances.get(node_id) if causal_distances is not None else None,
                depth,
            )
            _, value = self.score_admitted_node(
                node_data,
                cutoff_date,
                n_q=n_q_by_proposition[proposition_key],
                causal_decay=gamma,
            )
            if value is None:
                raise RuntimeError("admissibility changed during deterministic scoring")
            scored.append((node_id, node_data, value))
        scored.sort(key=lambda item: (-item[2], str(item[0])))
        return scored, decisions, n_q_by_proposition

    def score_nodes(
        self,
        nodes: List[Tuple[str, Dict[str, Any]]],
        cutoff_date: Any,
        causal_distances: Optional[Dict[str, int]] = None,
        max_depth: Optional[int] = None,
        *,
        strict_support_ledger: bool = False,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        scored, _, _ = self.score_nodes_with_decisions(
            nodes,
            cutoff_date,
            causal_distances,
            max_depth,
            strict_support_ledger=strict_support_ledger,
        )
        return [(node_id, dict(data), score) for node_id, data, score in scored]

    def score_event_list(
        self,
        events: List[Dict[str, Any]],
        cutoff_date: Any,
        entity_gamma: float = 1.0,
        *,
        strict_support_ledger: bool = False,
    ) -> List[Tuple[Dict[str, Any], float]]:
        parse_cutoff(cutoff_date)
        admitted: List[Tuple[str, Dict[str, Any]]] = []
        support_sets: Dict[str, set] = {}
        for index, event in enumerate(events):
            node_id = str(index)
            if not assess_evidence_admissibility(event, cutoff_date).admitted:
                continue
            key = _proposition_key(event) or canonical_proposition_key(node_id)
            support_sets.setdefault(key, set()).add(
                _record_id(event, node_id, strict_support_ledger)
            )
            admitted.append((node_id, event))
        result: List[Tuple[Dict[str, Any], float]] = []
        for node_id, event in admitted:
            key = _proposition_key(event) or canonical_proposition_key(node_id)
            _, value = self.score_admitted_node(
                event,
                cutoff_date,
                n_q=len(support_sets[key]),
                causal_decay=entity_gamma,
            )
            if value is not None:
                result.append((dict(event), value))
        result.sort(
            key=lambda item: (
                -item[1],
                _proposition_key(item[0]),
                str(item[0].get("record_id", "")),
            )
        )
        return result

    def aggregate_entity_validity(
        self,
        events: List[Dict[str, Any]],
        cutoff_date: Any,
        entity_gamma: float = 1.0,
        *,
        strict_support_ledger: bool = False,
    ) -> float:
        scored = self.score_event_list(
            events,
            cutoff_date,
            entity_gamma=entity_gamma,
            strict_support_ledger=strict_support_ledger,
        )
        if not scored:
            return 0.5
        return sum(score for _, score in scored) / len(scored)


__all__ = [
    "ADMISSIBLE_FALLBACK",
    "ADMISSIBLE_PRE_CUTOFF",
    "INADMISSIBLE",
    "AdmissibilityDecision",
    "ParsedTemporal",
    "TemporalValidityScorer",
    "assess_evidence_admissibility",
    "canonical_proposition_key",
    "is_admitted",
    "parse_cutoff",
    "parse_temporal",
]
