from __future__ import annotations

import copy
import hashlib
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

from chain.config import (
    COMPAT_PROFILE,
    ResolvedConfig,
    canonical_json,
    legacy_yes_no_axis,
    validate_binary_axis,
)
from chain.graph.hypergraph import MMDTHypergraph
from chain.skills.collector import collect_and_build
from chain.skills.reasoner import (
    _append_runtime_span,
    _runtime_spans,
    reason_aggregate,
    reason_step,
)


logger = logging.getLogger(__name__)

_OPERATIONAL_RUNTIME_TELEMETRY_SCHEMA = "chain-operational-runtime-telemetry-v1"
_REQUIRED_RUNTIME_STAGES = frozenset(
    {
        "active_graph_view",
        "reasoning_direction",
        "target_resolution",
        "evidence_retrieval_ctvf",
        "outcome_generation",
        "derivation_processing",
        "reasoning_round_total",
        "outcome_mapping",
        "causal_inference",
        "aggregation_total",
    }
)


class PredictionError(RuntimeError):
    pass


def _compat_case_id(question: str) -> str:
    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    return f"compat_{digest}"


def _native_yes_no_axis(axis: Mapping[str, Any]) -> bool:
    positive = axis["positive"]
    negative = axis["negative"]
    return (
        str(positive["id"]).casefold() == "yes"
        and str(positive["label"]).casefold() == "yes"
        and str(negative["id"]).casefold() == "no"
        and str(negative["label"]).casefold() == "no"
    )


def _validated_operational_runtime_telemetry(value: Any) -> Dict[str, Any]:


    if not isinstance(value, Mapping):
        raise PredictionError("reason_aggregate omitted operational_runtime_telemetry")
    if value.get("schema_version") != _OPERATIONAL_RUNTIME_TELEMETRY_SCHEMA:
        raise PredictionError("operational runtime telemetry schema mismatch")
    spans = value.get("spans")
    totals = value.get("stage_totals")
    if not isinstance(spans, list) or not spans:
        raise PredictionError("operational runtime telemetry spans must be non-empty")
    if not isinstance(totals, Mapping):
        raise PredictionError(
            "operational runtime telemetry stage_totals must be an object"
        )

    recomputed: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(spans):
        if not isinstance(raw, Mapping) or set(raw) != {
            "stage",
            "elapsed_seconds",
            "status",
            "round",
        }:
            raise PredictionError(f"operational runtime span {index} schema mismatch")
        stage = raw["stage"]
        elapsed = raw["elapsed_seconds"]
        status = raw["status"]
        round_num = raw["round"]
        if not isinstance(stage, str) or not stage.strip():
            raise PredictionError(
                f"operational runtime span {index} has an invalid stage"
            )
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise PredictionError(
                f"operational runtime span {index} has invalid elapsed_seconds"
            )
        if status not in {"ok", "error"}:
            raise PredictionError(
                f"operational runtime span {index} has an invalid status"
            )
        if (
            isinstance(round_num, bool)
            or not isinstance(round_num, int)
            or round_num <= 0
        ):
            raise PredictionError(
                f"operational runtime span {index} has an invalid round"
            )
        row = recomputed.setdefault(
            stage,
            {
                "count": 0,
                "elapsed_seconds": 0.0,
                "status_counts": {"ok": 0, "error": 0},
            },
        )
        row["count"] += 1
        row["elapsed_seconds"] += float(elapsed)
        row["status_counts"][status] += 1

    missing = sorted(_REQUIRED_RUNTIME_STAGES - set(recomputed))
    if missing:
        raise PredictionError(
            f"operational runtime telemetry is incomplete; missing {missing}"
        )
    if set(totals) != set(recomputed):
        raise PredictionError("operational runtime telemetry stage total keys mismatch")
    for stage, expected in recomputed.items():
        observed = totals.get(stage)
        if not isinstance(observed, Mapping) or set(observed) != {
            "count",
            "elapsed_seconds",
            "status_counts",
        }:
            raise PredictionError(
                f"operational runtime telemetry total for {stage!r} is invalid"
            )
        if observed.get("count") != expected["count"]:
            raise PredictionError(
                f"operational runtime telemetry count for {stage!r} does not close"
            )
        observed_elapsed = observed.get("elapsed_seconds")
        if (
            isinstance(observed_elapsed, bool)
            or not isinstance(observed_elapsed, (int, float))
            or not math.isclose(
                float(observed_elapsed),
                float(expected["elapsed_seconds"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise PredictionError(
                f"operational runtime telemetry elapsed total for {stage!r} does not close"
            )
        if observed.get("status_counts") != expected["status_counts"]:
            raise PredictionError(
                f"operational runtime telemetry status total for {stage!r} does not close"
            )
    return copy.deepcopy(dict(value))


def _legacy_operational_runtime_telemetry(
    *, rounds: int, latest_round: int
) -> Dict[str, Any]:


    round_count = max(1, int(rounds))
    final_round = max(1, int(latest_round))
    per_round = {
        "evidence_retrieval_ctvf",
        "outcome_generation",
        "derivation_processing",
        "reasoning_round_total",
    }
    first_round = {"active_graph_view", "reasoning_direction", "target_resolution"}
    stage_rounds: Dict[str, List[int]] = {
        stage: ([1] if stage in first_round else list(range(1, round_count + 1)))
        for stage in first_round | per_round
    }
    stage_rounds["aggregation_total"] = [final_round]
    stage_rounds["outcome_mapping"] = [final_round]
    stage_rounds["causal_inference"] = [final_round]
    spans: List[Dict[str, Any]] = []
    for stage in sorted(stage_rounds):
        for round_num in stage_rounds[stage]:
            spans.append(
                {
                    "stage": stage,
                    "elapsed_seconds": 0.0,
                    "status": "ok",
                    "round": int(round_num),
                }
            )
    stage_totals: Dict[str, Dict[str, Any]] = {}
    for span in spans:
        row = stage_totals.setdefault(
            span["stage"],
            {
                "count": 0,
                "elapsed_seconds": 0.0,
                "status_counts": {"ok": 0, "error": 0},
            },
        )
        row["count"] += 1
        row["elapsed_seconds"] += float(span["elapsed_seconds"])
        row["status_counts"][span["status"]] += 1
    return {
        "schema_version": _OPERATIONAL_RUNTIME_TELEMETRY_SCHEMA,
        "availability": "unavailable",
        "synthetic": True,
        "reason": "legacy_reason_aggregate_without_runtime_telemetry",
        "spans": spans,
        "stage_totals": stage_totals,
    }


class CHAINAgent:
    def __init__(
        self,
        hypergraph_dir: Optional[str] = None,
        *,
        config: Optional[ResolvedConfig] = None,
        hypergraph: Any = None,
        graph_validation: Optional[Mapping[str, Any]] = None,
        test_only_allow_unvalidated_graph: bool = False,
    ) -> None:
        if not isinstance(config, ResolvedConfig):
            raise TypeError("config must be a ResolvedConfig")
        self.config = config
        if test_only_allow_unvalidated_graph and self.config.is_paper:
            raise PredictionError(
                "paper profile forbids the unvalidated test-only graph seam"
            )
        self.graph_validation = dict(graph_validation or {})
        if hypergraph is not None:
            self.hypergraph = hypergraph
            if not self.graph_validation and not test_only_allow_unvalidated_graph:
                raise PredictionError(
                    "injected hypergraph requires a validated graph receipt"
                )
            if not self.graph_validation and test_only_allow_unvalidated_graph:
                self.graph_validation = {
                    "graph_bundle_sha256": hashlib.sha256(
                        b"chain-test-only-unvalidated-graph"
                    ).hexdigest(),
                    "profile_id": self.config.profile_id,
                    "publishable": False,
                    "test_only": True,
                }
        else:
            graph_dir = hypergraph_dir or self.config.hypergraph_dir
            self.hypergraph = MMDTHypergraph(
                graph_dir,
                validate=True,









                expected_profile_id=self.config.profile_id,
                expected_config=self.config,
                require_publishable=False,
            )
            self.graph_validation = dict(self.hypergraph.validation)
        self._validate_graph_receipt(test_only_allow_unvalidated_graph)
        self._cancel = threading.Event()

    def _validate_graph_receipt(self, test_only: bool) -> None:
        if not self.graph_validation:
            if test_only:
                return
            raise PredictionError("missing validated graph receipt")
        bundle_hash = str(self.graph_validation.get("graph_bundle_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_hash):
            raise PredictionError("graph receipt has no canonical graph_bundle_sha256")
        manifest = self.graph_validation.get("manifest", self.graph_validation)
        if not test_only:
            telemetry = self.graph_validation.get("telemetry")
            if not isinstance(manifest, Mapping) or not isinstance(telemetry, Mapping):
                raise PredictionError(
                    "graph receipt lacks manifest/telemetry validation closure"
                )
            if manifest.get("schema_version") != "chain-graph-bundle-v1":
                raise PredictionError(
                    "graph receipt has an unsupported manifest schema"
                )
            if telemetry.get("schema_version") != "chain-build-telemetry-v1":
                raise PredictionError(
                    "graph receipt has an unsupported telemetry schema"
                )
            if telemetry.get("status") != "success" or telemetry.get("errors") != []:
                raise PredictionError(
                    "graph receipt is not a successful error-free validation"
                )
            if manifest.get("graph_bundle_sha256") != bundle_hash:
                raise PredictionError("graph manifest and receipt bundle hashes differ")
            if telemetry.get("graph_bundle_sha256") != bundle_hash:
                raise PredictionError(
                    "graph telemetry and receipt bundle hashes differ"
                )
        object_hash = getattr(self.hypergraph, "graph_bundle_sha256", None)
        if object_hash is not None and str(object_hash) != bundle_hash:
            raise PredictionError(
                "graph object and validation receipt bundle hashes differ"
            )

    def cancel(self) -> None:
        self._cancel.set()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise InterruptedError("prediction cancelled")

    def predict(
        self,
        question: str,
        time_cutoff: str = "",
        skip_collect: bool = False,
        on_event: Any = None,
        local_file: str = "",
        online_source_descriptor: str = "",
        *,
        case_id: str = "",
        binary_axis: Optional[Mapping[str, Any]] = None,
        inference_config: Optional[ResolvedConfig] = None,
    ) -> Dict[str, Any]:
        resolved = inference_config or self.config
        if not isinstance(resolved, ResolvedConfig):
            raise TypeError("inference_config must be a ResolvedConfig")
        if (
            resolved.profile_id != self.config.profile_id
            or resolved.scientific_config_sha256 != self.config.scientific_config_sha256
        ):
            raise PredictionError("agent and case inference configuration mismatch")
        if not isinstance(question, str) or not question.strip():
            raise PredictionError("question must be non-empty")
        cutoff = str(time_cutoff or "").strip()
        if not cutoff:
            raise PredictionError("each case requires an explicit prediction cutoff")
        if self.config.is_paper and not skip_collect:
            raise PredictionError(
                "paper prediction requires prebuilt graph mode (skip_collect=True)"
            )
        if not skip_collect and not str(online_source_descriptor or "").strip():
            raise PredictionError(
                "online collection requires an explicit source descriptor and is compat-only"
            )
        canonical_case_id = str(case_id or "").strip()
        if not canonical_case_id:
            if resolved.profile != COMPAT_PROFILE:
                raise PredictionError("paper profile requires an explicit case_id")
            canonical_case_id = _compat_case_id(question)
        if binary_axis is None:
            if resolved.profile != COMPAT_PROFILE:
                raise PredictionError("paper profile requires an explicit binary_axis")
            axis = legacy_yes_no_axis()
        else:
            axis = validate_binary_axis(binary_axis)
        if resolved.is_paper and axis.get("source") == "legacy_adapter":
            raise PredictionError(
                "paper profile rejects the legacy binary-axis adapter"
            )
        if local_file:
            raise PredictionError(
                "local_file ingest is disabled in predict; build a validated replacement bundle first"
            )

        self._cancel.clear()

        def emit(event_type: str, data: Any) -> None:
            if on_event is not None:
                on_event(event_type, data)

        all_collect: list[dict[str, Any]] = []

        rg = None
        runtime_history: List[Dict[str, Any]] = []
        converged = False
        total_reason_round = 0
        phase_need_more = False
        reasoning_rounds: list[dict[str, Any]] = []

        for outer_index in range(resolved.max_outer_loops):
            iteration = outer_index + 1
            info_queries: List[str] = []
            if rg is not None and getattr(rg, "round_log", None):
                for entry in reversed(rg.round_log):
                    if entry.get("need_more_info") and entry.get("info_queries"):
                        info_queries = list(entry["info_queries"])
                        break
            self._check_cancel()
            emit(
                "phase_start",
                {"phase": 1, "iteration": iteration, "info_queries": info_queries},
            )

            if not skip_collect:
                for collect_index in range(resolved.collect_rounds_per_phase):
                    self._check_cancel()
                    extra_queries = (
                        info_queries if outer_index > 0 and collect_index == 0 else []
                    )
                    result = collect_and_build(
                        question,
                        self.hypergraph,
                        cutoff,
                        extra_queries=extra_queries,
                        config=resolved,
                        source_descriptor=online_source_descriptor,
                    )
                    updated_receipt = getattr(self.hypergraph, "validation", None)
                    if callable(updated_receipt):
                        updated_receipt = updated_receipt()
                    if not isinstance(updated_receipt, Mapping):
                        raise PredictionError(
                            "staged online collection did not expose its validation receipt"
                        )
                    self.graph_validation = dict(updated_receipt)
                    self._validate_graph_receipt(False)
                    if result.get("graph_bundle_sha256") != self.graph_validation.get(
                        "graph_bundle_sha256"
                    ):
                        raise PredictionError(
                            "collector result and active graph receipt bundle hashes differ"
                        )



                    if rg is not None:
                        runtime_history.extend(copy.deepcopy(_runtime_spans(rg)))
                    rg = None
                    all_collect.append(result)
                    emit(
                        "collect_done",
                        {
                            "round": len(all_collect),
                            "iteration": iteration,
                            "collect_i": collect_index + 1,
                            **result,
                        },
                    )
                    if (
                        collect_index >= 1
                        and result["search_results"] < resolved.minimum_new_results
                    ):
                        break

            self._check_cancel()
            emit("phase_start", {"phase": 2, "iteration": iteration})
            phase_need_more = False
            for reason_index in range(resolved.reason_rounds_per_phase):
                total_reason_round += 1
                self._check_cancel()
                round_started = time.perf_counter()
                prior_runtime_span_count = (
                    len(_runtime_spans(rg)) if rg is not None else 0
                )
                rg, step_result = reason_step(
                    question,
                    self.hypergraph,
                    rg,
                    total_reason_round,
                    cutoff_date=cutoff,
                    case_id=canonical_case_id,
                    binary_axis=axis,
                    inference_config=resolved,
                )
                runtime_spans = _runtime_spans(rg)
                if runtime_history:
                    runtime_spans[0:0] = runtime_history
                    runtime_history = []
                if not any(
                    span.get("stage") == "reasoning_round_total"
                    and span.get("round") == total_reason_round
                    for span in runtime_spans[prior_runtime_span_count:]
                    if isinstance(span, Mapping)
                ):
                    _append_runtime_span(
                        runtime_spans,
                        stage="reasoning_round_total",
                        round_num=total_reason_round,
                        started=round_started,
                        status="ok",
                    )
                if not isinstance(step_result, dict):
                    raise PredictionError("reason_step returned a non-object result")
                if not isinstance(step_result.get("scored_context"), Mapping):
                    raise PredictionError(
                        "reason_step omitted the complete scored_context audit payload"
                    )
                confidence = float(step_result.get("confidence", 0.0))
                if not math.isfinite(confidence):
                    raise PredictionError("reason_step returned non-finite confidence")
                phase_need_more = bool(step_result.get("need_more_info", False))
                round_trace = {
                    "schema_version": "chain-reasoning-round-trace-v1",
                    "round": total_reason_round,
                    "iteration": iteration,
                    "reason_i": reason_index + 1,
                    "step_result": copy.deepcopy(step_result),
                }
                try:
                    canonical_json(round_trace)
                except (TypeError, ValueError) as exc:
                    raise PredictionError(
                        "reason_step returned a non-serializable round trace"
                    ) from exc
                reasoning_rounds.append(round_trace)
                emit(
                    "reason_done",
                    {
                        "round": total_reason_round,
                        "iteration": iteration,
                        "reason_i": reason_index + 1,
                        "confidence": confidence,
                        "need_more": phase_need_more,
                    },
                )
                round_log = getattr(rg, "round_log", [])
                if len(round_log) >= resolved.stability_window:
                    recent = [
                        float(item.get("confidence", 0.0))
                        for item in round_log[-resolved.stability_window :]
                    ]
                    mean_confidence = sum(recent) / len(recent)
                    variance = sum(
                        (value - mean_confidence) ** 2 for value in recent
                    ) / len(recent)
                    if (
                        math.sqrt(variance) < resolved.stability_stdev
                        and mean_confidence >= resolved.stability_mean
                    ):
                        converged = True
                        break
                if confidence >= resolved.confidence_threshold and not phase_need_more:
                    converged = True
                    break
            if converged or (not phase_need_more and outer_index >= 1):
                break

        if rg is None:
            raise PredictionError("reasoning produced no reasoning graph")
        aggregation_started = time.perf_counter()
        prior_aggregation_span_count = len(_runtime_spans(rg))
        prediction = reason_aggregate(
            rg,
            question,
            cutoff_date=cutoff,
            case_id=canonical_case_id,
            binary_axis=axis,
            inference_config=resolved,
        )
        if not isinstance(prediction, dict):
            raise PredictionError("reason_aggregate returned a non-object result")
        runtime_spans = _runtime_spans(rg)
        if not any(
            span.get("stage") == "aggregation_total"
            and span.get("round")
            == int(prediction.get("latest_round", total_reason_round))
            for span in runtime_spans[prior_aggregation_span_count:]
            if isinstance(span, Mapping)
        ):
            _append_runtime_span(
                runtime_spans,
                stage="aggregation_total",
                round_num=int(prediction.get("latest_round", total_reason_round)),
                started=aggregation_started,
                status="ok",
            )
        if prediction.get("rounds") != len(reasoning_rounds):
            raise PredictionError(
                "reason_aggregate round count does not match the complete round trace"
            )
        if (
            not reasoning_rounds
            or prediction.get("latest_round") != reasoning_rounds[-1]["round"]
        ):
            raise PredictionError(
                "reason_aggregate latest_round does not match the complete round trace"
            )
        runtime_candidate = prediction.get("operational_runtime_telemetry")
        if runtime_candidate is None:
            runtime_payload = _legacy_operational_runtime_telemetry(
                rounds=int(prediction.get("rounds", len(reasoning_rounds))),
                latest_round=int(prediction.get("latest_round", total_reason_round)),
            )
        else:
            runtime_payload = _validated_operational_runtime_telemetry(
                runtime_candidate
            )
        try:
            p_event = float(prediction["p_event"])
            p_final = float(prediction["p_final_event"])
            p_llm = float(prediction["p_llm_event"])
            p_causal = float(prediction["p_causal_event"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionError(
                "reason_aggregate omitted canonical event probabilities"
            ) from exc
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (p_event, p_final, p_llm, p_causal)
        ):
            raise PredictionError(
                "canonical event probabilities must be finite and in [0,1]"
            )
        if p_event != p_final:
            raise PredictionError("p_event must exactly equal p_final_event")

        event_side = axis["positive"] if p_event >= 0.5 else axis["negative"]
        prediction.update(
            {
                "case_id": canonical_case_id,
                "cutoff": cutoff,
                "binary_axis": axis,
                "binary_axis_sha256": axis["binary_axis_sha256"],
                "p_llm_event": p_llm,
                "p_causal_event": p_causal,
                "p_final_event": p_final,
                "p_event": p_event,
                "answer": str(event_side["label"]).strip(),
                "confidence": p_event if p_event >= 0.5 else 1.0 - p_event,
                "profile_id": resolved.profile_id,
                "scientific_config_sha256": resolved.scientific_config_sha256,
                "graph_bundle_sha256": self.graph_validation["graph_bundle_sha256"],
                "collect_rounds": len(all_collect),
                "collect_stats": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"data_dir", "source_descriptor"}
                    }
                    for item in all_collect
                ],
                "converged": converged,
                "reasoning_rounds": reasoning_rounds,
                "operational_runtime_telemetry": runtime_payload,
            }
        )
        if _native_yes_no_axis(axis):
            prediction["p_yes"] = p_event
        emit(
            "aggregate",
            {
                "answer": prediction["answer"],
                "confidence": prediction["confidence"],
                "p_event": p_event,
            },
        )

        try:
            canonical_json(prediction)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"prediction is not canonically serializable: {exc}"
            ) from exc
        return prediction

    def test_batch(
        self,
        questions: List[Dict[str, Any]],
        time_cutoff: str = "",
        on_event: Any = None,
        results_dir: Optional[str] = None,
        start_index: int = 0,
    ) -> Dict[str, Any]:


        if results_dir:
            logger.warning(
                "results_dir is ignored; evaluate.py owns result publication"
            )
        items: list[dict[str, Any]] = []
        for index, item in enumerate(questions):
            if index < start_index:
                continue
            cutoff = str(item.get("cutoff") or time_cutoff or "").strip()
            prediction = self.predict(
                item["question"],
                time_cutoff=cutoff,
                skip_collect=True,
                on_event=on_event,
                case_id=str(item.get("id", f"item_{index:04d}")),
                binary_axis=item.get("binary_axis"),
            )
            items.append(
                {
                    "index": index,
                    "item_id": prediction["case_id"],
                    "question": item["question"],
                    "answer": prediction["answer"],
                    "confidence": prediction["confidence"],
                    "p_event": prediction["p_event"],
                    "prediction": prediction,
                }
            )
        return {"total": len(questions), "results_dir": None, "items": items}
