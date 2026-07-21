"""Deterministic experiment selection and treatment planning.

Planning is intentionally separated from model execution.  A caller first
builds and persists a :class:`SelectionPlan`, so ``max_base_units`` is applied
before *any* model call.  After valid Step 1 and Step 2 measurements are
available, :func:`build_treatment_plan` adds randomized treatment assignments,
leave-one-persona-out simulated consensus, and survey surprise.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .core import stable_sample_id


FIXED_DISTRIBUTION_PERCENTAGES: Tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0)

ProposalKey = Tuple[str, str, str]
BaseKey = Tuple[str, str, str, str, str]
BinaryBelief = Optional[Union[int, bool, Sequence[Optional[Union[int, bool]]]]]


def _require_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_percentage(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not 0.0 <= normalized <= 100.0:
        raise ValueError(f"{name} must be between 0 and 100")
    return normalized


@dataclass(frozen=True, order=True)
class ProposalUnit:
    persona: str
    category: str
    proposal: str

    def __post_init__(self) -> None:
        _require_non_empty_string("persona", self.persona)
        _require_non_empty_string("category", self.category)
        _require_non_empty_string("proposal", self.proposal)

    @property
    def key(self) -> ProposalKey:
        return (self.persona, self.category, self.proposal)

    def as_metadata(self) -> Dict[str, str]:
        return {
            "persona": self.persona,
            "category": self.category,
            "proposal": self.proposal,
        }

    @property
    def unit_id(self) -> str:
        return stable_sample_id("proposal_unit", self.as_metadata())


@dataclass(frozen=True, order=True)
class ActionUnit:
    persona: str
    category: str
    proposal: str
    action_type: str
    action: str

    def __post_init__(self) -> None:
        _require_non_empty_string("persona", self.persona)
        _require_non_empty_string("category", self.category)
        _require_non_empty_string("proposal", self.proposal)
        _require_non_empty_string("action_type", self.action_type)
        _require_non_empty_string("action", self.action)

    @property
    def key(self) -> BaseKey:
        return (
            self.persona,
            self.category,
            self.proposal,
            self.action_type,
            self.action,
        )

    @property
    def proposal_unit(self) -> ProposalUnit:
        return ProposalUnit(self.persona, self.category, self.proposal)

    def as_metadata(self) -> Dict[str, str]:
        return {
            "persona": self.persona,
            "category": self.category,
            "proposal": self.proposal,
            "action_type": self.action_type,
            "action": self.action,
        }

    @property
    def unit_id(self) -> str:
        return stable_sample_id("base_unit", self.as_metadata())


@dataclass(frozen=True)
class SelectionPlan:
    """The exact base and proposal units selected before model execution."""

    base_units: Tuple[ActionUnit, ...]
    proposal_units: Tuple[ProposalUnit, ...]
    max_base_units: Optional[int]
    selection_seed: int

    def __post_init__(self) -> None:
        if not self.base_units:
            raise ValueError("selection plan must contain at least one base unit")
        expected_proposals = tuple(
            sorted({unit.proposal_unit for unit in self.base_units})
        )
        if self.proposal_units != expected_proposals:
            raise ValueError(
                "proposal_units must be the deduplicated units from base_units"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_seed": self.selection_seed,
            "max_base_units": self.max_base_units,
            "base_units": [unit.as_metadata() for unit in self.base_units],
            "proposal_units": [unit.as_metadata() for unit in self.proposal_units],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionPlan":
        if not isinstance(payload, Mapping):
            raise TypeError("selection plan payload must be a mapping")
        try:
            base_payload = payload["base_units"]
            max_base_units = payload["max_base_units"]
            selection_seed = payload["selection_seed"]
        except KeyError as exc:
            raise ValueError(f"selection plan is missing field: {exc.args[0]}") from exc
        if not isinstance(base_payload, list):
            raise ValueError("selection plan base_units must be a list")
        base_units = tuple(sorted(_coerce_action_unit(value) for value in base_payload))
        proposal_units = tuple(sorted({unit.proposal_unit for unit in base_units}))
        return cls(
            base_units=base_units,
            proposal_units=proposal_units,
            max_base_units=max_base_units,
            selection_seed=selection_seed,
        )

    def expected_baseline_counts(self, replicates: int = 1) -> Dict[str, int]:
        _validate_positive_int("replicates", replicates)
        return {
            "step1": len(self.proposal_units) * replicates,
            "step2": len(self.proposal_units) * replicates,
            "step3": len(self.base_units) * replicates,
        }


def _coerce_action_unit(value: Union[ActionUnit, Mapping[str, Any]]) -> ActionUnit:
    if isinstance(value, ActionUnit):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("base units must be ActionUnit instances or mappings")
    try:
        return ActionUnit(
            persona=value["persona"],
            category=value["category"],
            proposal=value["proposal"],
            action_type=value["action_type"],
            action=value["action"],
        )
    except KeyError as exc:
        raise ValueError(f"base unit is missing field: {exc.args[0]}") from exc


def build_selection_plan(
    base_units: Iterable[Union[ActionUnit, Mapping[str, Any]]],
    *,
    max_base_units: Optional[int] = None,
    selection_seed: int = 42,
) -> SelectionPlan:
    """Build a stable sorted selection and apply the base-unit budget.

    The budget is deliberately applied before proposal-level Step 1/2 units are
    derived.  Thus a small budget cannot accidentally trigger full-data Step 1
    or Step 2 calls.
    """

    if max_base_units is not None:
        _validate_positive_int("max_base_units", max_base_units)
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("selection_seed must be an integer")

    normalized = sorted(_coerce_action_unit(value) for value in base_units)
    if not normalized:
        raise ValueError("at least one base unit is required")

    seen: set = set()
    duplicates: List[BaseKey] = []
    for unit in normalized:
        if unit.key in seen:
            duplicates.append(unit.key)
        seen.add(unit.key)
    if duplicates:
        raise ValueError(f"duplicate base units are not allowed: {duplicates[0]!r}")

    if max_base_units is not None and max_base_units < len(normalized):
        # Sort before sampling so input order never affects the seeded subset;
        # sort again afterwards so downstream sample order is deterministic.
        selected = sorted(
            random.Random(selection_seed).sample(normalized, max_base_units)
        )
    else:
        selected = normalized
    proposal_units = tuple(sorted({unit.proposal_unit for unit in selected}))
    return SelectionPlan(
        base_units=tuple(selected),
        proposal_units=proposal_units,
        max_base_units=max_base_units,
        selection_seed=selection_seed,
    )


def derive_seed(master_seed: int, *parts: Any) -> int:
    """Derive a stable positive 31-bit seed without Python's randomized hash."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    material = "\x1f".join([str(master_seed)] + [str(part) for part in parts])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class StageAssignment:
    stage: str
    unit_metadata: Mapping[str, Any]
    replicate_id: int
    seed: int
    sample_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "stage": self.stage,
            "metadata": {
                **dict(self.unit_metadata),
                "replicate_id": self.replicate_id,
                "seed": self.seed,
            },
        }


def build_baseline_assignments(
    selection: SelectionPlan,
    *,
    replicates: int = 1,
    master_seed: int = 42,
) -> Dict[str, Tuple[StageAssignment, ...]]:
    """Expand selected Step 1/2/3 units into reproducible replicates."""

    _validate_positive_int("replicates", replicates)
    result: Dict[str, List[StageAssignment]] = {"step1": [], "step2": [], "step3": []}

    for stage in ("step1", "step2"):
        for unit in selection.proposal_units:
            for replicate_id in range(replicates):
                seed = derive_seed(master_seed, stage, replicate_id, "generation")
                metadata: Dict[str, Any] = {
                    **unit.as_metadata(),
                    "replicate_id": replicate_id,
                    "seed": seed,
                }
                result[stage].append(
                    StageAssignment(
                        stage=stage,
                        unit_metadata=unit.as_metadata(),
                        replicate_id=replicate_id,
                        seed=seed,
                        sample_id=stable_sample_id(stage, metadata),
                    )
                )

    for unit in selection.base_units:
        for replicate_id in range(replicates):
            seed = derive_seed(master_seed, "step3", replicate_id, "generation")
            metadata = {
                **unit.as_metadata(),
                "replicate_id": replicate_id,
                "seed": seed,
            }
            result["step3"].append(
                StageAssignment(
                    stage="step3",
                    unit_metadata=unit.as_metadata(),
                    replicate_id=replicate_id,
                    seed=seed,
                    sample_id=stable_sample_id("step3", metadata),
                )
            )

    return {stage: tuple(assignments) for stage, assignments in result.items()}


@dataclass(frozen=True)
class TreatmentCondition:
    kind: str
    source: str
    percentage: Optional[float] = None
    excluded_persona: Optional[str] = None
    consensus_n: Optional[int] = None

    def __post_init__(self) -> None:
        _require_non_empty_string("kind", self.kind)
        _require_non_empty_string("source", self.source)
        if self.percentage is not None:
            _validate_percentage("percentage", self.percentage)
        if self.consensus_n is not None:
            _validate_positive_int("consensus_n", self.consensus_n)

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "treatment_kind": self.kind,
            "treatment_source": self.source,
            "distribution_percentage": self.percentage,
            "excluded_persona": self.excluded_persona,
            "consensus_n": self.consensus_n,
            "consensus_persona_n": self.consensus_n,
        }

    def identity_metadata(self) -> Dict[str, Any]:
        """Return design fields only, excluding realized baseline-derived values."""

        return {
            "treatment_kind": self.kind,
            "treatment_source": self.source,
            # Fixed percentages are assigned ex ante. Simulated consensus and
            # its n are realized from Step 1 and must not change sample IDs.
            "fixed_distribution_percentage": (
                self.percentage if self.kind == "fixed_hypothetical_survey" else None
            ),
            "excluded_persona": self.excluded_persona,
        }

    @property
    def condition_id(self) -> str:
        return stable_sample_id("treatment_condition", self.identity_metadata())


@dataclass(frozen=True)
class TreatmentAssignment:
    stage: str
    unit_metadata: Mapping[str, Any]
    condition: TreatmentCondition
    replicate_id: int
    seed: int
    order_index: int
    step2_predicted_percentage: Optional[float]
    survey_surprise: Optional[float]
    sample_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "stage": self.stage,
            "metadata": {
                **dict(self.unit_metadata),
                **self.condition.as_metadata(),
                "treatment_id": self.condition.condition_id,
                "replicate_id": self.replicate_id,
                "seed": self.seed,
                "order_index": self.order_index,
                "step2_predicted_percentage": self.step2_predicted_percentage,
                "survey_surprise": self.survey_surprise,
            },
        }


@dataclass(frozen=True)
class PlanningIssue:
    code: str
    stage: str
    unit_metadata: Mapping[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "unit_metadata": dict(self.unit_metadata),
            "message": self.message,
        }


@dataclass(frozen=True)
class TreatmentPlan:
    step4a: Tuple[TreatmentAssignment, ...]
    step4b: Tuple[TreatmentAssignment, ...]
    issues: Tuple[PlanningIssue, ...]
    fixed_percentages: Tuple[float, ...]
    replicates: int
    master_seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed_percentages": list(self.fixed_percentages),
            "replicates": self.replicates,
            "master_seed": self.master_seed,
            "step4a": [assignment.to_dict() for assignment in self.step4a],
            "step4b": [assignment.to_dict() for assignment in self.step4b],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def expected_counts(self) -> Dict[str, int]:
        return {"step4a": len(self.step4a), "step4b": len(self.step4b)}


def leave_one_persona_out_consensus(
    valid_binary_beliefs: Mapping[ProposalKey, BinaryBelief],
    target: ProposalUnit,
) -> Tuple[Optional[float], int]:
    """Compute simulated consensus excluding the target persona.

    Input values must be valid binary Step 1 decisions (0/1 or bool). ``None``
    represents an explicitly invalid/missing measurement and is excluded from
    the denominator.  The result is a percentage and the contributing persona
    count; no claim about the real U.S. population is made.
    """

    persona_means: List[float] = []
    for (persona, category, proposal), belief in valid_binary_beliefs.items():
        if (
            category != target.category
            or proposal != target.proposal
            or persona == target.persona
        ):
            continue
        observations: Sequence[Optional[Union[int, bool]]]
        if isinstance(belief, Sequence) and not isinstance(belief, (str, bytes)):
            observations = belief
        else:
            observations = (belief,)
        persona_values: List[int] = []
        for observation in observations:
            if observation is None:
                continue
            if isinstance(observation, bool):
                persona_values.append(int(observation))
            elif isinstance(observation, int) and observation in (0, 1):
                persona_values.append(observation)
            else:
                raise ValueError(
                    "consensus inputs must contain only binary 0/1, bool, or None"
                )
        if persona_values:
            # Each persona receives equal weight even when invalid replicates
            # leave different numbers of valid observations.
            persona_means.append(sum(persona_values) / len(persona_values))

    if not persona_means:
        return None, 0
    return 100.0 * sum(persona_means) / len(persona_means), len(persona_means)


def compute_survey_surprise(
    treatment_percentage: Optional[float],
    step2_predicted_percentage: Optional[float],
) -> Optional[float]:
    """Return treatment percentage minus the persona's Step 2 prediction."""

    if treatment_percentage is None or step2_predicted_percentage is None:
        return None
    treatment = _validate_percentage("treatment_percentage", treatment_percentage)
    predicted = _validate_percentage(
        "step2_predicted_percentage", step2_predicted_percentage
    )
    return treatment - predicted


def _normalize_fixed_percentages(values: Sequence[float]) -> Tuple[float, ...]:
    normalized = tuple(
        sorted(_validate_percentage("fixed percentage", value) for value in values)
    )
    if not normalized:
        raise ValueError("at least one fixed percentage is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("fixed percentages must be unique")
    return normalized


def _proposal_prediction(
    predictions: Mapping[ProposalKey, Optional[float]],
    unit: ProposalUnit,
) -> Optional[float]:
    value = predictions.get(unit.key)
    if value is None:
        return None
    return _validate_percentage("Step 2 predicted percentage", value)


def _conditions_for_unit(
    unit: ProposalUnit,
    *,
    fixed_percentages: Tuple[float, ...],
    valid_binary_beliefs: Optional[Mapping[ProposalKey, BinaryBelief]],
    include_simulated_consensus: bool,
    include_retest: bool,
    include_placebo: bool,
) -> Tuple[List[TreatmentCondition], Optional[PlanningIssue]]:
    conditions = [
        TreatmentCondition(
            kind="fixed_hypothetical_survey",
            source="hypothetical_survey",
            percentage=percentage,
        )
        for percentage in fixed_percentages
    ]

    issue: Optional[PlanningIssue] = None
    if include_simulated_consensus:
        consensus, consensus_n = (
            leave_one_persona_out_consensus(valid_binary_beliefs, unit)
            if valid_binary_beliefs is not None
            else (None, 0)
        )
        if consensus is None:
            issue = PlanningIssue(
                code="simulated_consensus_unavailable",
                stage="step4",
                unit_metadata=unit.as_metadata(),
                message="no valid other-persona Step 1 decisions are available",
            )
        conditions.append(
            TreatmentCondition(
                kind="simulated_persona_consensus",
                source="simulated_persona_consensus_leave_one_out",
                percentage=consensus,
                excluded_persona=unit.persona,
                consensus_n=consensus_n or None,
            )
        )

    if include_retest:
        conditions.append(
            TreatmentCondition(kind="no_information_retest", source="none")
        )
    if include_placebo:
        conditions.append(
            TreatmentCondition(kind="placebo_text", source="neutral_additional_text")
        )
    return conditions, issue


def _assign_treatments(
    *,
    stage: str,
    unit_metadata: Mapping[str, Any],
    unit_id: str,
    conditions: Sequence[TreatmentCondition],
    predicted_percentage: Optional[float],
    replicates: int,
    master_seed: int,
) -> List[TreatmentAssignment]:
    assignments: List[TreatmentAssignment] = []
    for replicate_id in range(replicates):
        shuffled = list(conditions)
        order_seed = derive_seed(
            master_seed, stage, unit_id, replicate_id, "treatment_order"
        )
        random.Random(order_seed).shuffle(shuffled)

        for order_index, condition in enumerate(shuffled):
            # A replicate seed is shared across experimental cells within a
            # stage. This supports efficient batching and common-random-number
            # comparisons while every sample ID remains unit-specific.
            seed = derive_seed(master_seed, stage, replicate_id, "generation")
            surprise = compute_survey_surprise(
                condition.percentage, predicted_percentage
            )
            identity_metadata = {
                **dict(unit_metadata),
                "treatment_id": condition.condition_id,
                "replicate_id": replicate_id,
                "seed": seed,
                "order_index": order_index,
            }
            assignments.append(
                TreatmentAssignment(
                    stage=stage,
                    unit_metadata=dict(unit_metadata),
                    condition=condition,
                    replicate_id=replicate_id,
                    seed=seed,
                    order_index=order_index,
                    step2_predicted_percentage=predicted_percentage,
                    survey_surprise=surprise,
                    sample_id=stable_sample_id(stage, identity_metadata),
                )
            )
    return assignments


def build_treatment_plan(
    selection: SelectionPlan,
    *,
    valid_binary_beliefs: Optional[Mapping[ProposalKey, BinaryBelief]] = None,
    step2_predictions: Mapping[ProposalKey, Optional[float]],
    fixed_percentages: Sequence[float] = FIXED_DISTRIBUTION_PERCENTAGES,
    include_simulated_consensus: bool = True,
    include_retest: bool = True,
    include_placebo: bool = True,
    replicates: int = 1,
    master_seed: int = 42,
) -> TreatmentPlan:
    """Build Step 4a/4b assignments after valid baseline measurements.

    Step 4a is expanded from deduplicated proposal units, never action units.
    Step 4b remains action-specific.  Treatment order is independently and
    reproducibly randomized for each unit and replicate.
    """

    _validate_positive_int("replicates", replicates)
    normalized_fixed = _normalize_fixed_percentages(fixed_percentages)
    if not isinstance(step2_predictions, Mapping):
        raise TypeError("step2_predictions must be a mapping")

    conditions_by_proposal: Dict[ProposalKey, List[TreatmentCondition]] = {}
    predictions_by_proposal: Dict[ProposalKey, Optional[float]] = {}
    issues: List[PlanningIssue] = []

    for unit in selection.proposal_units:
        conditions, issue = _conditions_for_unit(
            unit,
            fixed_percentages=normalized_fixed,
            valid_binary_beliefs=valid_binary_beliefs,
            include_simulated_consensus=include_simulated_consensus,
            include_retest=include_retest,
            include_placebo=include_placebo,
        )
        conditions_by_proposal[unit.key] = conditions
        prediction = _proposal_prediction(step2_predictions, unit)
        predictions_by_proposal[unit.key] = prediction
        if issue is not None:
            issues.append(issue)
        if prediction is None:
            issues.append(
                PlanningIssue(
                    code="step2_prediction_unavailable",
                    stage="step4",
                    unit_metadata=unit.as_metadata(),
                    message="survey surprise is unavailable without a valid Step 2 prediction",
                )
            )

    step4a: List[TreatmentAssignment] = []
    for unit in selection.proposal_units:
        step4a.extend(
            _assign_treatments(
                stage="step4a",
                unit_metadata=unit.as_metadata(),
                unit_id=unit.unit_id,
                conditions=conditions_by_proposal[unit.key],
                predicted_percentage=predictions_by_proposal[unit.key],
                replicates=replicates,
                master_seed=master_seed,
            )
        )

    step4b: List[TreatmentAssignment] = []
    for unit in selection.base_units:
        proposal_key = unit.proposal_unit.key
        step4b.extend(
            _assign_treatments(
                stage="step4b",
                unit_metadata=unit.as_metadata(),
                unit_id=unit.unit_id,
                conditions=conditions_by_proposal[proposal_key],
                predicted_percentage=predictions_by_proposal[proposal_key],
                replicates=replicates,
                master_seed=master_seed,
            )
        )

    return TreatmentPlan(
        step4a=tuple(step4a),
        step4b=tuple(step4b),
        issues=tuple(issues),
        fixed_percentages=normalized_fixed,
        replicates=replicates,
        master_seed=master_seed,
    )
