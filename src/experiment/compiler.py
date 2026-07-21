"""Strict ID joins and linear-time grouped result compilation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .checkpoints import CheckpointValidationError, validate_record_ids
from .core import ExperimentRecord
from .planning import ActionUnit, BaseKey, ProposalKey


RecordLike = Union[ExperimentRecord, Mapping[str, Any]]
BaseUnitLike = Union[ActionUnit, Mapping[str, Any]]
_PROPOSAL_FIELDS = ("persona", "category", "proposal")
_BASE_FIELDS = ("persona", "category", "proposal", "action_type", "action")


def _coerce_record(value: RecordLike) -> ExperimentRecord:
    return (
        value
        if isinstance(value, ExperimentRecord)
        else ExperimentRecord.from_dict(value)
    )


def _coerce_base_unit(value: BaseUnitLike) -> ActionUnit:
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


def join_records_by_id(
    records: Iterable[RecordLike],
    *,
    expected_stage: Optional[str] = None,
    expected_ids: Optional[Sequence[str]] = None,
    require_complete: bool = True,
) -> Dict[str, ExperimentRecord]:
    """Build a strict sample-ID index, independent of response order."""

    indexed: Dict[str, ExperimentRecord] = {}
    for value in records:
        record = _coerce_record(value)
        if expected_stage is not None and record.stage != expected_stage:
            raise CheckpointValidationError(
                f"record {record.sample_id} has stage {record.stage!r}, "
                f"expected {expected_stage!r}"
            )
        if record.sample_id in indexed:
            raise CheckpointValidationError(f"duplicate sample ID: {record.sample_id}")
        indexed[record.sample_id] = record

    if expected_ids is not None:
        validate_record_ids(expected_ids, indexed, require_complete=require_complete)
        return {
            sample_id: indexed[sample_id]
            for sample_id in expected_ids
            if sample_id in indexed
        }
    return indexed


def _metadata_key(
    metadata: Mapping[str, Any], fields: Sequence[str]
) -> Tuple[str, ...]:
    values: List[str] = []
    for field in fields:
        if field not in metadata:
            raise CheckpointValidationError(
                f"record metadata is missing field: {field}"
            )
        value = metadata[field]
        if not isinstance(value, str) or not value:
            raise CheckpointValidationError(
                f"record metadata field {field!r} must be a non-empty string"
            )
        values.append(value)
    return tuple(values)


def _record_order(record: ExperimentRecord) -> Tuple[int, int, str]:
    metadata = record.metadata
    order_index = metadata.get("order_index", -1)
    replicate_id = metadata.get("replicate_id", 0)
    if isinstance(order_index, bool) or not isinstance(order_index, int):
        raise CheckpointValidationError("order_index must be an integer when present")
    if isinstance(replicate_id, bool) or not isinstance(replicate_id, int):
        raise CheckpointValidationError("replicate_id must be an integer when present")
    return (replicate_id, order_index, record.sample_id)


def _group_records(
    records_by_id: Mapping[str, ExperimentRecord],
    fields: Sequence[str],
) -> Dict[Tuple[str, ...], List[ExperimentRecord]]:
    grouped: Dict[Tuple[str, ...], List[ExperimentRecord]] = {}
    for record in records_by_id.values():
        key = _metadata_key(record.metadata, fields)
        grouped.setdefault(key, []).append(record)
    for values in grouped.values():
        values.sort(key=_record_order)
    return grouped


def _validate_group_domain(
    stage: str,
    grouped: Mapping[Tuple[str, ...], Sequence[ExperimentRecord]],
    allowed_keys: set,
) -> None:
    unknown = set(grouped) - allowed_keys
    if unknown:
        raise CheckpointValidationError(
            f"{stage} contains a result for an unselected experiment unit: "
            f"{sorted(unknown)[0]!r}"
        )


def _serialized(records: Sequence[ExperimentRecord]) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in records]


def compile_grouped_results(
    base_units: Iterable[BaseUnitLike],
    *,
    step1_records: Iterable[RecordLike],
    step2_records: Iterable[RecordLike],
    step3_records: Iterable[RecordLike],
    step4a_records: Iterable[RecordLike],
    step4b_records: Iterable[RecordLike],
    expected_ids_by_stage: Optional[Mapping[str, Sequence[str]]] = None,
    require_complete: bool = True,
) -> List[Dict[str, Any]]:
    """Compile all logical stages without repeatedly scanning global lookups.

    ``B`` is the number of selected action-level base units and ``D`` is the
    total number of result records. Step 4a is grouped by proposal unit and then
    shared, by sample ID, across matching action rows. Step 4b remains grouped
    by action unit. No stage lookup is scanned inside the base-unit loop. The
    implementation is output-sensitive: ``O(B log B + D log D + output_size)``
    because deterministic sorting and the denormalized final rows explicitly
    repeat each proposal-level Step 4a record for its action rows.
    """

    if require_complete and expected_ids_by_stage is None:
        raise ValueError("expected_ids_by_stage is required when require_complete=True")

    normalized_bases = sorted(_coerce_base_unit(value) for value in base_units)
    if not normalized_bases:
        raise ValueError("at least one base unit is required")
    base_keys: set = set()
    for unit in normalized_bases:
        if unit.key in base_keys:
            raise CheckpointValidationError(f"duplicate base unit: {unit.key!r}")
        base_keys.add(unit.key)
    proposal_keys = {unit.proposal_unit.key for unit in normalized_bases}

    inputs = {
        "step1": step1_records,
        "step2": step2_records,
        "step3": step3_records,
        "step4a": step4a_records,
        "step4b": step4b_records,
    }
    if expected_ids_by_stage is not None:
        if not isinstance(expected_ids_by_stage, Mapping):
            raise TypeError("expected_ids_by_stage must be a mapping")
        missing_stages = set(inputs).difference(expected_ids_by_stage)
        unknown_stages = set(expected_ids_by_stage).difference(inputs)
        if missing_stages:
            raise ValueError(
                f"expected_ids_by_stage is missing stage: {sorted(missing_stages)[0]}"
            )
        if unknown_stages:
            raise ValueError(
                "expected_ids_by_stage contains unknown stage: "
                f"{sorted(unknown_stages)[0]}"
            )
    indexes: Dict[str, Dict[str, ExperimentRecord]] = {}
    for stage, records in inputs.items():
        expected = expected_ids_by_stage.get(stage) if expected_ids_by_stage else None
        indexes[stage] = join_records_by_id(
            records,
            expected_stage=stage,
            expected_ids=expected,
            require_complete=require_complete,
        )

    step1_group = _group_records(indexes["step1"], _PROPOSAL_FIELDS)
    step2_group = _group_records(indexes["step2"], _PROPOSAL_FIELDS)
    step3_group = _group_records(indexes["step3"], _BASE_FIELDS)
    step4a_group = _group_records(indexes["step4a"], _PROPOSAL_FIELDS)
    step4b_group = _group_records(indexes["step4b"], _BASE_FIELDS)

    for stage, grouped, allowed in (
        ("step1", step1_group, proposal_keys),
        ("step2", step2_group, proposal_keys),
        ("step3", step3_group, base_keys),
        ("step4a", step4a_group, proposal_keys),
        ("step4b", step4b_group, base_keys),
    ):
        _validate_group_domain(stage, grouped, allowed)

    if require_complete:
        for proposal_key in proposal_keys:
            for stage, grouped in (
                ("step1", step1_group),
                ("step2", step2_group),
                ("step4a", step4a_group),
            ):
                if proposal_key not in grouped:
                    raise CheckpointValidationError(
                        f"{stage} has no records for selected proposal unit: {proposal_key!r}"
                    )
        for base_key in base_keys:
            for stage, grouped in (("step3", step3_group), ("step4b", step4b_group)):
                if base_key not in grouped:
                    raise CheckpointValidationError(
                        f"{stage} has no records for selected base unit: {base_key!r}"
                    )

    compiled: List[Dict[str, Any]] = []
    for unit in normalized_bases:
        proposal_key: ProposalKey = unit.proposal_unit.key
        base_key: BaseKey = unit.key
        compiled.append(
            {
                "base_sample_id": unit.unit_id,
                **unit.as_metadata(),
                "step1_first_order_belief": _serialized(
                    step1_group.get(proposal_key, ())
                ),
                "step2_second_order_belief": _serialized(
                    step2_group.get(proposal_key, ())
                ),
                "step3_action_support_no_distribution": _serialized(
                    step3_group.get(base_key, ())
                ),
                "step4a_first_order_with_treatment": _serialized(
                    step4a_group.get(proposal_key, ())
                ),
                "step4b_action_support_with_treatment": _serialized(
                    step4b_group.get(base_key, ())
                ),
            }
        )
    return compiled
