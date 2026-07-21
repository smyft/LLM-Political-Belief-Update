"""Versioned, fingerprinted, atomic experiment checkpoints.

The store writes one immutable JSON shard per completed stage chunk. Records are
identified by stable sample IDs, making retries idempotent and allowing strict
duplicate, unknown, and missing-record checks before resumption or compilation.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .core import (
    ExperimentRecord,
    ResultStatus,
    canonical_json,
    hash_mapping,
    sha256_text,
)


SCHEMA_VERSION = 2
LOGICAL_STAGES: Tuple[str, ...] = ("step1", "step2", "step3", "step4a", "step4b")
STAGE_DEPENDENCIES: Mapping[str, Tuple[str, ...]] = {
    "step1": (),
    "step2": ("step1",),
    "step3": ("step1", "step2"),
    "step4a": ("step1", "step2", "step3"),
    "step4b": ("step1", "step2", "step3", "step4a"),
}
_SAFE_RUN_ID_RE = re.compile(r"\A[A-Za-z0-9_.-]+\Z")
_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.Lock] = {}
_INDEX_SCHEMA_VERSION = "3"
_BINARY_STAGES = frozenset({"step1", "step3", "step4a", "step4b"})
_SUPPORTED_PIPELINES = frozenset({"verbalize", "logprob"})
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_LOGPROB_SCORE_FIELDS = frozenset(
    {
        "probabilities",
        "label_logprobs",
        "candidate_mass",
        "residual_mass",
        "candidates",
        "sampled_token_id",
        "sampled_choice",
        "format_valid",
        "analysis_text",
        "analysis_text_kind",
        "estimator",
        "conditional_on_candidate_set",
        "scoring_temperature",
        "finish_reason",
    }
)


_FileSignature = Tuple[int, int, int, int, int, int]


@dataclass
class _StageAuthorityCache:
    """Store-local view derived exclusively from validated JSON shards.

    The SQLite database is deliberately absent from the authoritative fields.
    File signatures only decide whether this cached view may be reused; a cold
    store or any externally visible filesystem change falls back to reloading
    and validating every JSON shard.
    """

    chunk_signatures: Dict[str, _FileSignature]
    chunk_sample_ids: Dict[str, Tuple[str, ...]]
    sample_chunks: Dict[str, str]
    index_signature: _FileSignature


class CheckpointValidationError(ValueError):
    """Raised when a manifest or checkpoint violates reproducibility rules."""


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointValidationError(f"{name} must be numeric and not boolean")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise CheckpointValidationError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise CheckpointValidationError(f"{name} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise CheckpointValidationError(f"{name} must be at most {maximum:g}")
    return number


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _validate_percentage_value(value: Any, *, field: str) -> None:
    _finite_number(value, field, minimum=0.0, maximum=100.0)


def validate_logprob_value(value: Any, *, sample_id: str) -> None:
    """Validate the complete persisted value contract for a VALID logprob record."""

    prefix = f"record {sample_id} logprob value"
    if not isinstance(value, Mapping):
        raise CheckpointValidationError(f"{prefix} must be an object")
    missing = _LOGPROB_SCORE_FIELDS.difference(value)
    if missing:
        raise CheckpointValidationError(
            f"{prefix} is missing field: {sorted(missing)[0]}"
        )

    probabilities = value["probabilities"]
    if not isinstance(probabilities, Mapping) or set(probabilities) != {"Yes", "No"}:
        raise CheckpointValidationError(
            f"{prefix} probabilities must contain exactly Yes and No"
        )
    yes_probability = _finite_number(
        probabilities["Yes"], f"{prefix} Yes probability", minimum=0.0, maximum=1.0
    )
    no_probability = _finite_number(
        probabilities["No"], f"{prefix} No probability", minimum=0.0, maximum=1.0
    )
    if not math.isclose(
        yes_probability + no_probability, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise CheckpointValidationError(f"{prefix} probabilities must sum to one")

    label_logprobs = value["label_logprobs"]
    if not isinstance(label_logprobs, Mapping) or set(label_logprobs) != {
        "Yes",
        "No",
    }:
        raise CheckpointValidationError(
            f"{prefix} label_logprobs must contain exactly Yes and No"
        )
    yes_logprob = _finite_number(label_logprobs["Yes"], f"{prefix} Yes logprob")
    no_logprob = _finite_number(label_logprobs["No"], f"{prefix} No logprob")

    candidate_mass = _finite_number(
        value["candidate_mass"], f"{prefix} candidate_mass", minimum=0.0, maximum=1.0
    )
    residual_mass = _finite_number(
        value["residual_mass"], f"{prefix} residual_mass", minimum=0.0, maximum=1.0
    )
    if not math.isclose(candidate_mass + residual_mass, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise CheckpointValidationError(
            f"{prefix} candidate_mass and residual_mass must sum to one"
        )

    candidates = value["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 128:
        raise CheckpointValidationError(
            f"{prefix} candidates must be a list containing 2 to 128 entries"
        )
    candidate_by_id: Dict[int, str] = {}
    candidate_logprobs: Dict[str, List[float]] = {"Yes": [], "No": []}
    candidate_probabilities: List[float] = []
    for index, candidate in enumerate(candidates):
        candidate_prefix = f"{prefix} candidate {index}"
        if not isinstance(candidate, Mapping):
            raise CheckpointValidationError(f"{candidate_prefix} must be an object")
        required = {"token_id", "choice", "decoded_token", "logprob", "probability"}
        missing_candidate = required.difference(candidate)
        if missing_candidate:
            raise CheckpointValidationError(
                f"{candidate_prefix} is missing field: {sorted(missing_candidate)[0]}"
            )
        token_id = candidate["token_id"]
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise CheckpointValidationError(
                f"{candidate_prefix} token_id must be a non-negative integer"
            )
        if token_id in candidate_by_id:
            raise CheckpointValidationError(
                f"{prefix} candidate token IDs must be unique"
            )
        choice = candidate["choice"]
        if choice not in {"Yes", "No"}:
            raise CheckpointValidationError(
                f"{candidate_prefix} choice must be Yes or No"
            )
        decoded_token = candidate["decoded_token"]
        if decoded_token is not None and not isinstance(decoded_token, str):
            raise CheckpointValidationError(
                f"{candidate_prefix} decoded_token must be a string or null"
            )
        logprob = _finite_number(candidate["logprob"], f"{candidate_prefix} logprob")
        if logprob > 1e-9:
            raise CheckpointValidationError(
                f"{candidate_prefix} logprob cannot be positive"
            )
        probability = _finite_number(
            candidate["probability"],
            f"{candidate_prefix} probability",
            minimum=0.0,
            maximum=1.0,
        )
        if not math.isclose(
            probability, math.exp(logprob), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise CheckpointValidationError(
                f"{candidate_prefix} probability does not match logprob"
            )
        candidate_by_id[token_id] = choice
        candidate_logprobs[choice].append(logprob)
        candidate_probabilities.append(probability)

    if any(not values for values in candidate_logprobs.values()):
        raise CheckpointValidationError(f"{prefix} candidates must cover Yes and No")
    for choice, stored in (("Yes", yes_logprob), ("No", no_logprob)):
        calculated = _logsumexp(candidate_logprobs[choice])
        if not math.isclose(calculated, stored, rel_tol=1e-9, abs_tol=1e-9):
            raise CheckpointValidationError(
                f"{prefix} {choice} label_logprob does not match candidates"
            )
    if not math.isclose(
        math.fsum(candidate_probabilities),
        candidate_mass,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise CheckpointValidationError(
            f"{prefix} candidate_mass does not match candidates"
        )

    total_logprob = _logsumexp((yes_logprob, no_logprob))
    expected_yes = math.exp(yes_logprob - total_logprob)
    if not math.isclose(
        yes_probability, expected_yes, rel_tol=1e-9, abs_tol=1e-12
    ) or not math.isclose(
        no_probability, 1.0 - expected_yes, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise CheckpointValidationError(
            f"{prefix} conditional probabilities do not match label_logprobs"
        )

    sampled_token_id = value["sampled_token_id"]
    if (
        isinstance(sampled_token_id, bool)
        or not isinstance(sampled_token_id, int)
        or sampled_token_id not in candidate_by_id
    ):
        raise CheckpointValidationError(
            f"{prefix} sampled_token_id must identify a candidate token"
        )
    sampled_choice = value["sampled_choice"]
    if sampled_choice not in {"Yes", "No"}:
        raise CheckpointValidationError(f"{prefix} sampled_choice must be Yes or No")
    if candidate_by_id[sampled_token_id] != sampled_choice:
        raise CheckpointValidationError(
            f"{prefix} sampled_choice does not match sampled_token_id"
        )
    if value["format_valid"] is not True:
        raise CheckpointValidationError(
            f"{prefix} format_valid must be true for a VALID record"
        )

    analysis_text = value["analysis_text"]
    if not isinstance(analysis_text, str) or not analysis_text.strip():
        raise CheckpointValidationError(f"{prefix} analysis_text must be non-empty")
    if value["analysis_text_kind"] != "model_generated_visible_text":
        raise CheckpointValidationError(f"{prefix} has an invalid analysis_text_kind")
    if value["estimator"] != "bounded_single_token_candidate_set":
        raise CheckpointValidationError(f"{prefix} has an invalid estimator")
    if value["conditional_on_candidate_set"] is not True:
        raise CheckpointValidationError(
            f"{prefix} conditional_on_candidate_set must be true"
        )
    scoring_temperature = _finite_number(
        value["scoring_temperature"], f"{prefix} scoring_temperature"
    )
    if scoring_temperature != 0.0:
        raise CheckpointValidationError(f"{prefix} scoring_temperature must be zero")
    finish_reason = value["finish_reason"]
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise CheckpointValidationError(
            f"{prefix} finish_reason must be a string or null"
        )


def _validate_checkpoint_record(
    manifest: "RunManifest", record: ExperimentRecord
) -> None:
    if manifest.pipeline not in _SUPPORTED_PIPELINES:
        raise CheckpointValidationError(
            f"unsupported checkpoint pipeline: {manifest.pipeline!r}"
        )
    if not isinstance(record.status, ResultStatus):
        raise CheckpointValidationError(
            f"record {record.sample_id} status must be a ResultStatus"
        )
    if record.raw_response is not None and not isinstance(record.raw_response, str):
        raise CheckpointValidationError(
            f"record {record.sample_id} raw_response must be a string or null"
        )
    if record.status is ResultStatus.VALID:
        if record.error_code is not None or record.error_message is not None:
            raise CheckpointValidationError(
                f"VALID record {record.sample_id} cannot contain error fields"
            )
        if record.stage == "step2":
            _validate_percentage_value(
                record.value, field=f"record {record.sample_id} Step 2 value"
            )
        elif record.stage in _BINARY_STAGES:
            if manifest.pipeline == "verbalize":
                if not isinstance(record.value, str) or record.value not in {
                    "Yes",
                    "No",
                }:
                    raise CheckpointValidationError(
                        f"VALID verbalize record {record.sample_id} must be Yes or No"
                    )
            else:
                validate_logprob_value(record.value, sample_id=record.sample_id)
        return

    if not isinstance(record.error_code, str) or not record.error_code:
        raise CheckpointValidationError(
            f"{record.status.value.upper()} record {record.sample_id} requires error_code"
        )
    if not isinstance(record.error_message, str):
        raise CheckpointValidationError(
            f"{record.status.value.upper()} record {record.sample_id} requires error_message"
        )
    if (
        record.status is ResultStatus.INVALID
        and record.stage == "step2"
        and record.value is not None
    ):
        raise CheckpointValidationError(
            f"INVALID Step 2 record {record.sample_id} must have a null value"
        )
    if manifest.pipeline == "verbalize" and record.status is ResultStatus.INVALID:
        if record.value is not None:
            raise CheckpointValidationError(
                f"INVALID verbalize record {record.sample_id} must have a null value"
            )
    if (
        manifest.pipeline == "logprob"
        and record.status is ResultStatus.INVALID
        and record.stage in _BINARY_STAGES
        and record.value is not None
    ):
        if not isinstance(record.value, Mapping):
            raise CheckpointValidationError(
                f"INVALID logprob record {record.sample_id} diagnostics must be an object"
            )
        if "format_valid" in record.value:
            format_valid = record.value["format_valid"]
            sampled_choice = record.value.get("sampled_choice")
            if format_valid is not None and not isinstance(format_valid, bool):
                raise CheckpointValidationError(
                    f"INVALID logprob record {record.sample_id} format_valid must be "
                    "boolean or null"
                )
            if sampled_choice not in {None, "Yes", "No"}:
                raise CheckpointValidationError(
                    f"INVALID logprob record {record.sample_id} has invalid sampled_choice"
                )
            if (
                format_valid is None
                and sampled_choice is not None
                or format_valid is not None
                and format_valid != (sampled_choice in {"Yes", "No"})
            ):
                raise CheckpointValidationError(
                    f"INVALID logprob record {record.sample_id} has inconsistent format fields"
                )


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    try:
        serialized = canonical_json(records)
    except ValueError as exc:
        raise CheckpointValidationError(
            f"checkpoint records are not finite canonical JSON: {exc}"
        ) from exc
    return sha256_text(serialized)


def _reject_nonfinite_json_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_json_numbers(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite_json_numbers(nested, f"{path}[{index}]")


def _validate_stage(stage: str) -> None:
    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"unknown logical stage: {stage!r}")


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or run_id in {".", ".."}
        or not _SAFE_RUN_ID_RE.fullmatch(run_id)
    ):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize checkpoint mutations across threads and Unix processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CheckpointValidationError(
            f"checkpoint lock must not be a symbolic link: {path}"
        )
    key = str(path.resolve())
    with _LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(key, threading.Lock())
    with process_lock:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise CheckpointValidationError(
                f"cannot safely open checkpoint lock: {path}"
            ) from exc
        handle = os.fdopen(descriptor, "a+", encoding="utf-8")
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows fallback uses thread lock
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                pass
            handle.close()


def _normalize_expected_ids(
    expected: Mapping[str, Sequence[str]],
) -> Dict[str, Tuple[str, ...]]:
    if not isinstance(expected, Mapping):
        raise TypeError("expected_sample_ids must be a mapping")
    normalized: Dict[str, Tuple[str, ...]] = {}
    unknown_stages = set(expected) - set(LOGICAL_STAGES)
    if unknown_stages:
        raise ValueError(f"unknown expected-ID stage: {sorted(unknown_stages)[0]}")
    for stage in LOGICAL_STAGES:
        values = tuple(expected.get(stage, ()))
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{stage} expected sample IDs must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{stage} expected sample IDs contain duplicates")
        normalized[stage] = values
    return normalized


def _run_identity(
    *,
    schema_version: int,
    run_id: str,
    pipeline: str,
    created_at: str,
    config_fingerprint: str,
    data_fingerprint: str,
    prompt_fingerprint: str,
    sampling_plan: Mapping[str, Any],
    expected_sample_ids: Mapping[str, Sequence[str]],
    code_version: Optional[str],
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "pipeline": pipeline,
        "created_at": created_at,
        "config_fingerprint": config_fingerprint,
        "data_fingerprint": data_fingerprint,
        "prompt_fingerprint": prompt_fingerprint,
        "sampling_plan": sampling_plan,
        "expected_sample_ids": {
            stage: list(expected_sample_ids.get(stage, ())) for stage in LOGICAL_STAGES
        },
        "code_version": code_version,
    }


@dataclass(frozen=True)
class RunManifest:
    """Immutable run identity and exact expected sampling plan."""

    run_id: str
    pipeline: str
    created_at: str
    config: Mapping[str, Any]
    config_fingerprint: str
    data_hashes: Mapping[str, str]
    data_fingerprint: str
    prompt_hashes: Mapping[str, str]
    prompt_fingerprint: str
    sampling_plan: Mapping[str, Any]
    expected_sample_ids: Mapping[str, Tuple[str, ...]]
    run_fingerprint: str
    code_version: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if not isinstance(self.pipeline, str) or not self.pipeline:
            raise ValueError("pipeline must be a non-empty string")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != SCHEMA_VERSION
        ):
            raise CheckpointValidationError(
                f"unsupported manifest schema version: {self.schema_version}"
            )
        try:
            parsed_created_at = datetime.fromisoformat(
                self.created_at.replace("Z", "+00:00")
            )
        except (AttributeError, ValueError) as exc:
            raise CheckpointValidationError(
                "created_at must be an ISO-8601 timestamp"
            ) from exc
        if parsed_created_at.tzinfo is None:
            raise CheckpointValidationError("created_at must include a timezone")
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")
        if not isinstance(self.data_hashes, Mapping) or not isinstance(
            self.prompt_hashes, Mapping
        ):
            raise TypeError("data_hashes and prompt_hashes must be mappings")
        if not isinstance(self.sampling_plan, Mapping):
            raise TypeError("sampling_plan must be a mapping")

        # Detach from caller-owned mutable structures and normalize all nested
        # values to their persisted JSON representation.
        object.__setattr__(
            self, "config", json.loads(canonical_json(dict(self.config)))
        )
        object.__setattr__(
            self, "data_hashes", json.loads(canonical_json(dict(self.data_hashes)))
        )
        object.__setattr__(
            self, "prompt_hashes", json.loads(canonical_json(dict(self.prompt_hashes)))
        )
        object.__setattr__(
            self, "sampling_plan", json.loads(canonical_json(dict(self.sampling_plan)))
        )
        normalized_expected = _normalize_expected_ids(self.expected_sample_ids)
        object.__setattr__(self, "expected_sample_ids", normalized_expected)

        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Recheck nested mutable state before every persistence boundary."""

        if self.config_fingerprint != hash_mapping(self.config):
            raise CheckpointValidationError("config fingerprint does not match config")
        if self.data_fingerprint != hash_mapping(self.data_hashes):
            raise CheckpointValidationError(
                "data fingerprint does not match data hashes"
            )
        if self.prompt_fingerprint != hash_mapping(self.prompt_hashes):
            raise CheckpointValidationError(
                "prompt fingerprint does not match prompt hashes"
            )
        if self.run_fingerprint != self.compute_run_fingerprint():
            raise CheckpointValidationError("run fingerprint does not match manifest")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        pipeline: str,
        config: Mapping[str, Any],
        data_hashes: Mapping[str, str],
        prompt_hashes: Mapping[str, str],
        sampling_plan: Mapping[str, Any],
        expected_sample_ids: Mapping[str, Sequence[str]],
        code_version: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> "RunManifest":
        normalized_expected = _normalize_expected_ids(expected_sample_ids)
        created = created_at or datetime.now(timezone.utc).isoformat()
        # Canonical round trips reject non-JSON run state and normalize tuples
        # to lists, keeping persisted and in-memory fingerprints identical.
        config_copy = json.loads(canonical_json(dict(config)))
        data_copy = json.loads(canonical_json(dict(data_hashes)))
        prompt_copy = json.loads(canonical_json(dict(prompt_hashes)))
        plan_copy = json.loads(canonical_json(dict(sampling_plan)))
        config_fingerprint = hash_mapping(config_copy)
        data_fingerprint = hash_mapping(data_copy)
        prompt_fingerprint = hash_mapping(prompt_copy)
        fingerprint = hash_mapping(
            _run_identity(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                pipeline=pipeline,
                created_at=created,
                config_fingerprint=config_fingerprint,
                data_fingerprint=data_fingerprint,
                prompt_fingerprint=prompt_fingerprint,
                sampling_plan=plan_copy,
                expected_sample_ids=normalized_expected,
                code_version=code_version,
            )
        )

        return cls(
            run_id=run_id,
            pipeline=pipeline,
            created_at=created,
            config=config_copy,
            config_fingerprint=config_fingerprint,
            data_hashes=data_copy,
            data_fingerprint=data_fingerprint,
            prompt_hashes=prompt_copy,
            prompt_fingerprint=prompt_fingerprint,
            sampling_plan=plan_copy,
            expected_sample_ids=normalized_expected,
            run_fingerprint=fingerprint,
            code_version=code_version,
        )

    def compute_run_fingerprint(self) -> str:
        return hash_mapping(
            _run_identity(
                schema_version=self.schema_version,
                run_id=self.run_id,
                pipeline=self.pipeline,
                created_at=self.created_at,
                config_fingerprint=self.config_fingerprint,
                data_fingerprint=self.data_fingerprint,
                prompt_fingerprint=self.prompt_fingerprint,
                sampling_plan=self.sampling_plan,
                expected_sample_ids=self.expected_sample_ids,
                code_version=self.code_version,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        self.validate_integrity()
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "created_at": self.created_at,
            "config": dict(self.config),
            "config_fingerprint": self.config_fingerprint,
            "data_hashes": dict(self.data_hashes),
            "data_fingerprint": self.data_fingerprint,
            "prompt_hashes": dict(self.prompt_hashes),
            "prompt_fingerprint": self.prompt_fingerprint,
            "sampling_plan": dict(self.sampling_plan),
            "expected_sample_ids": {
                stage: list(self.expected_sample_ids.get(stage, ()))
                for stage in LOGICAL_STAGES
            },
            "run_fingerprint": self.run_fingerprint,
            "code_version": self.code_version,
        }
        return json.loads(canonical_json(payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(payload, Mapping):
            raise CheckpointValidationError("manifest JSON root must be an object")
        required = (
            "schema_version",
            "run_id",
            "pipeline",
            "created_at",
            "config",
            "config_fingerprint",
            "data_hashes",
            "data_fingerprint",
            "prompt_hashes",
            "prompt_fingerprint",
            "sampling_plan",
            "expected_sample_ids",
            "run_fingerprint",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise CheckpointValidationError(f"manifest is missing field: {missing[0]}")
        try:
            return cls(
                schema_version=payload["schema_version"],
                run_id=payload["run_id"],
                pipeline=payload["pipeline"],
                created_at=payload["created_at"],
                config=payload["config"],
                config_fingerprint=payload["config_fingerprint"],
                data_hashes=payload["data_hashes"],
                data_fingerprint=payload["data_fingerprint"],
                prompt_hashes=payload["prompt_hashes"],
                prompt_fingerprint=payload["prompt_fingerprint"],
                sampling_plan=payload["sampling_plan"],
                expected_sample_ids=payload["expected_sample_ids"],
                run_fingerprint=payload["run_fingerprint"],
                code_version=payload.get("code_version"),
            )
        except CheckpointValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointValidationError(f"invalid manifest: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence for a directory entry after atomic replacement."""

    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(str(directory), directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some platforms (notably Windows) do not permit opening or fsyncing a
        # directory. File contents have already been fsynced at call sites.
        pass


def atomic_write_json(path: Union[str, Path], payload: Any) -> None:
    """Atomically replace a JSON file using a temporary file in its directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _load_json_object(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise CheckpointValidationError(
            f"checkpoint JSON must not be a symbolic link: {path}"
        )
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise CheckpointValidationError(
            f"cannot load checkpoint {path}: {exc}"
        ) from exc

    class DuplicateJsonKey(ValueError):
        pass

    def unique_object_pairs(pairs: list) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJsonKey(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_json_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant is not allowed: {value}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=unique_object_pairs,
            parse_constant=reject_json_constant,
        )
        _reject_nonfinite_json_numbers(parsed)
    except DuplicateJsonKey as exc:
        raise CheckpointValidationError(
            f"cannot load checkpoint {path}: duplicate_json_key: {exc}"
        ) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"cannot load checkpoint {path}: invalid_json: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise CheckpointValidationError(
            f"cannot load checkpoint {path}: json_root_not_object: "
            "JSON root must be an object"
        )
    return parsed


def dependencies_for(resume_stage: str) -> Tuple[str, ...]:
    _validate_stage(resume_stage)
    return STAGE_DEPENDENCIES[resume_stage]


def validate_record_ids(
    expected_ids: Sequence[str],
    actual_ids: Iterable[str],
    *,
    require_complete: bool = True,
) -> Tuple[str, ...]:
    """Validate duplicate/unknown IDs and return missing IDs in expected order."""

    expected = tuple(expected_ids)
    if len(set(expected)) != len(expected):
        raise CheckpointValidationError("expected sample IDs contain duplicates")
    actual = list(actual_ids)
    if len(set(actual)) != len(actual):
        raise CheckpointValidationError("checkpoint contains duplicate sample IDs")

    expected_set = set(expected)
    unknown = [sample_id for sample_id in actual if sample_id not in expected_set]
    if unknown:
        raise CheckpointValidationError(
            f"checkpoint contains unknown sample ID: {unknown[0]}"
        )
    actual_set = set(actual)
    missing = tuple(sample_id for sample_id in expected if sample_id not in actual_set)
    if require_complete and missing:
        raise CheckpointValidationError(
            f"checkpoint is missing sample ID: {missing[0]}"
        )
    return missing


def _coerce_records(
    records: Iterable[Union[ExperimentRecord, Mapping[str, Any]]],
) -> List[ExperimentRecord]:
    result: List[ExperimentRecord] = []
    for record in records:
        result.append(
            record
            if isinstance(record, ExperimentRecord)
            else ExperimentRecord.from_dict(record)
        )
    return result


class CheckpointStore:
    """Filesystem store for one manifest and immutable per-stage chunks."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self._stage_authority_cache: Dict[
            Tuple[str, str, str], _StageAuthorityCache
        ] = {}

    def run_directory(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        root = self.root.resolve()
        unresolved = root / run_id
        if unresolved.is_symlink():
            raise CheckpointValidationError(
                f"checkpoint run directory must not be a symbolic link: {unresolved}"
            )
        candidate = unresolved.resolve()
        if candidate.parent != root:
            raise ValueError("run_id resolves outside the checkpoint root")
        return candidate

    def _run_lock_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / ".checkpoint.lock"

    def manifest_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "manifest.json"

    def save_manifest(self, manifest: RunManifest, *, overwrite: bool = False) -> Path:
        path = self.manifest_path(manifest.run_id)
        manifest.validate_integrity()
        with _exclusive_file_lock(self._run_lock_path(manifest.run_id)):
            if path.exists():
                existing = RunManifest.from_dict(_load_json_object(path))
                if canonical_json(existing.to_dict()) == canonical_json(
                    manifest.to_dict()
                ):
                    return path
                if not overwrite:
                    raise FileExistsError(
                        f"a different manifest already exists: {path}"
                    )
                if any(path.parent.glob("*/chunk_*.json")):
                    raise CheckpointValidationError(
                        "cannot overwrite a manifest after checkpoint shards exist"
                    )
            atomic_write_json(path, manifest.to_dict())
        return path

    def load_manifest(self, run_id: str) -> RunManifest:
        path = self.manifest_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        return RunManifest.from_dict(_load_json_object(path))

    def stage_directory(self, manifest: RunManifest, stage: str) -> Path:
        _validate_stage(stage)
        run_directory = self.run_directory(manifest.run_id)
        candidate = run_directory / stage
        if candidate.is_symlink():
            raise CheckpointValidationError(
                f"checkpoint stage directory must not be a symbolic link: {candidate}"
            )
        resolved = candidate.resolve()
        if resolved.parent != run_directory:
            raise CheckpointValidationError(
                "checkpoint stage directory resolves outside the run directory"
            )
        return candidate

    def _chunk_path(self, manifest: RunManifest, stage: str, chunk_index: int) -> Path:
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise ValueError("chunk_index must be a non-negative integer")
        return self.stage_directory(manifest, stage) / f"chunk_{chunk_index:08d}.json"

    def next_chunk_index(self, manifest: RunManifest, stage: str) -> int:
        """Return the next immutable shard index for a logical stage."""

        stage_dir = self.stage_directory(manifest, stage)
        indexes: List[int] = []
        if stage_dir.exists():
            for path in stage_dir.glob("chunk_*.json"):
                match = re.fullmatch(r"chunk_(\d{8})\.json", path.name)
                if match:
                    indexes.append(int(match.group(1)))
        return max(indexes, default=-1) + 1

    def _stage_index_path(self, manifest: RunManifest, stage: str) -> Path:
        return self.stage_directory(manifest, stage) / ".sample-index.sqlite3"

    @staticmethod
    def _file_signature(path: Path) -> _FileSignature:
        stat_result = path.lstat()
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_mode,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    def _stage_cache_key(
        self, manifest: RunManifest, stage: str
    ) -> Tuple[str, str, str]:
        return (
            str(self.stage_directory(manifest, stage).resolve()),
            manifest.run_fingerprint,
            stage,
        )

    def _cached_stage_is_unchanged(
        self,
        cache: _StageAuthorityCache,
        chunk_paths: Sequence[Path],
        index_path: Path,
    ) -> bool:
        """Return whether files still match this store's last validated view."""

        if not index_path.exists() or index_path.is_symlink():
            return False
        if {path.name for path in chunk_paths} != set(cache.chunk_signatures):
            return False
        try:
            if self._file_signature(index_path) != cache.index_signature:
                return False
            return all(
                self._file_signature(path) == cache.chunk_signatures[path.name]
                for path in chunk_paths
            )
        except OSError:
            return False

    @staticmethod
    def _create_index_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_name TEXT PRIMARY KEY,
                sample_count INTEGER NOT NULL CHECK(sample_count >= 0)
            );
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                chunk_name TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS chunks_generation_insert
            AFTER INSERT ON chunks BEGIN
                UPDATE metadata
                SET value = CAST(value AS INTEGER) + 1
                WHERE key = 'content_generation';
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_generation_update
            AFTER UPDATE ON chunks BEGIN
                UPDATE metadata
                SET value = CAST(value AS INTEGER) + 1
                WHERE key = 'content_generation';
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_generation_delete
            AFTER DELETE ON chunks BEGIN
                UPDATE metadata
                SET value = CAST(value AS INTEGER) + 1
                WHERE key = 'content_generation';
            END;
            CREATE TRIGGER IF NOT EXISTS samples_generation_insert
            AFTER INSERT ON samples BEGIN
                UPDATE metadata
                SET value = CAST(value AS INTEGER) + 1
                WHERE key = 'content_generation';
            END;
            CREATE TRIGGER IF NOT EXISTS samples_generation_update
            AFTER UPDATE ON samples BEGIN
                UPDATE metadata
                SET value = CAST(value AS INTEGER) + 1
                WHERE key = 'content_generation';
            END;
            CREATE TRIGGER IF NOT EXISTS samples_generation_delete
            AFTER DELETE ON samples BEGIN
                UPDATE metadata
                SET value = CAST(value AS INTEGER) + 1
                WHERE key = 'content_generation';
            END;
            """
        )

    @staticmethod
    def _expected_index_metadata(
        manifest: RunManifest,
        stage: str,
        *,
        chunk_count: int,
        sample_count: int,
    ) -> Dict[str, str]:
        return {
            "index_schema_version": _INDEX_SCHEMA_VERSION,
            "run_fingerprint": manifest.run_fingerprint,
            "stage": stage,
            "content_generation": str(chunk_count + sample_count),
        }

    def _rebuild_stage_index(
        self,
        manifest: RunManifest,
        stage: str,
        index_path: Path,
        chunk_sample_ids: Sequence[Tuple[Path, Sequence[str]]],
    ) -> None:
        """Atomically rebuild the disposable ID index from immutable JSON shards."""

        index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(temporary_path)
            self._create_index_schema(connection)
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("index_schema_version", _INDEX_SCHEMA_VERSION),
                    ("run_fingerprint", manifest.run_fingerprint),
                    ("stage", stage),
                    ("content_generation", "0"),
                ),
            )
            for chunk_path, sample_ids in chunk_sample_ids:
                connection.execute(
                    "INSERT INTO chunks(chunk_name, sample_count) VALUES (?, ?)",
                    (chunk_path.name, len(sample_ids)),
                )
                connection.executemany(
                    "INSERT INTO samples(sample_id, chunk_name) VALUES (?, ?)",
                    ((sample_id, chunk_path.name) for sample_id in sample_ids),
                )
            connection.commit()
            connection.close()
            connection = None
            os.replace(temporary_path, index_path)
            _fsync_directory(index_path.parent)
        except sqlite3.IntegrityError as exc:
            raise CheckpointValidationError(
                f"duplicate sample ID across checkpoint chunks in {stage}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()
            temporary_path.unlink(missing_ok=True)

    def _open_synchronized_stage_index(
        self,
        manifest: RunManifest,
        stage: str,
    ) -> Tuple[sqlite3.Connection, frozenset[str]]:
        """Open an index verified against the authoritative immutable shards."""

        stage_dir = self.stage_directory(manifest, stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._stage_index_path(manifest, stage)
        if index_path.is_symlink():
            raise CheckpointValidationError(
                f"checkpoint sample index must not be a symbolic link: {index_path}"
            )
        chunk_paths = tuple(sorted(stage_dir.glob("chunk_*.json")))
        cache_key = self._stage_cache_key(manifest, stage)
        cached = self._stage_authority_cache.get(cache_key)
        if cached is not None and self._cached_stage_is_unchanged(
            cached, chunk_paths, index_path
        ):
            connection: sqlite3.Connection | None = None
            try:
                connection = sqlite3.connect(index_path)
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                if metadata == self._expected_index_metadata(
                    manifest,
                    stage,
                    chunk_count=len(cached.chunk_sample_ids),
                    sample_count=len(cached.sample_chunks),
                ):
                    return connection, frozenset(cached.sample_chunks)
                connection.close()
            except sqlite3.DatabaseError:
                if connection is not None:
                    try:
                        connection.close()
                    except sqlite3.DatabaseError:
                        pass
            # A signature match is only an optimization hint. If even the
            # compact metadata check disagrees, discard it and re-establish the
            # view from the authoritative JSON files below.
            self._stage_authority_cache.pop(cache_key, None)

        chunk_sample_ids: List[Tuple[Path, Sequence[str]]] = []
        authoritative_chunks: Dict[str, int] = {}
        authoritative_samples: Dict[str, str] = {}
        for chunk_path in chunk_paths:
            records = self._load_chunk(manifest, stage, chunk_path)
            sample_ids = tuple(record.sample_id for record in records)
            chunk_sample_ids.append((chunk_path, sample_ids))
            authoritative_chunks[chunk_path.name] = len(sample_ids)
            for record in records:
                if record.sample_id in authoritative_samples:
                    raise CheckpointValidationError(
                        f"duplicate sample ID across checkpoint chunks in {stage}: "
                        f"{record.sample_id}"
                    )
                authoritative_samples[record.sample_id] = chunk_path.name

        rebuild = not index_path.exists()
        connection: sqlite3.Connection | None = None
        if not rebuild:
            try:
                connection = sqlite3.connect(index_path)
                self._create_index_schema(connection)
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                indexed_chunks = dict(
                    connection.execute("SELECT chunk_name, sample_count FROM chunks")
                )
                indexed_samples = dict(
                    connection.execute("SELECT sample_id, chunk_name FROM samples")
                )
                quick_check = connection.execute("PRAGMA quick_check").fetchone()
                rebuild = (
                    metadata
                    != self._expected_index_metadata(
                        manifest,
                        stage,
                        chunk_count=len(authoritative_chunks),
                        sample_count=len(authoritative_samples),
                    )
                    or indexed_chunks != authoritative_chunks
                    or indexed_samples != authoritative_samples
                    or quick_check != ("ok",)
                )
            except (sqlite3.DatabaseError, TypeError, ValueError):
                rebuild = True
            finally:
                if rebuild and connection is not None:
                    connection.close()
                    connection = None

        if rebuild:
            self._rebuild_stage_index(
                manifest,
                stage,
                index_path,
                chunk_sample_ids,
            )
            connection = sqlite3.connect(index_path)
        assert connection is not None
        self._stage_authority_cache[cache_key] = _StageAuthorityCache(
            chunk_signatures={
                chunk_path.name: self._file_signature(chunk_path)
                for chunk_path in chunk_paths
            },
            chunk_sample_ids={
                chunk_path.name: tuple(sample_ids)
                for chunk_path, sample_ids in chunk_sample_ids
            },
            sample_chunks=dict(authoritative_samples),
            index_signature=self._file_signature(index_path),
        )
        return connection, frozenset(authoritative_samples)

    def _cache_appended_chunk(
        self,
        manifest: RunManifest,
        stage: str,
        target: Path,
        sample_ids: Sequence[str],
    ) -> None:
        """Advance a warm authority cache after JSON and SQLite both commit."""

        cache = self._stage_authority_cache.get(self._stage_cache_key(manifest, stage))
        if cache is None:
            return
        cache.chunk_signatures[target.name] = self._file_signature(target)
        cache.chunk_sample_ids[target.name] = tuple(sample_ids)
        cache.sample_chunks.update((sample_id, target.name) for sample_id in sample_ids)
        cache.index_signature = self._file_signature(
            self._stage_index_path(manifest, stage)
        )

    def write_chunk(
        self,
        manifest: RunManifest,
        stage: str,
        chunk_index: int,
        records: Iterable[Union[ExperimentRecord, Mapping[str, Any]]],
    ) -> Path:
        _validate_stage(stage)
        manifest.validate_integrity()
        normalized = _coerce_records(records)
        if not normalized:
            raise ValueError("checkpoint chunks must contain at least one record")
        for record in normalized:
            if record.stage != stage:
                raise CheckpointValidationError(
                    f"record {record.sample_id} has stage {record.stage!r}, expected {stage!r}"
                )
            _validate_checkpoint_record(manifest, record)

        expected = manifest.expected_sample_ids[stage]
        validate_record_ids(
            expected,
            (record.sample_id for record in normalized),
            require_complete=False,
        )

        target = self._chunk_path(manifest, stage, chunk_index)
        records_payload = [record.to_dict() for record in normalized]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest.run_id,
            "run_fingerprint": manifest.run_fingerprint,
            "stage": stage,
            "chunk_index": chunk_index,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "records_sha256": _records_sha256(records_payload),
            "records": records_payload,
        }

        with _exclusive_file_lock(self._run_lock_path(manifest.run_id)):
            disk_manifest = self.load_manifest(manifest.run_id)
            if disk_manifest.run_fingerprint != manifest.run_fingerprint:
                raise CheckpointValidationError(
                    "provided manifest does not match the manifest stored on disk"
                )
            index, authoritative_sample_ids = self._open_synchronized_stage_index(
                manifest, stage
            )
            try:
                if target.exists():
                    existing_records = self._load_chunk(manifest, stage, target)
                    if canonical_json(
                        [record.to_dict() for record in existing_records]
                    ) == canonical_json(payload["records"]):
                        return target
                    raise FileExistsError(
                        f"checkpoint chunk already exists with different data: {target}"
                    )

                sample_ids = [record.sample_id for record in normalized]
                duplicate_ids = sorted(
                    set(sample_ids).intersection(authoritative_sample_ids)
                )
                if duplicate_ids:
                    raise CheckpointValidationError(
                        "sample ID already exists in another checkpoint chunk: "
                        f"{duplicate_ids[0]}"
                    )

                atomic_write_json(target, payload)
                with index:
                    index.execute(
                        "INSERT INTO chunks(chunk_name, sample_count) VALUES (?, ?)",
                        (target.name, len(sample_ids)),
                    )
                    index.executemany(
                        "INSERT INTO samples(sample_id, chunk_name) VALUES (?, ?)",
                        ((sample_id, target.name) for sample_id in sample_ids),
                    )
            finally:
                index.close()
            self._cache_appended_chunk(
                manifest,
                stage,
                target,
                sample_ids,
            )
        return target

    def _load_chunk(
        self,
        manifest: RunManifest,
        stage: str,
        path: Path,
    ) -> List[ExperimentRecord]:
        stage_directory = self.stage_directory(manifest, stage)
        if path.is_symlink() or path.resolve().parent != stage_directory.resolve():
            raise CheckpointValidationError(
                f"checkpoint chunk must be a regular file within its stage: {path}"
            )
        payload = _load_json_object(path)
        schema_version = payload.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != SCHEMA_VERSION
        ):
            raise CheckpointValidationError(f"unsupported checkpoint schema in {path}")
        if payload.get("run_id") != manifest.run_id:
            raise CheckpointValidationError(f"checkpoint run_id mismatch in {path}")
        if payload.get("run_fingerprint") != manifest.run_fingerprint:
            raise CheckpointValidationError(
                f"checkpoint fingerprint mismatch in {path}"
            )
        if payload.get("stage") != stage:
            raise CheckpointValidationError(f"checkpoint stage mismatch in {path}")
        chunk_index = payload.get("chunk_index")
        expected_name = (
            f"chunk_{chunk_index:08d}.json"
            if isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index >= 0
            else None
        )
        if expected_name != path.name:
            raise CheckpointValidationError(
                f"checkpoint chunk index mismatch in {path}"
            )
        records_payload = payload.get("records")
        if not isinstance(records_payload, list):
            raise CheckpointValidationError(
                f"checkpoint records must be a list in {path}"
            )
        stored_records_sha256 = payload.get("records_sha256")
        if not isinstance(stored_records_sha256, str) or not _SHA256_RE.fullmatch(
            stored_records_sha256
        ):
            raise CheckpointValidationError(
                f"checkpoint records_sha256 is missing or invalid in {path}"
            )
        actual_records_sha256 = _records_sha256(records_payload)
        if stored_records_sha256 != actual_records_sha256:
            raise CheckpointValidationError(
                f"checkpoint records digest mismatch in {path}"
            )
        try:
            records = _coerce_records(records_payload)
        except (TypeError, ValueError) as exc:
            raise CheckpointValidationError(
                f"invalid checkpoint record in {path}: {exc}"
            ) from exc
        for record in records:
            if record.stage != stage:
                raise CheckpointValidationError(
                    f"record {record.sample_id} has wrong stage in {path}"
                )
            _validate_checkpoint_record(manifest, record)
        return records

    def load_stage(
        self,
        manifest: RunManifest,
        stage: str,
        *,
        require_complete: bool = True,
    ) -> Dict[str, ExperimentRecord]:
        _validate_stage(stage)
        stage_dir = self.stage_directory(manifest, stage)
        records_by_id: Dict[str, ExperimentRecord] = {}
        if stage_dir.exists():
            for path in sorted(stage_dir.glob("chunk_*.json")):
                for record in self._load_chunk(manifest, stage, path):
                    if record.sample_id in records_by_id:
                        raise CheckpointValidationError(
                            f"duplicate sample ID across checkpoint chunks: {record.sample_id}"
                        )
                    records_by_id[record.sample_id] = record

        expected = manifest.expected_sample_ids[stage]
        validate_record_ids(expected, records_by_id, require_complete=require_complete)
        return {
            sample_id: records_by_id[sample_id]
            for sample_id in expected
            if sample_id in records_by_id
        }

    def missing_sample_ids(self, manifest: RunManifest, stage: str) -> Tuple[str, ...]:
        completed = self.load_stage(manifest, stage, require_complete=False)
        return validate_record_ids(
            manifest.expected_sample_ids[stage],
            completed,
            require_complete=False,
        )

    def validate_resume_dependencies(
        self,
        manifest: RunManifest,
        resume_stage: str,
    ) -> Dict[str, Dict[str, ExperimentRecord]]:
        """Load every complete dependency required before rerunning a stage."""

        loaded: Dict[str, Dict[str, ExperimentRecord]] = {}
        for dependency in dependencies_for(resume_stage):
            loaded[dependency] = self.load_stage(
                manifest,
                dependency,
                require_complete=True,
            )
        return loaded
