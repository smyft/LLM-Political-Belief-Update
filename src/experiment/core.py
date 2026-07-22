"""Shared, dependency-free primitives for experiment execution.

The concrete experiment runners deliberately keep model-specific behavior out of
this module.  The helpers here provide strict response validation, explicit
result status, stable sample identifiers, and reproducibility fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union


class ResultStatus(str, Enum):
    """Status of a parsed or generated experiment result."""

    VALID = "valid"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationResult:
    """A value together with an explicit validation outcome.

    ``INVALID`` means a response was received but did not satisfy the expected
    schema. ``ERROR`` is reserved for execution failures (transport errors,
    backend failures, and similar conditions).
    """

    status: ResultStatus
    value: Any = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    raw_response: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status is ResultStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "value": self.value,
            "error_code": self.error_code,
            "message": self.message,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class ExperimentRecord:
    """Self-contained result record used at checkpoint and compile boundaries."""

    sample_id: str
    stage: str
    metadata: Mapping[str, Any]
    status: ResultStatus
    value: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise ValueError("sample_id must be a non-empty string")
        if not self.stage or not isinstance(self.stage, str):
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "stage": self.stage,
            "metadata": dict(self.metadata),
            "status": self.status.value,
            "value": self.value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRecord":
        if not isinstance(payload, Mapping):
            raise TypeError("record payload must be a mapping")
        try:
            status = ResultStatus(payload["status"])
            return cls(
                sample_id=payload["sample_id"],
                stage=payload["stage"],
                metadata=payload["metadata"],
                status=status,
                value=payload.get("value"),
                error_code=payload.get("error_code"),
                error_message=payload.get("error_message"),
                raw_response=payload.get("raw_response"),
            )
        except KeyError as exc:
            raise ValueError(
                f"record is missing required field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class BinarySummary:
    """Binary response counts with invalid/error observations kept separate."""

    yes: int
    no: int
    invalid: int
    error: int

    @property
    def valid_total(self) -> int:
        return self.yes + self.no

    @property
    def yes_ratio(self) -> Optional[float]:
        if self.valid_total == 0:
            return None
        return self.yes / self.valid_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yes": self.yes,
            "no": self.no,
            "invalid": self.invalid,
            "error": self.error,
            "valid_total": self.valid_total,
            "yes_ratio": self.yes_ratio,
        }


def summarize_binary_results(
    results: Iterable[Union[ValidationResult, ExperimentRecord]],
) -> BinarySummary:
    """Aggregate only valid Yes/No values; never impute failures as a label."""

    yes = no = invalid = error = 0
    for result in results:
        if not isinstance(result, (ValidationResult, ExperimentRecord)):
            raise TypeError(
                "binary summaries require ValidationResult or ExperimentRecord values"
            )
        if result.status is ResultStatus.INVALID:
            invalid += 1
        elif result.status is ResultStatus.ERROR:
            error += 1
        elif result.value == "Yes":
            yes += 1
        elif result.value == "No":
            no += 1
        else:
            raise ValueError("a valid binary result must have value 'Yes' or 'No'")
    return BinarySummary(yes=yes, no=no, invalid=invalid, error=error)


_FENCED_JSON_RE = re.compile(
    r"\A\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_PERCENTAGE_RE = re.compile(
    r"\A\s*(?P<number>(?:100(?:\.0+)?|(?:\d{1,2})(?:\.\d+)?))"
    r"\s*(?:%|percent)?\s*\Z",
    flags=re.IGNORECASE,
)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is not allowed: {value}")


def _unique_object_pairs(pairs: list) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid(code: str, message: str, raw_response: Optional[str]) -> ValidationResult:
    return ValidationResult(
        status=ResultStatus.INVALID,
        error_code=code,
        message=message,
        raw_response=raw_response,
    )


def parse_json_object(response_text: str) -> ValidationResult:
    """Parse a response that is either a JSON object or one fenced JSON object.

    Surrounding prose, non-object JSON roots, duplicate keys, and Python's
    non-standard ``NaN``/``Infinity`` constants are rejected.  This avoids the
    previous behavior of guessing a JSON fragment from arbitrary reasoning.
    """

    if not isinstance(response_text, str) or not response_text.strip():
        return _invalid("empty_response", "response is empty", response_text)

    stripped = response_text.strip()
    fenced = _FENCED_JSON_RE.fullmatch(stripped)
    json_text = fenced.group("body") if fenced else stripped

    try:
        value = json.loads(
            json_text,
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as exc:
        return _invalid("duplicate_json_key", str(exc), response_text)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return _invalid("invalid_json", str(exc), response_text)

    if not isinstance(value, dict):
        return _invalid(
            "json_root_not_object", "JSON root must be an object", response_text
        )

    return ValidationResult(
        status=ResultStatus.VALID,
        value=value,
        raw_response=response_text,
    )


def parse_yes_no_response(response_text: str) -> ValidationResult:
    """Parse an object whose answer normalizes to the Yes/No enum.

    Additional object fields such as ``thinking`` are allowed, but the answer
    itself must be an unambiguous string enum after trimming and case folding.
    No raw-text fallback is used.
    """

    parsed = parse_json_object(response_text)
    if not parsed.is_valid:
        return parsed

    answer = parsed.value.get("answer")
    if not isinstance(answer, str):
        return _invalid(
            "answer_not_string",
            "answer must be the string 'Yes' or 'No'",
            response_text,
        )

    normalized = answer.strip().casefold()
    if normalized not in {"yes", "no"}:
        return _invalid(
            "invalid_yes_no_answer",
            "answer must be exactly 'Yes' or 'No'",
            response_text,
        )

    return ValidationResult(
        status=ResultStatus.VALID,
        value="Yes" if normalized == "yes" else "No",
        raw_response=response_text,
    )


def parse_percentage_response(response_text: str) -> ValidationResult:
    """Parse a percentage exclusively from a JSON object's ``answer`` field.

    Numeric JSON values and complete strings such as ``"62.5%"`` or
    ``"62.5 percent"`` are accepted. Values outside ``[0, 100]`` are invalid;
    they are never clamped. The returned value is a float so decimal responses
    remain intact until an explicit treatment-rounding policy is applied.
    """

    parsed = parse_json_object(response_text)
    if not parsed.is_valid:
        return parsed

    if "answer" not in parsed.value:
        return _invalid(
            "missing_answer", "JSON object is missing 'answer'", response_text
        )

    answer = parsed.value["answer"]
    number: Optional[float]
    try:
        if isinstance(answer, bool):
            number = None
        elif isinstance(answer, (int, float)):
            number = float(answer)
        elif isinstance(answer, str):
            match = _PERCENTAGE_RE.fullmatch(answer)
            number = float(match.group("number")) if match else None
        else:
            number = None
    except (OverflowError, TypeError, ValueError):
        number = None

    if number is None or not math.isfinite(number):
        return _invalid(
            "invalid_percentage_type",
            "answer must be a finite number or a complete percentage string",
            response_text,
        )
    if not 0.0 <= number <= 100.0:
        return _invalid(
            "percentage_out_of_range",
            "percentage answer must be between 0 and 100",
            response_text,
        )

    return ValidationResult(
        status=ResultStatus.VALID,
        value=number,
        raw_response=response_text,
    )


def _validate_canonical_mapping_keys(value: Any, path: str = "$") -> None:
    """Require string keys throughout values used for canonical JSON."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"mapping keys must be strings at {path}; got {type(key).__name__}"
                )
            _validate_canonical_mapping_keys(item, f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_canonical_mapping_keys(item, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON used by IDs and fingerprints.

    Mapping keys must be strings at every nesting level. Python's JSON encoder
    otherwise coerces integer-like keys to strings, which could give distinct
    input mappings the same fingerprint.
    """

    try:
        _validate_canonical_mapping_keys(value)
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON serializable: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it wholly into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_mapping(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return sha256_text(canonical_json(dict(value)))


def hash_templates(templates: Mapping[str, str]) -> Dict[str, str]:
    """Return per-template content hashes with deterministic key ordering."""

    if not isinstance(templates, Mapping):
        raise TypeError("templates must be a mapping")
    result: Dict[str, str] = {}
    for name in sorted(templates):
        content = templates[name]
        if not isinstance(name, str) or not isinstance(content, str):
            raise TypeError("template names and contents must be strings")
        result[name] = sha256_text(content)
    return result


def hash_files(paths: Mapping[str, Union[str, Path]]) -> Dict[str, str]:
    """Return content hashes for named data files.

    Logical names, rather than absolute local paths, are persisted so a run can
    be reproduced after moving or cloning the repository.
    """

    if not isinstance(paths, Mapping):
        raise TypeError("paths must be a mapping")
    result: Dict[str, str] = {}
    for name in sorted(paths):
        if not isinstance(name, str) or not name:
            raise ValueError("file fingerprint names must be non-empty strings")
        result[name] = sha256_file(paths[name])
    return result


def stable_sample_id(
    stage: str, metadata: Mapping[str, Any], namespace: str = "llm-pbu-v1"
) -> str:
    """Create a stable full SHA-256 sample ID from stage and metadata."""

    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    payload = {
        "namespace": namespace,
        "stage": stage,
        "metadata": dict(metadata),
    }
    return f"{stage}:{sha256_text(canonical_json(payload))}"


def make_record(
    stage: str,
    metadata: Mapping[str, Any],
    validation: ValidationResult,
    *,
    sample_id: Optional[str] = None,
) -> ExperimentRecord:
    """Build a self-contained record from a validation result."""

    record_id = sample_id or stable_sample_id(stage, metadata)
    return ExperimentRecord(
        sample_id=record_id,
        stage=stage,
        metadata=dict(metadata),
        status=validation.status,
        value=validation.value,
        error_code=validation.error_code,
        error_message=validation.message,
        raw_response=validation.raw_response,
    )
