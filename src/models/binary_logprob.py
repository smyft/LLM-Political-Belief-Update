"""Utilities for bounded, single-token Yes/No probability extraction.

The experiment intentionally scores only a finite set of tokenizer-specific
single-token spellings of ``Yes`` and ``No``.  It never asks the inference
backend for a full-vocabulary distribution.  Returned probabilities are
therefore conditional on that finite candidate set; ``candidate_mass`` records
how much of the model's full next-token probability mass the set captures.

This module has no dependency on vLLM, PyTorch, or NumPy so that the scoring
rules can be tested on a CPU-only machine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any, Literal, TypeAlias


BinaryChoice: TypeAlias = Literal["Yes", "No"]

# vLLM 0.24 caps SamplingParams.logprob_token_ids at 128.  The default
# surface-form Cartesian product below has 120 entries before token-id
# deduplication, so it is bounded independently of tokenizer vocabulary size.
MAX_CANDIDATE_TOKEN_IDS = 128

_CHOICE_CASES: Mapping[BinaryChoice, tuple[str, ...]] = {
    "Yes": ("Yes", "yes", "YES"),
    "No": ("No", "no", "NO"),
}
_PREFIXES = ("", " ", "\n", "\t")
_SUFFIXES = ("", ".", ",", "!", "?")


def default_candidate_surfaces() -> dict[BinaryChoice, tuple[str, ...]]:
    """Return the finite text variants used to discover candidate token IDs."""

    return {
        choice: tuple(
            f"{prefix}{word}{suffix}"
            for prefix in _PREFIXES
            for word in cases
            for suffix in _SUFFIXES
        )
        for choice, cases in _CHOICE_CASES.items()
    }


def _coerce_token_ids(encoded: Any) -> list[int]:
    """Normalize common tokenizer ``encode`` return types to a flat ID list."""

    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise TypeError("tokenizer.encode() must return a sequence of token IDs")

    token_ids = list(encoded)
    # Some tokenizers return a single batch dimension.
    if len(token_ids) == 1 and isinstance(token_ids[0], Sequence):
        token_ids = list(token_ids[0])
    if not all(
        isinstance(token_id, Integral) and not isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise TypeError("tokenizer.encode() returned a non-integer token ID")
    normalized = [int(token_id) for token_id in token_ids]
    if any(token_id < 0 for token_id in normalized):
        raise ValueError("tokenizer.encode() returned a negative token ID")
    return normalized


def build_yes_no_candidate_map(
    tokenizer: Any,
    *,
    surfaces: Mapping[BinaryChoice, Sequence[str]] | None = None,
    max_candidates: int = MAX_CANDIDATE_TOKEN_IDS,
) -> dict[int, BinaryChoice]:
    """Build a deduplicated mapping from one-token variants to Yes/No labels.

    Multi-token encodings are deliberately discarded because the estimator
    scores only the first answer token.  A token ID mapping to both labels is
    rejected instead of being resolved arbitrarily.
    """

    if max_candidates < 2 or max_candidates > MAX_CANDIDATE_TOKEN_IDS:
        raise ValueError(
            f"max_candidates must be between 2 and {MAX_CANDIDATE_TOKEN_IDS}"
        )
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer must provide an encode() method")

    selected_surfaces = default_candidate_surfaces() if surfaces is None else surfaces
    candidate_map: dict[int, BinaryChoice] = {}
    excluded_token_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", ()) or ())
        if isinstance(token_id, Integral) and not isinstance(token_id, bool)
    }
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    if isinstance(unknown_token_id, Integral) and not isinstance(
        unknown_token_id, bool
    ):
        excluded_token_ids.add(int(unknown_token_id))

    for choice in ("Yes", "No"):
        variants = selected_surfaces.get(choice, ())
        for surface in variants:
            if not isinstance(surface, str):
                raise TypeError("candidate surface forms must be strings")
            token_ids = _coerce_token_ids(
                tokenizer.encode(surface, add_special_tokens=False)
            )
            if len(token_ids) != 1:
                continue

            token_id = token_ids[0]
            if token_id < 0:
                raise ValueError(f"tokenizer returned negative token ID {token_id}")
            if token_id in excluded_token_ids:
                continue
            previous = candidate_map.get(token_id)
            if previous is not None and previous != choice:
                raise ValueError(
                    f"token ID {token_id} maps to both {previous!r} and {choice!r}"
                )
            candidate_map[token_id] = choice

    labels_found = set(candidate_map.values())
    missing = [choice for choice in ("Yes", "No") if choice not in labels_found]
    if missing:
        raise ValueError(
            "tokenizer has no supported single-token candidate for: "
            + ", ".join(missing)
        )
    if len(candidate_map) > max_candidates:
        raise ValueError(
            f"candidate map contains {len(candidate_map)} token IDs; "
            f"vLLM supports at most {max_candidates}"
        )

    return dict(sorted(candidate_map.items()))


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _invalid_result(
    error: str,
    *,
    sampled_token_id: int | None,
    sampled_choice: BinaryChoice | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "valid": False,
        "error": error,
        "probabilities": None,
        "label_logprobs": None,
        "candidate_mass": None,
        "residual_mass": None,
        "candidates": candidates,
        "sampled_token_id": sampled_token_id,
        "sampled_choice": sampled_choice,
        "format_valid": sampled_choice is not None,
    }


def score_yes_no_candidates(
    first_token_logprobs: Mapping[int, Any] | None,
    candidate_map: Mapping[int, BinaryChoice],
    *,
    sampled_token_id: int | None = None,
) -> dict[str, Any]:
    """Score all requested Yes/No candidates at the first generated position.

    ``first_token_logprobs`` is the vLLM-style mapping from token ID to an
    object with ``logprob`` and optionally ``decoded_token`` attributes.  Every
    candidate requested through ``logprob_token_ids`` must be present and
    finite; otherwise the result is explicitly invalid and no probability is
    imputed.

    The returned Yes/No values are conditional probabilities within the finite
    candidate set, not full-vocabulary probabilities.
    """

    normalized_map = dict(candidate_map)
    sampled_choice = normalized_map.get(sampled_token_id)
    candidates: list[dict[str, Any]] = []

    if not normalized_map:
        return _invalid_result(
            "empty_candidate_map",
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, Integral)
        or int(token_id) < 0
        for token_id in normalized_map
    ):
        return _invalid_result(
            "invalid_candidate_token_id",
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )
    if len(normalized_map) > MAX_CANDIDATE_TOKEN_IDS:
        return _invalid_result(
            "too_many_candidates",
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )
    if set(normalized_map.values()) != {"Yes", "No"}:
        return _invalid_result(
            "candidate_map_missing_label",
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )
    if not first_token_logprobs:
        return _invalid_result(
            "first_token_logprobs_unavailable",
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )

    missing_token_ids = sorted(set(normalized_map).difference(first_token_logprobs))
    if missing_token_ids:
        return _invalid_result(
            "missing_candidate_logprobs:" + ",".join(map(str, missing_token_ids)),
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )

    label_values: dict[BinaryChoice, list[float]] = {"Yes": [], "No": []}
    for token_id, choice in sorted(normalized_map.items()):
        logprob_obj = first_token_logprobs[token_id]
        raw_logprob = getattr(logprob_obj, "logprob", logprob_obj)
        try:
            logprob = float(raw_logprob)
        except (OverflowError, TypeError, ValueError):
            return _invalid_result(
                f"invalid_candidate_logprob:{token_id}",
                sampled_token_id=sampled_token_id,
                sampled_choice=sampled_choice,
                candidates=candidates,
            )
        if not math.isfinite(logprob):
            return _invalid_result(
                f"non_finite_candidate_logprob:{token_id}",
                sampled_token_id=sampled_token_id,
                sampled_choice=sampled_choice,
                candidates=candidates,
            )
        # A log probability cannot be meaningfully positive.  Permit only a
        # tiny floating-point overshoot and normalize that overshoot to zero
        # before exponentiation so malformed backend values cannot overflow.
        if logprob > 1e-9:
            return _invalid_result(
                f"positive_candidate_logprob:{token_id}",
                sampled_token_id=sampled_token_id,
                sampled_choice=sampled_choice,
                candidates=candidates,
            )
        if logprob > 0.0:
            logprob = 0.0

        decoded_token = getattr(logprob_obj, "decoded_token", None)
        candidates.append(
            {
                "token_id": int(token_id),
                "choice": choice,
                "decoded_token": (
                    decoded_token if isinstance(decoded_token, str) else None
                ),
                "logprob": logprob,
                "probability": math.exp(logprob),
            }
        )
        label_values[choice].append(logprob)

    label_logprobs = {
        choice: _logsumexp(values) for choice, values in label_values.items()
    }
    total_logprob = _logsumexp(list(label_logprobs.values()))
    candidate_mass = math.exp(total_logprob)
    # Log probabilities should describe a normalized vocabulary distribution.
    # Permit tiny floating-point overshoot, but reject substantively impossible
    # mass rather than silently concealing malformed backend data.
    if candidate_mass > 1.0 + 1e-9:
        return _invalid_result(
            "candidate_mass_exceeds_one",
            sampled_token_id=sampled_token_id,
            sampled_choice=sampled_choice,
            candidates=candidates,
        )
    candidate_mass = min(1.0, candidate_mass)

    yes_probability = math.exp(label_logprobs["Yes"] - total_logprob)
    probabilities = {
        "Yes": yes_probability,
        "No": 1.0 - yes_probability,
    }

    return {
        "valid": True,
        "error": None,
        "probabilities": probabilities,
        "label_logprobs": label_logprobs,
        "candidate_mass": candidate_mass,
        "residual_mass": 1.0 - candidate_mass,
        "candidates": candidates,
        "sampled_token_id": sampled_token_id,
        "sampled_choice": sampled_choice,
        "format_valid": sampled_choice is not None,
    }


__all__ = [
    "BinaryChoice",
    "MAX_CANDIDATE_TOKEN_IDS",
    "build_yes_no_candidate_map",
    "default_candidate_surfaces",
    "score_yes_no_candidates",
]
