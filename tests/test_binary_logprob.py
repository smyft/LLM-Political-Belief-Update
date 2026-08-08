import json
import math
from types import SimpleNamespace

import pytest

from src.models.binary_logprob import (
    LOGPROB_NUMERIC_TOLERANCE,
    MAX_CANDIDATE_TOKEN_IDS,
    build_yes_no_candidate_map,
    default_candidate_surfaces,
    score_yes_no_candidates,
)


class SurfaceTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.encodings.get(text, [900, 901])


def lp(value, text=None):
    return SimpleNamespace(logprob=value, decoded_token=text)


def test_default_surface_set_is_bounded():
    surfaces = default_candidate_surfaces()
    assert len(surfaces["Yes"]) + len(surfaces["No"]) == 120
    assert 120 <= MAX_CANDIDATE_TOKEN_IDS


def test_candidate_map_keeps_only_deduplicated_single_token_variants():
    tokenizer = SurfaceTokenizer(
        {
            "Yes": [1],
            " yes": [2],
            "YES!": [1],  # duplicate token ID for the same label
            "No": [3],
            "\nNO!": [4],
            " no": [8, 9],  # multi-token variants are intentionally excluded
        }
    )

    candidate_map = build_yes_no_candidate_map(tokenizer)

    assert candidate_map == {1: "Yes", 2: "Yes", 3: "No", 4: "No"}


def test_candidate_map_rejects_cross_label_token_collision():
    tokenizer = SurfaceTokenizer({"Yes": [7], "No": [7]})

    with pytest.raises(ValueError, match="maps to both"):
        build_yes_no_candidate_map(tokenizer)


def test_candidate_map_excludes_unknown_and_special_tokens():
    tokenizer = SurfaceTokenizer(
        {
            "Yes": [1],
            "YES!": [0],
            "No": [2],
            "NO!": [0],
            " yes": [99],
        }
    )
    tokenizer.unk_token_id = 0
    tokenizer.all_special_ids = [0, 99]

    assert build_yes_no_candidate_map(tokenizer) == {1: "Yes", 2: "No"}


def test_candidate_map_rejects_missing_label():
    tokenizer = SurfaceTokenizer({"Yes": [7]})

    with pytest.raises(ValueError, match="No"):
        build_yes_no_candidate_map(tokenizer)


def test_negative_candidate_token_ids_are_rejected():
    tokenizer = SurfaceTokenizer({"Yes": [-1], "No": [2]})
    with pytest.raises(ValueError, match="negative token ID"):
        build_yes_no_candidate_map(
            tokenizer,
            surfaces={"Yes": ("Yes",), "No": ("No",)},
        )

    result = score_yes_no_candidates({-1: -0.1, 2: -0.2}, {-1: "Yes", 2: "No"})
    assert result["valid"] is False
    assert result["error"] == "invalid_candidate_token_id"


def test_candidate_map_enforces_vllm_limit():
    tokenizer = SurfaceTokenizer({"Yes": [1], " yes": [2], "No": [3]})

    with pytest.raises(ValueError, match="at most 2"):
        build_yes_no_candidate_map(tokenizer, max_candidates=2)


def test_score_aggregates_variants_and_reports_captured_mass():
    candidate_map = {1: "Yes", 2: "Yes", 3: "No"}
    logprobs = {
        1: lp(math.log(0.2), "Yes"),
        2: lp(math.log(0.3), " yes"),
        3: lp(math.log(0.1), "No"),
        99: lp(math.log(0.4), "Maybe"),  # sampled token may be an extra entry
    }

    result = score_yes_no_candidates(logprobs, candidate_map, sampled_token_id=1)

    assert result["valid"] is True
    assert result["error"] is None
    assert result["format_valid"] is True
    assert result["sampled_choice"] == "Yes"
    assert result["candidate_mass"] == pytest.approx(0.6)
    assert result["residual_mass"] == pytest.approx(0.4)
    assert result["probabilities"]["Yes"] == pytest.approx(5 / 6)
    assert result["probabilities"]["No"] == pytest.approx(1 / 6)
    assert result["label_logprobs"]["Yes"] == pytest.approx(math.log(0.5))
    assert len(result["candidates"]) == 3
    json.dumps(result)  # persisted results must contain only JSON-native values


def test_non_candidate_sample_is_flagged_without_corrupting_probability_score():
    result = score_yes_no_candidates(
        {1: lp(math.log(0.4)), 2: lp(math.log(0.2)), 9: lp(math.log(0.4))},
        {1: "Yes", 2: "No"},
        sampled_token_id=9,
    )

    assert result["valid"] is True
    assert result["format_valid"] is False
    assert result["sampled_choice"] is None
    assert result["probabilities"]["Yes"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("logprobs", "error"),
    [
        (None, "first_token_logprobs_unavailable"),
        ({}, "first_token_logprobs_unavailable"),
        ({1: lp(-1.0)}, "missing_candidate_logprobs:2"),
        ({1: lp(float("nan")), 2: lp(-1.0)}, "non_finite_candidate_logprob:1"),
        ({1: lp(float("inf")), 2: lp(-1.0)}, "non_finite_candidate_logprob:1"),
        ({1: lp("bad"), 2: lp(-1.0)}, "invalid_candidate_logprob:1"),
        ({1: lp(10**10000), 2: lp(-1.0)}, "invalid_candidate_logprob:1"),
        ({1: lp(800.0), 2: lp(-1.0)}, "positive_candidate_logprob:1"),
    ],
)
def test_invalid_inputs_never_receive_imputed_probabilities(logprobs, error):
    result = score_yes_no_candidates(logprobs, {1: "Yes", 2: "No"})

    assert result["valid"] is False
    assert result["error"] == error
    assert result["probabilities"] is None
    assert result["label_logprobs"] is None
    assert result["candidate_mass"] is None
    assert result["residual_mass"] is None


def test_impossible_candidate_mass_is_invalid_instead_of_clamped():
    result = score_yes_no_candidates(
        {1: lp(0.0), 2: lp(0.0)},
        {1: "Yes", 2: "No"},
    )

    assert result["valid"] is False
    assert result["error"] == "candidate_mass_exceeds_one"
    assert result["probabilities"] is None


def test_candidate_mass_rounding_overshoot_within_tolerance_is_clamped():
    overshoot = LOGPROB_NUMERIC_TOLERANCE / 2
    result = score_yes_no_candidates(
        {1: lp(math.log(0.75)), 2: lp(math.log(0.25 + overshoot))},
        {1: "Yes", 2: "No"},
        sampled_token_id=1,
    )

    assert result["valid"] is True
    assert result["candidate_mass"] == 1.0
    assert result["residual_mass"] == 0.0
    assert result["probabilities"]["Yes"] == pytest.approx(
        0.75 / (1.0 + overshoot)
    )


def test_candidate_mass_overshoot_beyond_tolerance_is_invalid():
    overshoot = LOGPROB_NUMERIC_TOLERANCE * 2
    result = score_yes_no_candidates(
        {1: lp(math.log(0.75)), 2: lp(math.log(0.25 + overshoot))},
        {1: "Yes", 2: "No"},
        sampled_token_id=1,
    )

    assert result["valid"] is False
    assert result["error"] == "candidate_mass_exceeds_one"


def test_positive_logprob_within_tolerance_is_clamped_when_mass_is_plausible():
    result = score_yes_no_candidates(
        {
            1: lp(LOGPROB_NUMERIC_TOLERANCE / 2),
            2: lp(math.log(LOGPROB_NUMERIC_TOLERANCE / 4)),
        },
        {1: "Yes", 2: "No"},
    )

    assert result["valid"] is True
    assert result["candidates"][0]["logprob"] == 0.0
    assert result["candidate_mass"] == 1.0


def test_positive_logprob_beyond_tolerance_is_invalid():
    result = score_yes_no_candidates(
        {1: lp(LOGPROB_NUMERIC_TOLERANCE * 2), 2: lp(-2.0)},
        {1: "Yes", 2: "No"},
    )

    assert result["valid"] is False
    assert result["error"] == "positive_candidate_logprob:1"


def test_clamped_positive_logprob_does_not_hide_impossible_total_mass():
    result = score_yes_no_candidates(
        {1: lp(1e-12), 2: lp(-2.0)},
        {1: "Yes", 2: "No"},
    )

    assert result["valid"] is False
    assert result["error"] == "candidate_mass_exceeds_one"
    assert result["candidates"][0]["logprob"] == 0.0
