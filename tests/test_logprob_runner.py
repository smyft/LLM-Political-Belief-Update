import json
import math
import shutil
from types import SimpleNamespace

import pytest

import src.experiment.logprob_experiment_runner as runner_module
from src.experiment.core import ResultStatus
from src.experiment.logprob_experiment_runner import (
    LogprobExperimentRunner,
    build_parser,
    main,
)
from src.experiment.planning import (
    StageAssignment,
    TreatmentAssignment,
    TreatmentCondition,
)
from src.models.binary_logprob import LOGPROB_NUMERIC_TOLERANCE


class FakeBackend:
    def __init__(
        self,
        *,
        chat_results=None,
        continuation_results=None,
        chat_error=None,
        continuation_error=None,
        preflight_error=None,
    ):
        self.enable_thinking = False
        self.chat_results = list(chat_results or [])
        self.continuation_results = list(continuation_results or [])
        self.chat_error = chat_error
        self.continuation_error = continuation_error
        self.preflight_error = preflight_error
        self.preflight_calls = 0
        self.chat_calls = []
        self.continuation_calls = []
        self.call_order = []
        self.closed = False

    def preflight_bounded_scoring(self):
        self.preflight_calls += 1
        self.call_order.append("preflight")
        if self.preflight_error is not None:
            raise self.preflight_error

    def preflight_continuation_scoring(self):
        raise AssertionError("runner must use fresh-turn bounded scoring")

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        self.call_order.append("chat")
        if self.chat_error is not None:
            raise self.chat_error
        return self.chat_results.pop(0)

    def chat_with_bounded_candidates(self, **kwargs):
        self.continuation_calls.append(kwargs)
        self.call_order.append("continuation")
        if self.continuation_error is not None:
            raise self.continuation_error
        return self.continuation_results.pop(0)

    def chat_with_continuation(self, **kwargs):
        del kwargs
        raise AssertionError("runner must not use assistant-message continuation")

    def close(self):
        self.closed = True


class AdaptiveFakeBackend(FakeBackend):
    """Return valid CPU-only responses for an entire tiny experiment run."""

    def __init__(self):
        super().__init__()
        self.chat_batch_sizes = []
        self.continuation_batch_sizes = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        self.call_order.append("chat")
        dialogues = kwargs["dialogue_history"]
        self.chat_batch_sizes.append(len(dialogues))
        responses = []
        for dialogue in dialogues:
            prompt = dialogue[-1]["content"]
            text = (
                '{"analysis":"estimate", "answer":55}'
                if "What percentage" in prompt
                else "bounded analysis"
            )
            responses.append({"generated_text": text, "finish_reason": "stop"})
        return responses

    def chat_with_bounded_candidates(self, **kwargs):
        self.continuation_calls.append(kwargs)
        self.call_order.append("continuation")
        count = len(kwargs["dialogue_history"])
        self.continuation_batch_sizes.append(count)
        return [score_response() for _ in range(count)]


def score_response(
    *,
    yes=0.75,
    candidate_mass=0.8,
    sampled_choice="Yes",
    format_valid=True,
):
    no = 1.0 - yes
    return {
        "generated_text": sampled_choice or "Other",
        "finish_reason": "length",
        "valid": True,
        "error": None,
        "probabilities": {"Yes": yes, "No": no},
        "label_logprobs": {
            "Yes": math.log(candidate_mass * yes),
            "No": math.log(candidate_mass * no),
        },
        "candidate_mass": candidate_mass,
        "residual_mass": 1.0 - candidate_mass,
        "candidates": [
            {
                "token_id": 11,
                "choice": "Yes",
                "decoded_token": "Yes",
                "logprob": math.log(candidate_mass * yes),
                "probability": candidate_mass * yes,
            },
            {
                "token_id": 22,
                "choice": "No",
                "decoded_token": "No",
                "logprob": math.log(candidate_mass * no),
                "probability": candidate_mass * no,
            },
        ],
        "sampled_token_id": 11 if sampled_choice == "Yes" else None,
        "sampled_choice": sampled_choice,
        "format_valid": format_valid,
    }


def baseline_assignment(
    stage="step1", *, proposal="A policy", action="Take action", index=0
):
    metadata = {
        "persona": "none",
        "category": "economy",
        "proposal": proposal,
    }
    if stage == "step3":
        metadata.update({"action_type": "public", "action": action})
    return StageAssignment(
        stage=stage,
        unit_metadata=metadata,
        replicate_id=0,
        seed=123,
        sample_id=f"{stage}:sample-{index}",
    )


def treatment_assignment(
    stage,
    kind,
    *,
    percentage=None,
    consensus_n=None,
    action="Take action",
    index=0,
):
    metadata = {
        "persona": "none",
        "category": "economy",
        "proposal": "A policy",
    }
    if stage == "step4b":
        metadata.update({"action_type": "public", "action": action})
    condition = TreatmentCondition(
        kind=kind,
        source={
            "fixed_hypothetical_survey": "hypothetical_survey",
            "simulated_persona_consensus": "simulated_persona_consensus_leave_one_out",
            "no_information_retest": "none",
            "placebo_text": "neutral_additional_text",
        }[kind],
        percentage=percentage,
        excluded_persona="none" if kind == "simulated_persona_consensus" else None,
        consensus_n=consensus_n,
    )
    return TreatmentAssignment(
        stage=stage,
        unit_metadata=metadata,
        condition=condition,
        replicate_id=0,
        seed=123,
        order_index=index,
        step2_predicted_percentage=50.0,
        survey_surprise=(percentage - 50.0) if percentage is not None else None,
        sample_id=f"{stage}:{kind}-{index}",
    )


def make_runner(backend=None, **kwargs):
    runner = LogprobExperimentRunner(
        "fake/model",
        llm_interface=backend,
        show_progress=False,
        **kwargs,
    )
    runner.load_prompt_templates()
    return runner


def test_prompt_specs_validate_every_exact_placeholder_set(tmp_path):
    runner = make_runner(FakeBackend())

    assert set(runner.load_prompt_templates()) == set(runner.prompt_specs)
    for name in {
        "step1_phase2",
        "step3_phase2",
        "step4a_phase2",
        "step4b_phase2",
    }:
        assert runner.prompt_specs[name][1] == frozenset()
        assert "{ANALYSIS_TEXT}" not in runner.prompt_templates[name]

    prompts = tmp_path / "prompts"
    shutil.copytree(runner.prompts_dir, prompts)
    path = prompts / "step1_phase1.txt"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n{UNEXPECTED}\n", encoding="utf-8"
    )
    broken = LogprobExperimentRunner(
        "fake/model", prompts_dir=prompts, llm_interface=FakeBackend()
    )
    with pytest.raises(ValueError, match="UNEXPECTED"):
        broken.load_prompt_templates()


@pytest.mark.parametrize("stage", ["step1", "step2"])
def test_preflight_failure_is_fatal_before_any_chat(stage):
    backend = FakeBackend(
        preflight_error=RuntimeError("continuation scoring is incompatible")
    )
    runner = make_runner(backend)

    with pytest.raises(RuntimeError, match="continuation scoring is incompatible"):
        runner._execute_stage_chunk(stage, [baseline_assignment(stage)])

    assert backend.preflight_calls == 1
    assert backend.chat_calls == []
    assert backend.continuation_calls == []
    assert backend.call_order == ["preflight"]


def test_preflight_runs_once_before_step2_and_binary_chat():
    backend = FakeBackend(
        chat_results=[
            [{"generated_text": json.dumps({"analysis": "estimate", "answer": 47.5})}],
            [{"generated_text": "bounded analysis", "finish_reason": "stop"}],
        ],
        continuation_results=[[score_response()]],
    )
    runner = make_runner(backend)

    step2 = runner._execute_stage_chunk("step2", [baseline_assignment("step2")])[0]
    step1 = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert step2.status is ResultStatus.VALID
    assert step1.status is ResultStatus.VALID
    assert backend.preflight_calls == 1
    assert backend.call_order == ["preflight", "chat", "chat", "continuation"]


def test_preflight_is_bound_to_backend_identity_and_reset_by_cleanup():
    first = FakeBackend()
    second = FakeBackend()
    runner = make_runner(first)

    assert runner._preflight_bounded_scoring() is first
    assert runner._preflight_bounded_scoring() is first
    runner.llm_interface = second
    assert runner._preflight_bounded_scoring() is second

    assert first.preflight_calls == 1
    assert second.preflight_calls == 1

    runner.cleanup()
    runner.llm_interface = second
    assert runner._preflight_bounded_scoring() is second
    assert second.preflight_calls == 2


def test_preflight_rejects_backend_without_disabled_thinking_contract():
    backend = FakeBackend()
    backend.enable_thinking = True
    runner = make_runner(backend)

    with pytest.raises(RuntimeError, match="must declare enable_thinking=False"):
        runner._preflight_bounded_scoring()

    assert backend.preflight_calls == 0


def test_binary_stage_scores_fresh_turn_after_completed_visible_analysis():
    backend = FakeBackend(
        chat_results=[
            [{"generated_text": "MODEL ANALYSIS {policy}", "finish_reason": "stop"}]
        ],
        continuation_results=[[score_response()]],
    )
    runner = make_runner(backend, temperature=0.7)
    assignment = baseline_assignment("step1")

    record = runner._execute_stage_chunk("step1", [assignment])[0]

    assert record.status is ResultStatus.VALID
    assert record.value["probabilities"] == {"Yes": 0.75, "No": 0.25}
    assert record.value["candidate_mass"] == pytest.approx(0.8)
    assert record.value["residual_mass"] == pytest.approx(0.2)
    assert record.value["analysis_text"] == "MODEL ANALYSIS {policy}"
    assert record.value["analysis_text_kind"] == "model_generated_visible_text"
    assert record.value["conditional_on_candidate_set"] is True
    assert record.value["scoring_temperature"] == 0.0
    assert record.value["format_valid"] is True

    continuation = backend.continuation_calls[0]
    dialogue = continuation["dialogue_history"][0]
    assert dialogue[-2] == {
        "role": "assistant",
        "content": "MODEL ANALYSIS {policy}",
    }
    assert dialogue[-1]["role"] == "user"
    assert "MODEL ANALYSIS {policy}" not in dialogue[-1]["content"]
    assert "Respond with exactly Yes or No" in dialogue[-1]["content"]
    assert all("reasoning_content" not in message for message in dialogue)
    assert continuation["max_tokens"] == 1
    assert continuation["temperature"] == 0.0
    assert "logprobs" not in continuation
    assert backend.chat_calls[0]["temperature"] == 0.7


@pytest.mark.parametrize(
    "analysis_text",
    [
        "Yes",
        "The expected benefits outweigh the costs.\nYes.",
        "<think>The expected costs dominate.</think>**No**",
        "Yes, because the expected benefits outweigh the costs.",
        "After weighing the evidence, the final answer is No.",
        "The considerations favor the proposal. Therefore, Yes.",
        json.dumps({"analysis": "done", "answer": "Yes"}),
        "```json\n" + json.dumps({"answer": False}) + "\n```",
    ],
)
def test_phase1_explicit_final_answer_is_invalid_without_continuation(analysis_text):
    backend = FakeBackend(
        chat_results=[[{"generated_text": analysis_text, "finish_reason": "stop"}]]
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.INVALID
    assert record.error_code == "phase1_contains_final_answer"
    assert record.raw_response == analysis_text
    assert backend.continuation_calls == []


def test_phase1_truncation_is_invalid_without_continuation():
    backend = FakeBackend(
        chat_results=[
            [{"generated_text": "unfinished analysis", "finish_reason": "length"}]
        ]
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.INVALID
    assert record.error_code == "phase1_truncated"
    assert record.raw_response == "unfinished analysis"
    assert backend.continuation_calls == []


def test_phase1_finish_reason_schema_error_does_not_request_continuation():
    backend = FakeBackend(
        chat_results=[[{"generated_text": "analysis", "finish_reason": 7}]]
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.ERROR
    assert record.error_code == "phase1_response_schema_error"
    assert "finish_reason" in record.error_message
    assert backend.continuation_calls == []


def test_phase1_can_discuss_yes_and_no_without_being_treated_as_a_final_answer():
    backend = FakeBackend(
        chat_results=[
            [
                {
                    "generated_text": (
                        "A Yes response emphasizes benefits, while a No response "
                        "emphasizes costs. There is no evidence that either "
                        "consideration settles the answer."
                    ),
                    "finish_reason": "stop",
                }
            ]
        ],
        continuation_results=[[score_response()]],
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.VALID
    assert len(backend.continuation_calls) == 1


def test_non_candidate_sample_is_invalid_but_retains_bounded_score_for_diagnostics():
    backend = FakeBackend(
        chat_results=[[{"generated_text": "analysis"}]],
        continuation_results=[
            [score_response(format_valid=False, sampled_choice=None)]
        ],
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.INVALID
    assert record.error_code == "sampled_token_outside_yes_no_candidates"
    assert record.value["probabilities"] == {"Yes": 0.75, "No": 0.25}
    assert record.value["candidate_mass"] == pytest.approx(0.8)
    assert record.value["residual_mass"] == pytest.approx(0.2)
    assert LogprobExperimentRunner._binary_decision(record) is None


def test_format_valid_must_agree_with_sampled_choice():
    inconsistent = score_response(format_valid=True, sampled_choice=None)
    backend = FakeBackend(
        chat_results=[[{"generated_text": "analysis"}]],
        continuation_results=[[inconsistent]],
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.ERROR
    assert record.error_code == "phase2_response_schema_error"
    assert "agree with sampled_choice" in record.error_message

    declared_invalid = FakeBackend(
        chat_results=[[{"generated_text": "analysis"}]],
        continuation_results=[
            [
                {
                    "valid": False,
                    "error": "backend_declared_invalid",
                    "sampled_choice": None,
                    "format_valid": True,
                }
            ]
        ],
    )
    malformed_runner = make_runner(declared_invalid)

    malformed_record = malformed_runner._execute_stage_chunk(
        "step1", [baseline_assignment("step1")]
    )[0]

    assert malformed_record.status is ResultStatus.ERROR
    assert malformed_record.error_code == "phase2_response_schema_error"
    assert "agree with sampled_choice" in malformed_record.error_message


def test_cross_inconsistent_score_is_per_sample_error_before_checkpoint_write():
    inconsistent = score_response()
    inconsistent["probabilities"] = {"Yes": 0.9, "No": 0.1}
    backend = FakeBackend(
        chat_results=[[{"generated_text": "analysis"}]],
        continuation_results=[[inconsistent]],
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is ResultStatus.ERROR
    assert record.error_code == "phase2_response_schema_error"
    assert "conditional probabilities do not match" in record.error_message


def test_step2_uses_strict_percentage_json_and_never_calls_continuation():
    backend = FakeBackend(
        chat_results=[
            [
                {"generated_text": "47 percent"},
                {"generated_text": '{"analysis":"estimate", "answer":47.5}'},
            ]
        ]
    )
    runner = make_runner(backend)
    assignments = [
        baseline_assignment("step2", proposal="Policy A", index=0),
        baseline_assignment("step2", proposal="Policy B", index=1),
    ]

    invalid, valid = runner._execute_stage_chunk("step2", assignments)

    assert invalid.status is ResultStatus.INVALID
    assert invalid.value is None
    assert valid.status is ResultStatus.VALID
    assert valid.value == 47.5
    assert backend.continuation_calls == []


def test_step2_truncation_is_reported_before_json_parsing():
    backend = FakeBackend(
        chat_results=[
            [{"generated_text": '{"analysis":"unfinished', "finish_reason": "length"}]
        ]
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step2", [baseline_assignment("step2")])[0]

    assert record.status is ResultStatus.INVALID
    assert record.error_code == "step2_truncated"
    assert record.raw_response == '{"analysis":"unfinished'


def test_step2_finish_reason_schema_error_is_per_sample_error():
    backend = FakeBackend(
        chat_results=[
            [
                {
                    "generated_text": '{"analysis":"estimate","answer":50}',
                    "finish_reason": 7,
                }
            ]
        ]
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step2", [baseline_assignment("step2")])[0]

    assert record.status is ResultStatus.ERROR
    assert record.error_code == "backend_response_schema_error"
    assert "finish_reason" in record.error_message


def test_unavailable_simulated_consensus_is_invalid_without_any_model_call():
    backend = FakeBackend(chat_error=AssertionError("backend must not be called"))
    runner = make_runner(backend)
    assignment = treatment_assignment(
        "step4a", "simulated_persona_consensus", percentage=None
    )

    record = runner._execute_stage_chunk("step4a", [assignment])[0]

    assert record.status is ResultStatus.INVALID
    assert record.error_code == "simulated_consensus_unavailable"
    assert backend.preflight_calls == 0
    assert backend.chat_calls == []
    assert backend.continuation_calls == []


def test_treatment_prompt_contracts_are_distinct_and_retest_reuses_baseline():
    runner = make_runner(FakeBackend())

    baseline_step1 = runner._render_phase1_prompt("step1", baseline_assignment("step1"))
    retest_step4a = runner._render_phase1_prompt(
        "step4a", treatment_assignment("step4a", "no_information_retest")
    )
    assert retest_step4a == baseline_step1

    baseline_step3 = runner._render_phase1_prompt("step3", baseline_assignment("step3"))
    retest_step4b = runner._render_phase1_prompt(
        "step4b", treatment_assignment("step4b", "no_information_retest")
    )
    assert retest_step4b == baseline_step3

    fixed = runner._render_phase1_prompt(
        "step4a",
        treatment_assignment("step4a", "fixed_hypothetical_survey", percentage=70),
    )
    assert "hypothetical" in fixed.casefold()
    assert "not a verified real-world poll" in fixed
    assert "70%" in fixed

    tiny_fixed = runner._render_phase1_prompt(
        "step4a",
        treatment_assignment(
            "step4a", "fixed_hypothetical_survey", percentage=0.000001
        ),
    )
    assert "0.000001%" in tiny_fixed
    assert "e-" not in tiny_fixed.casefold()

    simulated = runner._render_phase1_prompt(
        "step4a",
        treatment_assignment(
            "step4a", "simulated_persona_consensus", percentage=62.5, consensus_n=3
        ),
    )
    assert "Across 3 other simulated persona conditions" in simulated
    assert "after averaging valid replicates within each persona" in simulated
    assert "62.5%" in simulated
    assert "U.S. population" not in simulated

    placebo = runner._render_phase1_prompt(
        "step4a", treatment_assignment("step4a", "placebo_text")
    )
    assert "no opinion result or percentage is provided" in placebo
    assert "62.5%" not in placebo


@pytest.mark.parametrize(
    ("response", "expected_status", "expected_code"),
    [
        (
            {
                "generated_text": "Maybe",
                "valid": False,
                "error": "missing_candidate_logprobs:22",
            },
            ResultStatus.INVALID,
            "missing_candidate_logprobs:22",
        ),
        (
            {
                "generated_text": "Yes",
                "valid": True,
                "probabilities": {"Yes": 0.6, "No": 0.4},
            },
            ResultStatus.ERROR,
            "phase2_response_schema_error",
        ),
        (
            {
                "generated_text": "Maybe",
                "finish_reason": 7,
                "valid": False,
                "error": "sampled_token_outside_yes_no_candidates",
            },
            ResultStatus.ERROR,
            "phase2_response_schema_error",
        ),
    ],
)
def test_continuation_invalid_scores_and_backend_schema_errors_are_not_imputed(
    response, expected_status, expected_code
):
    backend = FakeBackend(
        chat_results=[[{"generated_text": "analysis"}]],
        continuation_results=[[response]],
    )
    runner = make_runner(backend)

    record = runner._execute_stage_chunk("step1", [baseline_assignment("step1")])[0]

    assert record.status is expected_status
    assert record.error_code == expected_code
    assert record.value["analysis_text"] == "analysis"
    assert LogprobExperimentRunner._binary_decision(record) is None


@pytest.mark.parametrize("stage", ["step2", "step3"])
def test_backend_batch_exception_is_fatal_and_leaves_chunk_retryable(stage):
    backend = FakeBackend(chat_error=RuntimeError("GPU unavailable"))
    runner = make_runner(backend)

    with pytest.raises(RuntimeError, match="chunk was left incomplete") as exc_info:
        runner._execute_stage_chunk(stage, [baseline_assignment(stage)])

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "GPU unavailable" in str(exc_info.value.__cause__)


def test_continuation_batch_exception_is_fatal_and_leaves_chunk_retryable():
    backend = FakeBackend(
        chat_results=[[{"generated_text": "analysis", "finish_reason": "stop"}]],
        continuation_error=RuntimeError("CUDA out of memory"),
    )
    runner = make_runner(backend)

    with pytest.raises(RuntimeError, match="phase-2.*chunk was left incomplete"):
        runner._execute_stage_chunk("step1", [baseline_assignment("step1")])


def test_logprob_pipeline_rejects_api_and_estimates_two_phase_sequences():
    with pytest.raises(ValueError, match="local vLLM"):
        LogprobExperimentRunner("fake/model", use_api=True)

    runner = make_runner(FakeBackend())
    counts = {"step1": 2, "step2": 3, "step3": 4, "step4a": 5, "step4b": 6}
    assert runner._backend_request_estimate(counts) == {
        "step1": 4,
        "step2": 3,
        "step3": 8,
        "step4a": 10,
        "step4b": 12,
    }
    assert runner._extra_manifest_config() == {
        "binary_estimator": "bounded_single_token_candidate_set",
        "scoring_temperature": 0.0,
        "thinking_mode": "disabled_via_chat_template",
        "bounded_scoring_protocol": ("completed_analysis_new_user_fresh_assistant_v1"),
        "bounded_scoring_numeric_tolerance": LOGPROB_NUMERIC_TOLERANCE,
    }


def test_logprob_checkpoint_requires_fresh_turn_scoring_contract():
    runner = make_runner(FakeBackend())
    config = runner._extra_manifest_config()

    runner._apply_extra_manifest_config(config)

    incompatible = dict(config)
    incompatible.pop("bounded_scoring_protocol")
    with pytest.raises(ValueError, match="bounded-scoring protocol"):
        runner._apply_extra_manifest_config(incompatible)

    incompatible = dict(config)
    incompatible["bounded_scoring_protocol"] = "assistant_reasoning_content"
    with pytest.raises(ValueError, match="bounded-scoring protocol"):
        runner._apply_extra_manifest_config(incompatible)

    incompatible = dict(config)
    incompatible.pop("bounded_scoring_numeric_tolerance")
    with pytest.raises(ValueError, match="numeric tolerance"):
        runner._apply_extra_manifest_config(incompatible)

    incompatible = dict(config)
    incompatible["bounded_scoring_numeric_tolerance"] = 1e-9
    with pytest.raises(ValueError, match="numeric tolerance"):
        runner._apply_extra_manifest_config(incompatible)


def test_model_revision_and_remote_code_opt_in_reach_lazy_factory():
    captured = {}
    backend = FakeBackend()

    def factory(**kwargs):
        captured.update(kwargs)
        return backend

    runner = LogprobExperimentRunner(
        "fake/model",
        model_revision="0123456789abcdef",
        trust_remote_code=True,
        llm_factory=factory,
    )
    assert captured == {}

    assert runner.initialize_llm() is backend
    assert captured["use_api"] is False
    assert captured["revision"] == "0123456789abcdef"
    assert captured["tokenizer_revision"] == "0123456789abcdef"
    assert captured["code_revision"] == "0123456789abcdef"
    assert captured["trust_remote_code"] is True
    assert captured["enable_thinking"] is False


def test_parser_reads_real_argv_and_exposes_reproducibility_and_treatments():
    args = build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "deadbeef",
            "--tokenizer-revision",
            "tokenizer-deadbeef",
            "--code-revision",
            "code-deadbeef",
            "--replicates",
            "3",
            "--max-base-units",
            "7",
            "--fixed-percentages",
            "20",
            "80",
            "--no-placebo",
            "--trust-remote-code",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "16",
            "--language-model-only",
            "--resume-from",
            "logprob-run",
            "--resume-step",
            "step4a",
        ]
    )

    assert args.model == "org/model"
    assert args.model_revision == "deadbeef"
    assert args.tokenizer_revision == "tokenizer-deadbeef"
    assert args.code_revision == "code-deadbeef"
    assert args.replicates == 3
    assert args.max_base_units == 7
    assert args.fixed_percentages == [20.0, 80.0]
    assert args.no_placebo is True
    assert args.trust_remote_code is True
    assert args.max_model_len == 4096
    assert args.max_num_seqs == 16
    assert args.language_model_only is True
    assert args.resume_from == "logprob-run"
    assert args.resume_step == "step4a"


def test_dry_run_and_checkpoint_listing_do_not_initialize_backend(
    monkeypatch, tmp_path, capsys
):
    def forbidden_initialize(_self):
        raise AssertionError("dry/list operations must not initialize a model")

    monkeypatch.setattr(LogprobExperimentRunner, "initialize_llm", forbidden_initialize)

    assert (
        main(
            [
                "--model",
                "fake/model",
                "--dry-run",
                "--persona",
                "none",
                "--max-base-units",
                "1",
                "--no-progress",
                "--results-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    dry_payload = capsys.readouterr().out
    assert '"pipeline": "logprob"' in dry_payload
    assert '"total_backend_sequences"' in dry_payload

    assert main(["--list-checkpoints", "--results-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "[]"

    assert (
        main(
            [
                "--resume-from",
                "missing-run",
                "--results-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "manifest not found" in capsys.readouterr().err


def test_tiny_end_to_end_run_uses_fake_backend_and_compiles_explicit_statuses(tmp_path):
    backend = AdaptiveFakeBackend()
    runner = LogprobExperimentRunner(
        "fake/model",
        results_dir=tmp_path,
        fixed_percentages=(50,),
        llm_interface=backend,
        show_progress=False,
    )

    output = runner.run_experiments(personas=["none"], max_base_units=1)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["pipeline"] == "logprob"
    assert payload["manifest"]["config"]["bounded_scoring_protocol"] == (
        "completed_analysis_new_user_fresh_assistant_v1"
    )
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    assert len(row["step1_first_order_belief"]) == 1
    assert row["step2_second_order_belief"][0]["value"] == 55.0
    for field in (
        "step4a_first_order_with_treatment",
        "step4b_action_support_with_treatment",
    ):
        records = row[field]
        assert len(records) == 4  # fixed, unavailable simulated, retest, placebo
        assert [record["status"] for record in records].count("invalid") == 1
        unavailable = next(
            record for record in records if record["status"] == "invalid"
        )
        assert unavailable["error_code"] == "simulated_consensus_unavailable"

    # The unresolved simulated condition is excluded before each Step 4 model call.
    assert backend.chat_batch_sizes[-2:] == [3, 3]
    assert backend.continuation_batch_sizes[-2:] == [3, 3]


def test_main_uses_passed_argv_not_process_argv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner_module.sys,
        "argv",
        ["program", "--definitely-not-a-real-option"],
    )

    assert (
        main(
            [
                "--model",
                "fake/model",
                "--dry-run",
                "--persona",
                "none",
                "--max-base-units",
                "1",
                "--no-progress",
                "--results-dir",
                str(tmp_path),
            ]
        )
        == 0
    )


def test_main_prints_batch_failure_root_cause_and_run_id(monkeypatch, tmp_path, capsys):
    def fail_with_cause(self, **kwargs):
        del kwargs
        self.active_manifest = SimpleNamespace(run_id="test-run-id")
        try:
            raise OSError("GPU worker exited")
        except OSError as cause:
            raise RuntimeError("Step 1 vLLM batch failed") from cause

    monkeypatch.setattr(LogprobExperimentRunner, "run_experiments", fail_with_cause)

    assert (
        main(
            [
                "--model",
                "fake/model",
                "--results-dir",
                str(tmp_path),
                "--no-progress",
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "Root cause: OSError: GPU worker exited" in error
    assert "Checkpoint run ID: test-run-id" in error


def test_dry_run_and_resume_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--dry-run", "--resume-from", "existing-run"])

    assert exc_info.value.code == 2
    assert (
        "--dry-run and --resume-from are mutually exclusive" in capsys.readouterr().err
    )
