import contextlib
import io
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment.checkpoints import CheckpointValidationError
from src.experiment.core import (
    ExperimentRecord,
    ResultStatus,
    canonical_json,
    sha256_text,
)
from src.experiment.planning import (
    StageAssignment,
    TreatmentAssignment,
    TreatmentCondition,
)
from src.experiment.verbalize_experiment_runner import (
    VerbalizeExperimentRunner,
    _runner_kwargs,
    build_parser,
    main,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPOSITORY_ROOT / "src" / "prompts" / "verbalize"


class FakeLLM:
    def __init__(self, *, invalid=False, fail_on=None, enable_thinking=False):
        self.invalid = invalid
        self.fail_on = fail_on
        self.enable_thinking = enable_thinking
        self.calls = []

    def chat(self, dialogue_history, **kwargs):
        self.calls.append((dialogue_history, kwargs))
        if dialogue_history and isinstance(dialogue_history[0], dict):
            dialogues = [dialogue_history]
        else:
            dialogues = dialogue_history
        outputs = []
        for dialogue in dialogues:
            prompt = dialogue[-1]["content"]
            if self.fail_on and self.fail_on in prompt:
                raise RuntimeError("synthetic transport failure")
            if self.invalid:
                generated = "Yes"
            elif "What percentage" in prompt:
                generated = '{"analysis": "estimate", "answer": 62.5}'
            else:
                generated = '{"analysis": "assessment", "answer": "Yes"}'
            outputs.append({"generated_text": generated, "finish_reason": "stop"})
        return outputs

    def close(self):
        return None


def write_fixture_data(directory: Path, *, two_proposals=False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "entities.json").write_text(
        json.dumps({"politicians": ["Alice"], "platforms": ["Example Party"]}),
        encoding="utf-8",
    )

    def proposal(name):
        return {
            "political_proposal": name,
            "actions": [
                {
                    "action_type": "Personal Commitment",
                    "action_description": f"Personally support {name}",
                },
                {
                    "action_type": "Public Advocacy",
                    "action_description": f"Publicly advocate for {name}",
                },
                {
                    "action_type": "Strategic Support",
                    "action_description": f"Strategically support {name}",
                },
            ],
        }

    proposals = [proposal("Good policy")]
    if two_proposals:
        proposals.append(proposal("Bad policy"))
    (directory / "proposal_actions.json").write_text(
        json.dumps({"category": proposals}),
        encoding="utf-8",
    )


def stage_assignment(stage, proposal="Good policy", action=None, sample_id="sample"):
    metadata = {
        "persona": "none",
        "category": "category",
        "proposal": proposal,
    }
    if action is not None:
        metadata.update(
            {
                "action_type": "Personal Commitment",
                "action": action,
            }
        )
    return StageAssignment(
        stage=stage,
        unit_metadata=metadata,
        replicate_id=0,
        seed=123,
        sample_id=sample_id,
    )


def treatment_assignment(
    stage,
    condition,
    *,
    sample_id="treatment",
    action="Support the policy",
):
    metadata = {
        "persona": "none",
        "category": "category",
        "proposal": "Good policy",
    }
    if stage == "step4b":
        metadata.update(
            {
                "action_type": "Personal Commitment",
                "action": action,
            }
        )
    return TreatmentAssignment(
        stage=stage,
        unit_metadata=metadata,
        condition=condition,
        replicate_id=0,
        seed=123,
        order_index=0,
        step2_predicted_percentage=50.0,
        survey_surprise=(
            condition.percentage - 50.0 if condition.percentage is not None else None
        ),
        sample_id=sample_id,
    )


class VerbalizeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.data_dir = self.root / "data"
        self.results_dir = self.root / "results"
        write_fixture_data(self.data_dir)

    def runner(self, llm=None, **kwargs):
        return VerbalizeExperimentRunner(
            model_name=kwargs.pop("model_name", "fake/model"),
            data_dir=self.data_dir,
            prompts_dir=PROMPTS_DIR,
            results_dir=self.results_dir,
            llm_interface=llm,
            show_progress=False,
            **kwargs,
        )

    def test_none_persona_is_empty_and_prompt_schema_is_fail_fast(self):
        runner = self.runner(FakeLLM())
        runner.load_prompt_templates()
        self.assertEqual(runner.get_persona_prompt("none"), "")

        custom_prompts = self.root / "prompts"
        shutil.copytree(PROMPTS_DIR, custom_prompts)
        step1 = custom_prompts / "step1.txt"
        step1.write_text(
            step1.read_text(encoding="utf-8") + "\n{lowercase}\n", encoding="utf-8"
        )
        invalid_runner = VerbalizeExperimentRunner(
            "fake/model",
            data_dir=self.data_dir,
            prompts_dir=custom_prompts,
            results_dir=self.results_dir,
            llm_interface=FakeLLM(),
        )
        with self.assertRaisesRegex(ValueError, "placeholders"):
            invalid_runner.load_prompt_templates()

    def test_distribution_retest_and_placebo_prompts_are_distinct(self):
        runner = self.runner(FakeLLM())
        runner.load_prompt_templates()
        fixed = treatment_assignment(
            "step4a",
            TreatmentCondition(
                kind="fixed_hypothetical_survey",
                source="hypothetical_survey",
                percentage=70.0,
            ),
        )
        simulated = treatment_assignment(
            "step4a",
            TreatmentCondition(
                kind="simulated_persona_consensus",
                source="simulated_persona_consensus_leave_one_out",
                percentage=55.5,
                excluded_persona="none",
                consensus_n=2,
            ),
        )
        retest = treatment_assignment(
            "step4a",
            TreatmentCondition(kind="no_information_retest", source="none"),
        )
        placebo = treatment_assignment(
            "step4a",
            TreatmentCondition(kind="placebo_text", source="neutral_additional_text"),
        )

        fixed_prompt = runner._build_prompt("step4a", fixed)
        simulated_prompt = runner._build_prompt("step4a", simulated)
        retest_prompt = runner._build_prompt("step4a", retest)
        placebo_prompt = runner._build_prompt("step4a", placebo)
        self.assertIn("In this hypothetical survey vignette, 70%", fixed_prompt)
        self.assertIn(
            "Across 2 other simulated persona conditions (excluding the current "
            "persona), after averaging valid replicates within each persona, 55.5%",
            simulated_prompt,
        )
        self.assertNotIn("U.S. population", simulated_prompt)
        self.assertNotIn("Hypothetical Survey Statement", retest_prompt)
        self.assertNotIn("%", retest_prompt)
        self.assertIn("no opinion result or percentage is provided", placebo_prompt)
        self.assertNotEqual(retest_prompt, placebo_prompt)

    def test_strict_invalid_response_and_decimal_step2(self):
        invalid_runner = self.runner(FakeLLM(invalid=True))
        invalid_runner.load_prompt_templates()
        invalid = invalid_runner._execute_stage_chunk(
            "step1",
            [stage_assignment("step1")],
        )[0]
        self.assertEqual(invalid.status, ResultStatus.INVALID)
        self.assertIsNone(invalid.value)

        valid_runner = self.runner(FakeLLM())
        valid_runner.load_prompt_templates()
        prediction = valid_runner._execute_stage_chunk(
            "step2",
            [stage_assignment("step2")],
        )[0]
        self.assertEqual(prediction.status, ResultStatus.VALID)
        self.assertEqual(prediction.value, 62.5)

    def test_api_failure_is_isolated_per_assignment(self):
        fake = FakeLLM(fail_on="Bad policy")
        runner = self.runner(fake, use_api=True, api_max_workers=2)
        runner.load_prompt_templates()
        records = runner._execute_stage_chunk(
            "step1",
            [
                stage_assignment("step1", "Good policy", sample_id="good"),
                stage_assignment("step1", "Bad policy", sample_id="bad"),
            ],
        )
        self.assertEqual([record.sample_id for record in records], ["good", "bad"])
        self.assertEqual(records[0].status, ResultStatus.VALID)
        self.assertEqual(records[1].status, ResultStatus.ERROR)
        self.assertEqual(records[1].error_code, "backend_exception")

    def test_unavailable_simulated_consensus_skips_model_call(self):
        fake = FakeLLM()
        runner = self.runner(fake)
        runner.load_prompt_templates()
        unresolved = treatment_assignment(
            "step4a",
            TreatmentCondition(
                kind="simulated_persona_consensus",
                source="simulated_persona_consensus_leave_one_out",
                excluded_persona="none",
            ),
        )
        record = runner._execute_stage_chunk("step4a", [unresolved])[0]
        self.assertEqual(record.status, ResultStatus.INVALID)
        self.assertEqual(record.error_code, "simulated_consensus_unavailable")
        self.assertEqual(fake.calls, [])

    def test_budget_is_planned_before_calls_and_step4a_is_proposal_level(self):
        factory_calls = []

        def forbidden_factory(**kwargs):
            factory_calls.append(kwargs)
            raise AssertionError("dry-run must not initialize a model")

        runner = VerbalizeExperimentRunner(
            "fake/model",
            data_dir=self.data_dir,
            prompts_dir=PROMPTS_DIR,
            results_dir=self.results_dir,
            llm_factory=forbidden_factory,
            fixed_percentages=[50],
            include_simulated_consensus=False,
            include_retest=False,
            include_placebo=False,
        )
        plan = runner.dry_run(personas=["none"], max_base_units=3)
        self.assertEqual(plan["base_units"], 3)
        self.assertEqual(plan["proposal_units"], 1)
        self.assertEqual(plan["logical_sample_counts"]["step4a"], 1)
        self.assertEqual(plan["logical_sample_counts"]["step4b"], 3)
        self.assertEqual(factory_calls, [])

    def test_backend_thinking_contract_and_resources_reach_lazy_factory(self):
        local_calls = []

        def local_factory(**kwargs):
            local_calls.append(kwargs)
            return FakeLLM()

        local = self.runner(
            None,
            llm_factory=local_factory,
            max_model_len=4096,
            max_num_seqs=16,
            language_model_only=True,
        )
        self.assertFalse(local.enable_thinking)
        self.assertEqual(
            local._extra_manifest_config(),
            {"thinking_mode": "disabled_via_chat_template"},
        )
        local.initialize_llm()
        self.assertEqual(len(local_calls), 1)
        self.assertEqual(local_calls[0]["max_model_len"], 4096)
        self.assertEqual(local_calls[0]["max_num_seqs"], 16)
        self.assertTrue(local_calls[0]["language_model_only"])
        self.assertFalse(local_calls[0]["enable_thinking"])

        api_calls = []

        def api_factory(**kwargs):
            api_calls.append(kwargs)
            return FakeLLM()

        api = self.runner(None, llm_factory=api_factory, use_api=True)
        self.assertIsNone(api.enable_thinking)
        self.assertEqual(
            api._extra_manifest_config(), {"thinking_mode": "provider_managed"}
        )
        api.initialize_llm()
        self.assertEqual(len(api_calls), 1)
        self.assertIsNone(api_calls[0]["enable_thinking"])

        with self.assertRaisesRegex(
            CheckpointValidationError, "thinking-mode contract"
        ):
            local._apply_extra_manifest_config({"thinking_mode": "provider_managed"})
        with self.assertRaisesRegex(ValueError, "apply only to local vLLM"):
            self.runner(FakeLLM(), use_api=True, max_model_len=4096)

    def test_local_backend_must_explicitly_disable_thinking_before_chat(self):
        for marker in (True, None):
            with self.subTest(enable_thinking=marker):
                backend = FakeLLM(enable_thinking=marker)
                runner = self.runner(backend)
                runner.load_prompt_templates()

                with self.assertRaisesRegex(
                    RuntimeError, "explicitly declare enable_thinking=False"
                ):
                    runner._execute_stage_chunk("step1", [stage_assignment("step1")])
                self.assertEqual(backend.calls, [])

    def test_cli_parses_local_vllm_resource_controls(self):
        args = build_parser().parse_args(
            [
                "--model",
                "fake/model",
                "--max-model-len",
                "4096",
                "--max-num-seqs",
                "16",
                "--language-model-only",
            ]
        )
        kwargs = _runner_kwargs(args)

        self.assertEqual(kwargs["max_model_len"], 4096)
        self.assertEqual(kwargs["max_num_seqs"], 16)
        self.assertTrue(kwargs["language_model_only"])

    def test_data_change_after_planning_fails_before_manifest_or_model_call(self):
        fake = FakeLLM()
        runner = self.runner(
            fake,
            fixed_percentages=[50],
            include_simulated_consensus=False,
            include_retest=False,
            include_placebo=False,
        )
        runner.dry_run(personas=["none"], max_base_units=1)

        proposal_path = self.data_dir / "proposal_actions.json"
        proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
        proposals["category"][0]["political_proposal"] = "Changed policy"
        proposal_path.write_text(json.dumps(proposals), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "changed after its validated data snapshot"
        ):
            runner.run_experiments(personas=["none"], max_base_units=1)
        self.assertEqual(fake.calls, [])
        self.assertFalse((self.results_dir / "checkpoints").exists())

    def test_full_run_embeds_manifest_status_summary_and_resume_is_idempotent(self):
        fake = FakeLLM()
        runner = self.runner(
            fake,
            fixed_percentages=[50],
            include_simulated_consensus=False,
            include_retest=False,
            include_placebo=False,
            chunk_size=2,
            max_model_len=4096,
            max_num_seqs=16,
            language_model_only=True,
        )
        output_path = runner.run_experiments(personas=["none"], max_base_units=3)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["manifest"]["config"]["source_tree_sha256"],
            runner._source_tree_sha256(),
        )
        self.assertEqual(
            payload["manifest"]["config"]["thinking_mode"],
            "disabled_via_chat_template",
        )
        self.assertEqual(
            payload["manifest"]["config"]["vllm"],
            {
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 1,
                "dtype": "auto",
                "enforce_eager": False,
                "max_model_len": 4096,
                "max_num_seqs": 16,
                "language_model_only": True,
                "enable_thinking": False,
            },
        )
        self.assertEqual(payload["status_summary"]["step4a"]["expected"], 1)
        self.assertEqual(payload["status_summary"]["step4b"]["expected"], 3)
        self.assertEqual(payload["status_summary"]["step4b"]["valid"], 3)
        self.assertEqual(payload["status_summary"]["step4b"]["missing_rate"], 0)
        self.assertFalse(runner.has_execution_errors())

        def forbidden_factory(**kwargs):
            raise AssertionError("complete checkpoint resume must not call a model")

        resumed = VerbalizeExperimentRunner(
            "placeholder/model",
            data_dir=self.data_dir,
            prompts_dir=PROMPTS_DIR,
            results_dir=self.results_dir,
            llm_factory=forbidden_factory,
        )
        resumed_output = resumed.run_experiments_from_step(
            runner.active_manifest.run_id,
            "step4a",
        )
        self.assertTrue(resumed_output.is_file())
        self.assertFalse(resumed.has_execution_errors())
        self.assertEqual(resumed.max_model_len, 4096)
        self.assertEqual(resumed.max_num_seqs, 16)
        self.assertTrue(resumed.language_model_only)
        self.assertFalse(resumed.enable_thinking)

        step4a_dir = (
            self.results_dir / "checkpoints" / runner.active_manifest.run_id / "step4a"
        )
        chunk_path = next(step4a_dir.glob("chunk_*.json"))
        chunk_payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunk_payload["records"][0]["metadata"]["distribution_percentage"] = 75.0
        # Recompute the unkeyed integrity digest so this test reaches the
        # independent assignment-reconstruction guard. Digest-only tampering
        # is covered by the checkpoint-store tests.
        chunk_payload["records_sha256"] = sha256_text(
            canonical_json(chunk_payload["records"])
        )
        chunk_path.write_text(json.dumps(chunk_payload), encoding="utf-8")

        stale_resume = VerbalizeExperimentRunner(
            "placeholder/model",
            data_dir=self.data_dir,
            prompts_dir=PROMPTS_DIR,
            results_dir=self.results_dir,
            llm_factory=forbidden_factory,
        )
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "metadata does not match the reconstructed assignment",
        ):
            stale_resume.run_experiments_from_step(
                runner.active_manifest.run_id,
                "step4a",
            )

    def test_resume_rejects_an_existing_or_injected_backend(self):
        original = self.runner(
            FakeLLM(),
            model_name="model-B",
            fixed_percentages=[50],
            include_simulated_consensus=False,
            include_retest=False,
            include_placebo=False,
        )
        original.run_experiments(personas=["none"], max_base_units=1)

        wrong_backend = FakeLLM()
        resumed = self.runner(wrong_backend, model_name="model-A")
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "refuses to reuse an existing or injected backend",
        ):
            resumed.run_experiments_from_step(
                original.active_manifest.run_id,
                "step4b",
            )
        self.assertEqual(wrong_backend.calls, [])

    def test_logprob_binary_decision_uses_sampled_choice_not_probability_argmax(self):
        metadata = {"persona": "none", "category": "category", "proposal": "P"}
        sampled_no = ExperimentRecord(
            sample_id="sampled-no",
            stage="step1",
            metadata=metadata,
            status=ResultStatus.VALID,
            value={
                "probabilities": {"Yes": 0.9, "No": 0.1},
                "sampled_choice": "No",
                "format_valid": True,
            },
        )
        tied_but_sampled_yes = ExperimentRecord(
            sample_id="sampled-yes",
            stage="step1",
            metadata=metadata,
            status=ResultStatus.VALID,
            value={
                "probabilities": {"Yes": 0.5, "No": 0.5},
                "sampled_choice": "Yes",
                "format_valid": True,
            },
        )
        invalid_format = ExperimentRecord(
            sample_id="invalid-format",
            stage="step1",
            metadata=metadata,
            status=ResultStatus.VALID,
            value={
                "probabilities": {"Yes": 0.9, "No": 0.1},
                "sampled_choice": "Yes",
                "format_valid": False,
            },
        )

        self.assertEqual(VerbalizeExperimentRunner._binary_decision(sampled_no), 0)
        self.assertEqual(
            VerbalizeExperimentRunner._binary_decision(tied_but_sampled_yes), 1
        )
        self.assertIsNone(VerbalizeExperimentRunner._binary_decision(invalid_format))

    def test_vllm_mode_ignores_unrelated_bad_openrouter_environment_and_accepts_local_path(
        self,
    ):
        with patch.dict(os.environ, {"OPENROUTER_BASE_URL": "not-a-url"}):
            runner = self.runner(FakeLLM(), model_name="/models/local-checkpoint")
        self.assertIsNone(runner.api_base_url)
        self.assertEqual(runner.model_name, "/models/local-checkpoint")

    def test_api_base_url_is_loaded_from_cwd_dotenv_before_manifest_resolution(self):
        (self.root / ".env").write_text(
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/test-v1\n",
            encoding="utf-8",
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(Path, "cwd", return_value=self.root),
        ):
            runner = self.runner(FakeLLM(), use_api=True)
        self.assertEqual(runner.api_base_url, "https://openrouter.ai/api/test-v1")

    def test_remote_code_requires_and_inherits_pinned_revisions(self):
        with self.assertRaisesRegex(ValueError, "requires code_revision"):
            self.runner(FakeLLM(), trust_remote_code=True)

        runner = self.runner(
            FakeLLM(),
            model_revision="model-commit",
            trust_remote_code=True,
        )
        self.assertEqual(runner.tokenizer_revision, "model-commit")
        self.assertEqual(runner.code_revision, "model-commit")

        with self.assertRaisesRegex(ValueError, "only to local vLLM"):
            self.runner(
                FakeLLM(),
                use_api=True,
                model_revision="not-an-api-revision",
            )

    def test_runner_rejects_nonfinite_api_delays_and_boolean_gpu_fraction(self):
        for kwargs in (
            {"api_retry_base_delay": math.nan},
            {"api_retry_max_delay": math.inf},
            {"gpu_memory_utilization": True},
            {"dtype": " "},
            {"enforce_eager": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    self.runner(FakeLLM(), **kwargs)

    def test_api_rejects_nondefault_local_vllm_controls(self):
        for kwargs in (
            {"gpu_memory_utilization": 0.5},
            {"tensor_parallel_size": 2},
            {"dtype": "half"},
            {"enforce_eager": True},
            {"max_model_len": 4096},
            {"max_num_seqs": 16},
            {"language_model_only": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "only to local vLLM"):
                    self.runner(FakeLLM(), use_api=True, **kwargs)

    def test_cli_dry_run_uses_real_argv_without_initializing_model(self):
        stdout = io.StringIO()
        with patch.object(
            VerbalizeExperimentRunner,
            "initialize_llm",
            side_effect=AssertionError("dry-run initialized a model"),
        ):
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "--model",
                        "fake/model",
                        "--data-dir",
                        str(self.data_dir),
                        "--prompts-dir",
                        str(PROMPTS_DIR),
                        "--results-dir",
                        str(self.results_dir),
                        "--personas",
                        "none",
                        "--max-base-units",
                        "1",
                        "--max-model-len",
                        "4096",
                        "--max-num-seqs",
                        "16",
                        "--language-model-only",
                        "--dry-run",
                    ]
                )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["base_units"], 1)

    def test_cli_resume_does_not_require_repeating_model(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(
                [
                    "--resume-from",
                    "missing-run",
                    "--results-dir",
                    str(self.results_dir),
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("manifest not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
