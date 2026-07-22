"""Verbalized-response political belief experiment runner.

The runner delegates deterministic selection, treatment assignment, checkpoint
validation, and compilation to :mod:`src.experiment.base_runner`.  This module
only instantiates verbalize prompts and strictly validates the model's JSON
answers.  Invalid responses remain explicit observations and are never
silently converted to ``Yes``, ``No``, or a default percentage.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Mapping, Sequence

from src.data.data_loader import instantiate_prompt
from src.experiment.base_runner import (
    BaseExperimentRunner,
    default_results_directory,
    print_json,
)
from src.experiment.checkpoints import CheckpointValidationError
from src.experiment.core import (
    ExperimentRecord,
    ResultStatus,
    ValidationResult,
    make_record,
    parse_percentage_response,
    parse_yes_no_response,
)
from src.experiment.planning import (
    StageAssignment,
    TreatmentAssignment,
    format_presented_percentage,
)


class VerbalizeExperimentRunner(BaseExperimentRunner):
    """Run the strict JSON verbalize pipeline."""

    pipeline_name = "verbalize"
    prompt_specs = {
        "step1": (
            "step1.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL"}),
        ),
        "step2": (
            "step2.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL"}),
        ),
        "step3": (
            "step3.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL", "CORRESPONDING_ACTION"}),
        ),
        "step4a": (
            "step4a.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL", "DISTRIBUTION"}),
        ),
        "step4a_placebo": (
            "step4a_placebo.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL"}),
        ),
        "step4b": (
            "step4b.txt",
            frozenset(
                {
                    "PERSONA_INJECTION",
                    "POLICY_PROPOSAL",
                    "CORRESPONDING_ACTION",
                    "DISTRIBUTION",
                }
            ),
        ),
        "step4b_placebo": (
            "step4b_placebo.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL", "CORRESPONDING_ACTION"}),
        ),
    }

    def __init__(
        self,
        model_name: str,
        *,
        use_api: bool = False,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        language_model_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a strict-JSON runner with a backend-specific thinking contract.

        Local chat templates must not emit a reasoning preamble around the JSON
        object, so thinking is always disabled for vLLM. Hosted APIs do not expose
        this chat-template switch and therefore retain the provider-managed mode.
        """

        if "enable_thinking" in kwargs:
            raise TypeError(
                "VerbalizeExperimentRunner fixes enable_thinking from the backend; "
                "callers cannot override it"
            )
        super().__init__(
            model_name=model_name,
            use_api=use_api,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            language_model_only=language_model_only,
            enable_thinking=None if use_api else False,
            **kwargs,
        )

    def _thinking_mode(self) -> str:
        return "provider_managed" if self.use_api else "disabled_via_chat_template"

    def _extra_manifest_config(self) -> Mapping[str, Any]:
        return {"thinking_mode": self._thinking_mode()}

    def _apply_extra_manifest_config(self, config: Mapping[str, Any]) -> None:
        expected_mode = self._thinking_mode()
        if config.get("thinking_mode") != expected_mode:
            raise CheckpointValidationError(
                "checkpoint has an incompatible verbalize thinking-mode contract"
            )
        expected_switch = None if self.use_api else False
        if self.enable_thinking is not expected_switch:
            raise CheckpointValidationError(
                "checkpoint vLLM thinking switch does not match the verbalize contract"
            )

    def initialize_llm(self) -> Any:
        """Initialize and verify the strict local chat-template contract."""

        backend = super().initialize_llm()
        if not self.use_api and getattr(backend, "enable_thinking", None) is not False:
            raise RuntimeError(
                "local verbalize backend must explicitly declare enable_thinking=False"
            )
        return backend

    def _distribution_text(self, assignment: TreatmentAssignment) -> str | None:
        condition = assignment.condition
        if condition.kind == "fixed_hypothetical_survey":
            if condition.percentage is None:
                return None
            percentage = format_presented_percentage(condition.percentage)
            return (
                "In this hypothetical survey vignette, "
                f"{percentage}% of respondents are described as believing this "
                "policy will be beneficial to the United States."
            )
        if condition.kind == "simulated_persona_consensus":
            if condition.percentage is None or condition.consensus_n is None:
                return None
            percentage = format_presented_percentage(condition.percentage)
            return (
                f"Across {condition.consensus_n} other simulated persona conditions "
                "(excluding the current persona), after averaging valid replicates "
                f"within each persona, {percentage}% answered Yes to the "
                "policy-benefit question."
            )
        raise CheckpointValidationError(
            f"treatment {condition.kind!r} does not carry distribution information"
        )

    @staticmethod
    def _assignment_metadata(
        assignment: StageAssignment | TreatmentAssignment,
    ) -> dict[str, Any]:
        return dict(assignment.to_dict()["metadata"])

    def _build_prompt(
        self,
        stage: str,
        assignment: StageAssignment | TreatmentAssignment,
    ) -> str | None:
        metadata = assignment.unit_metadata
        common = {
            "PERSONA_INJECTION": self.get_persona_prompt(metadata["persona"]),
            "POLICY_PROPOSAL": metadata["proposal"],
        }
        if isinstance(assignment, StageAssignment):
            if stage in {"step1", "step2"}:
                return self._instantiate(stage, **common)
            if stage == "step3":
                return self._instantiate(
                    "step3",
                    **common,
                    CORRESPONDING_ACTION=metadata["action"],
                )
            raise CheckpointValidationError(
                f"baseline assignment cannot be executed in stage {stage!r}"
            )

        if not isinstance(assignment, TreatmentAssignment) or stage not in {
            "step4a",
            "step4b",
        }:
            raise CheckpointValidationError(
                f"invalid assignment type for stage {stage!r}"
            )
        kind = assignment.condition.kind
        if kind == "no_information_retest":
            template_name = "step1" if stage == "step4a" else "step3"
            values = dict(common)
            if stage == "step4b":
                values["CORRESPONDING_ACTION"] = metadata["action"]
            return self._instantiate(template_name, **values)
        if kind == "placebo_text":
            template_name = f"{stage}_placebo"
            values = dict(common)
            if stage == "step4b":
                values["CORRESPONDING_ACTION"] = metadata["action"]
            return self._instantiate(template_name, **values)
        if kind not in {
            "fixed_hypothetical_survey",
            "simulated_persona_consensus",
        }:
            raise CheckpointValidationError(f"unknown treatment kind: {kind!r}")
        distribution = self._distribution_text(assignment)
        if distribution is None:
            return None
        values = {**common, "DISTRIBUTION": distribution}
        if stage == "step4b":
            values["CORRESPONDING_ACTION"] = metadata["action"]
        return self._instantiate(stage, **values)

    def _instantiate(self, template_name: str, **values: Any) -> str:
        """Render a template whose exact placeholder contract was prevalidated."""

        try:
            expected = self.prompt_specs[template_name][1]
            template = self.prompt_templates[template_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown or unloaded prompt template: {template_name}"
            ) from exc
        provided = frozenset(values)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise ValueError(
                f"prompt {template_name} values do not match its contract: "
                + "; ".join(details)
            )
        return instantiate_prompt(template, **values)

    @staticmethod
    def _error_record(
        assignment: StageAssignment | TreatmentAssignment,
        metadata: Mapping[str, Any],
        code: str,
        message: str,
    ) -> ExperimentRecord:
        return make_record(
            assignment.stage,
            metadata,
            ValidationResult(
                status=ResultStatus.ERROR,
                error_code=code,
                message=message,
            ),
            sample_id=assignment.sample_id,
        )

    def _execute_stage_chunk(
        self,
        stage: str,
        assignments: Sequence[StageAssignment | TreatmentAssignment],
    ) -> list[ExperimentRecord]:
        """Generate and parse one same-seed stage chunk in assignment order."""

        if not assignments:
            return []
        records: list[ExperimentRecord | None] = [None] * len(assignments)
        pending_indexes: list[int] = []
        dialogues: list[list[dict[str, str]]] = []

        for index, assignment in enumerate(assignments):
            if assignment.stage != stage:
                raise CheckpointValidationError(
                    f"assignment {assignment.sample_id} belongs to {assignment.stage}, not {stage}"
                )
            metadata = self._assignment_metadata(assignment)
            prompt = self._build_prompt(stage, assignment)
            if prompt is None:
                records[index] = make_record(
                    stage,
                    metadata,
                    ValidationResult(
                        status=ResultStatus.INVALID,
                        error_code="simulated_consensus_unavailable",
                        message=(
                            "no valid leave-one-persona-out simulated consensus "
                            "was available; no model call was made"
                        ),
                    ),
                    sample_id=assignment.sample_id,
                )
                continue
            pending_indexes.append(index)
            dialogues.append([{"role": "user", "content": prompt}])

        if pending_indexes:
            seed = assignments[pending_indexes[0]].seed
            if any(assignments[index].seed != seed for index in pending_indexes):
                raise CheckpointValidationError(
                    "a stage chunk must use one generation seed"
                )
            outputs = self._chat_backend(
                dialogues,
                seed=seed,
                max_tokens=self.max_tokens,
                desc=f"{self.pipeline_name} {stage}",
            )
            if len(outputs) != len(pending_indexes):
                raise CheckpointValidationError(
                    f"backend returned {len(outputs)} outputs for {len(pending_indexes)} prompts"
                )
            for index, output in zip(pending_indexes, outputs):
                assignment = assignments[index]
                metadata = self._assignment_metadata(assignment)
                if isinstance(output, BaseException):
                    records[index] = self._error_record(
                        assignment,
                        metadata,
                        "backend_exception",
                        f"{type(output).__name__}: {output}",
                    )
                    continue
                backend_error = output.get("error")
                if backend_error is not None:
                    records[index] = self._error_record(
                        assignment,
                        metadata,
                        "backend_error",
                        str(backend_error),
                    )
                    continue
                generated_text = output.get("generated_text")
                if not isinstance(generated_text, str):
                    records[index] = self._error_record(
                        assignment,
                        metadata,
                        "backend_schema_error",
                        "backend result must contain a string generated_text field",
                    )
                    continue
                validation = (
                    parse_percentage_response(generated_text)
                    if stage == "step2"
                    else parse_yes_no_response(generated_text)
                )
                records[index] = make_record(
                    stage,
                    metadata,
                    validation,
                    sample_id=assignment.sample_id,
                )

        if any(record is None for record in records):
            raise RuntimeError(
                "internal error: a verbalize assignment produced no record"
            )
        return [record for record in records if record is not None]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _percentage(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("must be a finite percentage from 0 to 100")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run political belief experiments using strict verbalized JSON answers."
    )
    parser.add_argument("--model", help="model identifier or local path")
    parser.add_argument("--model-revision", help="optional immutable model revision")
    parser.add_argument(
        "--tokenizer-revision",
        help="optional tokenizer revision (defaults to --model-revision)",
    )
    parser.add_argument(
        "--code-revision",
        help="optional remote-code revision (defaults to --model-revision)",
    )
    parser.add_argument(
        "--persona",
        "--personas",
        nargs="+",
        action="extend",
        dest="personas",
        help="persona labels; repeat either option as needed (default: all)",
    )
    parser.add_argument(
        "--category",
        "--categories",
        nargs="+",
        action="extend",
        dest="categories",
        help="policy categories; repeat either option as needed (default: all)",
    )
    parser.add_argument(
        "--max-base-units",
        "--max-experiments",
        dest="max_base_units",
        type=_positive_int,
        help="cap persona-proposal-action units before any model call",
    )
    parser.add_argument("--replicates", type=_positive_int, default=1)
    parser.add_argument(
        "--fixed-percentages", nargs="+", type=_percentage, default=[10, 30, 50, 70, 90]
    )
    parser.set_defaults(
        include_simulated_consensus=True,
        include_retest=True,
        include_placebo=True,
    )
    parser.add_argument(
        "--simulated-consensus",
        dest="include_simulated_consensus",
        action="store_true",
        help="include leave-one-persona-out simulated consensus (default)",
    )
    parser.add_argument(
        "--no-simulated-consensus",
        dest="include_simulated_consensus",
        action="store_false",
    )
    parser.add_argument("--retest", dest="include_retest", action="store_true")
    parser.add_argument("--no-retest", dest="include_retest", action="store_false")
    parser.add_argument("--placebo", dest="include_placebo", action="store_true")
    parser.add_argument("--no-placebo", dest="include_placebo", action="store_false")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=_positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=_positive_int, default=128)
    parser.add_argument("--data-dir")
    parser.add_argument("--prompts-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument(
        "--use-api", action="store_true", help="use an OpenAI-compatible API"
    )
    parser.add_argument("--api-base-url", help="API endpoint; defaults to OpenRouter")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-max-retries", type=_nonnegative_int, default=2)
    parser.add_argument("--api-max-workers", type=_positive_int, default=4)
    parser.add_argument("--api-retry-base-delay", type=float, default=0.5)
    parser.add_argument("--api-retry-max-delay", type=float, default=8.0)
    parser.add_argument("--api-retry-total-timeout", type=float, default=180.0)

    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--max-model-len",
        type=_positive_int,
        help="Cap vLLM context length; must exceed --max-tokens",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=_positive_int,
        help="Cap the number of sequences scheduled concurrently by vLLM",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Skip multimodal towers for text-only local inference",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="plan and print counts only"
    )
    parser.add_argument("--resume-from", metavar="RUN_ID")
    parser.add_argument(
        "--resume-step",
        choices=("step1", "step2", "step3", "step4a", "step4b"),
        default="step1",
    )
    parser.add_argument("--list-checkpoints", action="store_true")
    return parser


def _runner_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model or "__restored_from_checkpoint_manifest__",
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "code_revision": args.code_revision,
        "data_dir": args.data_dir,
        "prompts_dir": args.prompts_dir,
        "results_dir": args.results_dir,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "replicates": args.replicates,
        "fixed_percentages": args.fixed_percentages,
        "include_simulated_consensus": args.include_simulated_consensus,
        "include_retest": args.include_retest,
        "include_placebo": args.include_placebo,
        "chunk_size": args.chunk_size,
        "use_api": args.use_api,
        "api_base_url": args.api_base_url,
        "api_timeout": args.api_timeout,
        "api_max_retries": args.api_max_retries,
        "api_max_workers": args.api_max_workers,
        "api_retry_base_delay": args.api_retry_base_delay,
        "api_retry_max_delay": args.api_retry_max_delay,
        "api_retry_total_timeout": args.api_retry_total_timeout,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "language_model_only": args.language_model_only,
        "show_progress": not args.no_progress,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process-compatible status code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.resume_from:
        parser.error("--dry-run and --resume-from are mutually exclusive")
    default_results = default_results_directory()
    if args.list_checkpoints:
        print_json(
            VerbalizeExperimentRunner.list_checkpoint_runs(
                args.results_dir or default_results
            )
        )
        return 0
    if not args.model and not args.resume_from:
        parser.error("--model is required for a new run or dry-run")

    runner: VerbalizeExperimentRunner | None = None
    try:
        runner = VerbalizeExperimentRunner(**_runner_kwargs(args))
        if args.dry_run:
            print_json(
                runner.dry_run(
                    personas=args.personas,
                    categories=args.categories,
                    max_base_units=args.max_base_units,
                )
            )
            return 0
        if args.resume_from:
            output_path = runner.run_experiments_from_step(
                args.resume_from,
                args.resume_step,
            )
        else:
            output_path = runner.run_experiments(
                personas=args.personas,
                categories=args.categories,
                max_base_units=args.max_base_units,
            )
        print(f"Results written to {output_path}")
        invalid_count = sum(
            int(values.get("invalid", 0))
            for values in runner.last_status_summary.values()
        )
        if invalid_count:
            print(
                f"Warning: run contains {invalid_count} invalid observations; "
                "see status_summary in the result JSON.",
                file=sys.stderr,
            )
        if runner.has_execution_errors():
            print(
                "Run contains backend ERROR observations and is not fully successful. "
                "Checkpoint shards are immutable; start a new run to retry them.",
                file=sys.stderr,
            )
            return 1
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if runner is not None:
            runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
