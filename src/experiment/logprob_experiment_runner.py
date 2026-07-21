"""Logprob-based political-response experiment runner.

Every binary stage uses a deliberately bounded, two-phase estimator:

1. generate a concise analysis without a final answer; and
2. continue an assistant message for exactly one token while requesting only a
   finite tokenizer-specific set of single-token Yes/No variants.

The resulting Yes/No probabilities are conditional on that finite candidate
set.  ``candidate_mass`` and ``residual_mass`` are persisted with every valid
record so downstream analysis can quantify the approximation.  Step 2 is the
only single-phase stage and is parsed as strict percentage JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.data_loader import instantiate_prompt
from src.experiment.base_runner import (
    Assignment,
    BaseExperimentRunner,
    default_results_directory,
    print_json,
)
from src.experiment.checkpoints import (
    CheckpointValidationError,
    validate_logprob_value,
)
from src.experiment.core import (
    ExperimentRecord,
    ResultStatus,
    canonical_json,
    make_record,
    parse_percentage_response,
)
from src.experiment.planning import (
    TreatmentAssignment,
    format_presented_percentage,
)


_BINARY_STAGES = frozenset({"step1", "step3", "step4a", "step4b"})
_TREATMENT_KINDS = frozenset(
    {
        "fixed_hypothetical_survey",
        "simulated_persona_consensus",
        "no_information_retest",
        "placebo_text",
    }
)
_SCORE_FIELDS = (
    "probabilities",
    "label_logprobs",
    "candidate_mass",
    "residual_mass",
    "candidates",
    "sampled_token_id",
    "sampled_choice",
    "format_valid",
)
_MARKED_BINARY_ANSWER = r"(?:\*\*|__|[`\"'])?(?:yes|no)(?:\*\*|__|[`\"'])?"
_STANDALONE_FINAL_ANSWER_RE = re.compile(
    rf"^\s*{_MARKED_BINARY_ANSWER}\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_ANSWER_LINE_RE = re.compile(
    rf"(?:^|[\r\n])\s*{_MARKED_BINARY_ANSWER}\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_POST_THINK_FINAL_ANSWER_RE = re.compile(
    rf"</think>\s*{_MARKED_BINARY_ANSWER}\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_LEADING_BECAUSE_ANSWER_RE = re.compile(
    rf"^\s*{_MARKED_BINARY_ANSWER}(?:\s*[,;:\-\u2013\u2014])?\s+because\b",
    re.IGNORECASE,
)
_LABELED_FINAL_ANSWER_RE = re.compile(
    rf"\b(?:final\s+answer|answer)\s*(?::|is)\s*"
    rf"{_MARKED_BINARY_ANSWER}\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_CONCLUSION_FINAL_ANSWER_RE = re.compile(
    rf"\b(?:therefore|thus|hence|so)\s*[,;:\-]?\s*"
    rf"(?:the\s+answer\s+is\s+)?{_MARKED_BINARY_ANSWER}\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


class LogprobExperimentRunner(BaseExperimentRunner):
    """Run the local-vLLM, bounded-candidate logprob pipeline."""

    pipeline_name = "logprob"
    prompt_specs = {
        "step1_phase1": (
            "step1_phase1.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL"}),
        ),
        "step1_phase2": ("step1_phase2.txt", frozenset({"ANALYSIS_TEXT"})),
        "step2": (
            "step2.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL"}),
        ),
        "step3_phase1": (
            "step3_phase1.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL", "CORRESPONDING_ACTION"}),
        ),
        "step3_phase2": ("step3_phase2.txt", frozenset({"ANALYSIS_TEXT"})),
        "step4a_phase1": (
            "step4a_phase1.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL", "DISTRIBUTION"}),
        ),
        "step4a_phase2": ("step4a_phase2.txt", frozenset({"ANALYSIS_TEXT"})),
        "step4a_placebo_phase1": (
            "step4a_placebo_phase1.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL"}),
        ),
        "step4b_phase1": (
            "step4b_phase1.txt",
            frozenset(
                {
                    "PERSONA_INJECTION",
                    "POLICY_PROPOSAL",
                    "CORRESPONDING_ACTION",
                    "DISTRIBUTION",
                }
            ),
        ),
        "step4b_phase2": ("step4b_phase2.txt", frozenset({"ANALYSIS_TEXT"})),
        "step4b_placebo_phase1": (
            "step4b_placebo_phase1.txt",
            frozenset({"PERSONA_INJECTION", "POLICY_PROPOSAL", "CORRESPONDING_ACTION"}),
        ),
    }

    def __init__(
        self,
        model_name: str,
        *,
        data_dir: str | Path | None = None,
        prompts_dir: str | Path | None = None,
        results_dir: str | Path | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        seed: int = 42,
        replicates: int = 1,
        fixed_percentages: Sequence[float] = (10.0, 30.0, 50.0, 70.0, 90.0),
        include_simulated_consensus: bool = True,
        include_retest: bool = True,
        include_placebo: bool = True,
        chunk_size: int = 128,
        use_api: bool = False,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        code_revision: str | None = None,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        llm_factory: Any = None,
        llm_interface: Any | None = None,
        show_progress: bool = True,
    ) -> None:
        if use_api:
            raise ValueError(
                "LogprobExperimentRunner requires a local vLLM backend; API mode is not supported."
            )
        base_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "data_dir": data_dir,
            "prompts_dir": prompts_dir,
            "results_dir": results_dir,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "replicates": replicates,
            "fixed_percentages": fixed_percentages,
            "include_simulated_consensus": include_simulated_consensus,
            "include_retest": include_retest,
            "include_placebo": include_placebo,
            "chunk_size": chunk_size,
            "use_api": False,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "code_revision": code_revision,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "llm_interface": llm_interface,
            "show_progress": show_progress,
        }
        if llm_factory is not None:
            base_kwargs["llm_factory"] = llm_factory
        super().__init__(**base_kwargs)
        self._continuation_scoring_preflight_backend: Any | None = None

    def _preflight_continuation_scoring(self) -> Any:
        """Initialize and validate continuation scoring once, before any chat call."""

        backend = self.initialize_llm()
        if self._continuation_scoring_preflight_backend is backend:
            return backend
        preflight = getattr(backend, "preflight_continuation_scoring", None)
        if not callable(preflight):
            raise RuntimeError(
                "logprob backend does not provide preflight_continuation_scoring()"
            )
        preflight()
        self._continuation_scoring_preflight_backend = backend
        return backend

    def cleanup(self) -> None:
        """Release the backend and require a replacement backend to preflight."""

        try:
            super().cleanup()
        finally:
            self._continuation_scoring_preflight_backend = None

    def _extra_manifest_config(self) -> Mapping[str, Any]:
        return {
            "binary_estimator": "bounded_single_token_candidate_set",
            "scoring_temperature": 0.0,
        }

    def _apply_extra_manifest_config(self, config: Mapping[str, Any]) -> None:
        if config.get("binary_estimator") != "bounded_single_token_candidate_set":
            raise ValueError("checkpoint has an incompatible binary estimator")
        if config.get("scoring_temperature") != 0.0:
            raise ValueError("checkpoint has an incompatible scoring temperature")

    def _apply_manifest_config(self, config: Mapping[str, Any]) -> None:
        super()._apply_manifest_config(config)
        if self.use_api:
            raise ValueError("a logprob checkpoint cannot enable API mode")

    @staticmethod
    def _assignment_metadata(assignment: Assignment) -> dict[str, Any]:
        to_dict = getattr(assignment, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("experiment assignments must provide to_dict()")
        payload = to_dict()
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise TypeError("assignment metadata must be a mapping")
        return dict(metadata)

    @staticmethod
    def _error_record(
        assignment: Assignment,
        metadata: Mapping[str, Any],
        code: str,
        message: str,
        *,
        value: Any = None,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            sample_id=assignment.sample_id,
            stage=assignment.stage,
            metadata=dict(metadata),
            status=ResultStatus.ERROR,
            value=value,
            error_code=code,
            error_message=message,
        )

    @staticmethod
    def _invalid_record(
        assignment: Assignment,
        metadata: Mapping[str, Any],
        code: str,
        message: str,
        *,
        value: Any = None,
        raw_response: str | None = None,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            sample_id=assignment.sample_id,
            stage=assignment.stage,
            metadata=dict(metadata),
            status=ResultStatus.INVALID,
            value=value,
            error_code=code,
            error_message=message,
            raw_response=raw_response,
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        if not isinstance(response, Mapping):
            raise TypeError("backend response must be a mapping")
        text = response.get("generated_text")
        if not isinstance(text, str):
            raise TypeError("backend response generated_text must be a string")
        return text

    @staticmethod
    def _phase1_contains_final_answer(text: str) -> bool:
        candidate = text.strip()
        fenced = _JSON_FENCE_RE.fullmatch(candidate)
        if fenced is not None:
            candidate = fenced.group(1).strip()
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(key, str) and key.casefold() == "answer":
                    if isinstance(value, bool):
                        return True
                    if isinstance(value, str) and value.strip().casefold() in {
                        "yes",
                        "no",
                    }:
                        return True
        return any(
            pattern.search(text) is not None
            for pattern in (
                _STANDALONE_FINAL_ANSWER_RE,
                _TERMINAL_ANSWER_LINE_RE,
                _POST_THINK_FINAL_ANSWER_RE,
                _LEADING_BECAUSE_ANSWER_RE,
                _LABELED_FINAL_ANSWER_RE,
                _CONCLUSION_FINAL_ANSWER_RE,
            )
        )

    def _distribution_text(self, assignment: TreatmentAssignment) -> str:
        condition = assignment.condition
        percentage = condition.percentage
        if condition.kind == "fixed_hypothetical_survey":
            if percentage is None:
                raise ValueError("fixed survey treatment has no percentage")
            return (
                "In this hypothetical experimental survey vignette, "
                f"{format_presented_percentage(percentage)}% of respondents answered Yes: "
                "they said the policy would be beneficial to the United States. "
                "This is an assigned hypothetical value, not a verified real-world poll."
            )
        if condition.kind == "simulated_persona_consensus":
            if percentage is None:
                raise ValueError("simulated consensus is unavailable")
            if condition.consensus_n is None:
                raise ValueError(
                    "simulated consensus has no contributing-persona count"
                )
            return (
                f"Across {condition.consensus_n} other simulated persona conditions "
                "(excluding the current persona), after averaging valid replicates "
                f"within each persona, {format_presented_percentage(percentage)}% answered "
                "Yes to the policy-benefit question. This statistic summarizes "
                "simulated model outputs only."
            )
        raise ValueError(f"treatment {condition.kind!r} has no distribution statement")

    def _instantiate(self, template_name: str, **values: Any) -> str:
        """Render one already-validated prompt without re-parsing injected text.

        Base loading has already established the exact placeholder set.  A
        second unresolved-placeholder scan after interpolation would
        incorrectly reject ordinary model/data text such as ``{policy}``.
        """

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

    def _render_phase1_prompt(self, stage: str, assignment: Assignment) -> str:
        metadata = self._assignment_metadata(assignment)
        values = {
            "PERSONA_INJECTION": self.get_persona_prompt(metadata["persona"]),
            "POLICY_PROPOSAL": metadata["proposal"],
        }

        if stage == "step1":
            template_name = "step1_phase1"
        elif stage == "step3":
            template_name = "step3_phase1"
            values["CORRESPONDING_ACTION"] = metadata["action"]
        elif stage in {"step4a", "step4b"}:
            if not isinstance(assignment, TreatmentAssignment):
                raise TypeError(f"{stage} requires TreatmentAssignment values")
            kind = assignment.condition.kind
            if kind not in _TREATMENT_KINDS:
                raise ValueError(f"unsupported treatment kind: {kind!r}")

            if kind == "no_information_retest":
                # A retest deliberately reuses the exact baseline prompt contract.
                template_name = "step1_phase1" if stage == "step4a" else "step3_phase1"
            elif kind == "placebo_text":
                template_name = f"{stage}_placebo_phase1"
            else:
                template_name = f"{stage}_phase1"
                values["DISTRIBUTION"] = self._distribution_text(assignment)
            if stage == "step4b":
                values["CORRESPONDING_ACTION"] = metadata["action"]
        else:
            raise ValueError(f"unsupported binary stage: {stage}")

        return self._instantiate(template_name, **values)

    def _render_phase2_prompt(
        self,
        stage: str,
        assignment: Assignment,
        analysis_text: str,
    ) -> str:
        if not analysis_text.strip():
            raise ValueError("analysis text must not be empty")

        template_stage = stage
        if stage in {"step4a", "step4b"}:
            if not isinstance(assignment, TreatmentAssignment):
                raise TypeError(f"{stage} requires TreatmentAssignment values")
            if assignment.condition.kind == "no_information_retest":
                template_stage = "step1" if stage == "step4a" else "step3"
        return self._instantiate(
            f"{template_stage}_phase2",
            ANALYSIS_TEXT=analysis_text,
        )

    @staticmethod
    def _simulated_consensus_missing(assignment: Assignment) -> bool:
        return (
            isinstance(assignment, TreatmentAssignment)
            and assignment.condition.kind == "simulated_persona_consensus"
            and assignment.condition.percentage is None
        )

    @staticmethod
    def _validate_score_payload(response: Mapping[str, Any]) -> tuple[bool, str | None]:
        """Validate the JSON-native bounded candidate score returned by the backend."""

        valid = response.get("valid")
        if valid is False:
            error = response.get("error")
            sampled_token_id = response.get("sampled_token_id")
            if sampled_token_id is not None and (
                isinstance(sampled_token_id, bool)
                or not isinstance(sampled_token_id, int)
                or sampled_token_id < 0
            ):
                raise TypeError(
                    "invalid continuation sampled_token_id must be a non-negative "
                    "integer or null"
                )
            sampled_choice = response.get("sampled_choice")
            if sampled_choice not in {None, "Yes", "No"}:
                raise ValueError(
                    "invalid continuation sampled_choice must be Yes, No, or null"
                )
            format_valid = response.get("format_valid")
            if format_valid is not None and not isinstance(format_valid, bool):
                raise TypeError(
                    "invalid continuation format_valid must be a boolean or null"
                )
            if (
                format_valid is None
                and sampled_choice is not None
                or format_valid is not None
                and format_valid != (sampled_choice in {"Yes", "No"})
            ):
                raise ValueError(
                    "invalid continuation format_valid must agree with sampled_choice"
                )
            candidates = response.get("candidates")
            if candidates is not None and not isinstance(candidates, list):
                raise TypeError(
                    "invalid continuation candidates must be a list or null"
                )
            # Backend-declared INVALID diagnostics are still persisted. Ensure
            # they cannot defer a non-finite/non-JSON failure to write_chunk(),
            # where it would incorrectly abort the whole run.
            canonical_json({field: response.get(field) for field in _SCORE_FIELDS})
            return False, (
                error if isinstance(error, str) and error else "invalid_logprob_score"
            )
        if valid is not True:
            raise TypeError("continuation response valid must be a boolean")
        if not isinstance(response.get("generated_text"), str):
            raise TypeError(
                "valid continuation response generated_text must be a string"
            )
        finish_reason = response.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TypeError("continuation finish_reason must be a string or null")

        probabilities = response.get("probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != {
            "Yes",
            "No",
        }:
            raise TypeError(
                "valid continuation response must contain Yes/No probabilities"
            )
        probability_values: list[float] = []
        for label in ("Yes", "No"):
            value = probabilities[label]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} probability must be numeric")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(
                    f"{label} probability must be finite and between 0 and 1"
                )
            probability_values.append(number)
        if not math.isclose(sum(probability_values), 1.0, abs_tol=1e-9):
            raise ValueError("Yes/No probabilities must sum to one")

        masses: list[float] = []
        for field in ("candidate_mass", "residual_mass"):
            value = response.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be numeric")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{field} must be finite and between 0 and 1")
            masses.append(number)
        if not math.isclose(sum(masses), 1.0, abs_tol=1e-9):
            raise ValueError("candidate_mass and residual_mass must sum to one")

        label_logprobs = response.get("label_logprobs")
        if not isinstance(label_logprobs, Mapping) or set(label_logprobs) != {
            "Yes",
            "No",
        }:
            raise TypeError(
                "valid continuation response must contain Yes/No label_logprobs"
            )
        for value in label_logprobs.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("label logprobs must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("label logprobs must be finite")

        if not isinstance(response.get("candidates"), list):
            raise TypeError("continuation candidates must be a list")
        sampled_token_id = response.get("sampled_token_id")
        if sampled_token_id is not None and (
            isinstance(sampled_token_id, bool) or not isinstance(sampled_token_id, int)
        ):
            raise TypeError("sampled_token_id must be an integer or null")
        if response.get("sampled_choice") not in {None, "Yes", "No"}:
            raise ValueError("sampled_choice must be Yes, No, or null")
        format_valid = response.get("format_valid")
        if not isinstance(format_valid, bool):
            raise TypeError("format_valid must be a boolean")
        choice_is_binary = response.get("sampled_choice") in {"Yes", "No"}
        if format_valid != choice_is_binary:
            raise ValueError("format_valid must agree with sampled_choice")

        payload = {field: response.get(field) for field in _SCORE_FIELDS}
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if not format_valid:
            return False, "sampled_token_outside_yes_no_candidates"
        return True, None

    @staticmethod
    def _score_value(response: Mapping[str, Any], analysis_text: str) -> dict[str, Any]:
        value = {field: response.get(field) for field in _SCORE_FIELDS}
        value.update(
            {
                # This is visible model-generated text from phase 1, not hidden
                # chain-of-thought or backend-internal reasoning.
                "analysis_text": analysis_text,
                "analysis_text_kind": "model_generated_visible_text",
                "estimator": "bounded_single_token_candidate_set",
                "conditional_on_candidate_set": response.get("probabilities")
                is not None,
                "scoring_temperature": 0.0,
                "finish_reason": response.get("finish_reason"),
            }
        )
        return value

    @classmethod
    def _valid_score_record(
        cls,
        assignment: Assignment,
        metadata: Mapping[str, Any],
        response: Mapping[str, Any],
        analysis_text: str,
    ) -> ExperimentRecord:
        raw_response = response.get("generated_text")
        return ExperimentRecord(
            sample_id=assignment.sample_id,
            stage=assignment.stage,
            metadata=dict(metadata),
            status=ResultStatus.VALID,
            value=cls._score_value(response, analysis_text),
            raw_response=raw_response if isinstance(raw_response, str) else None,
        )

    def _execute_step2_chunk(
        self,
        assignments: Sequence[Assignment],
    ) -> list[ExperimentRecord]:
        records: dict[str, ExperimentRecord] = {}
        executable: list[tuple[Assignment, dict[str, Any], list[dict[str, str]]]] = []

        for assignment in assignments:
            metadata: dict[str, Any] = {}
            try:
                metadata = self._assignment_metadata(assignment)
                prompt = self._instantiate(
                    "step2",
                    PERSONA_INJECTION=self.get_persona_prompt(metadata["persona"]),
                    POLICY_PROPOSAL=metadata["proposal"],
                )
                executable.append(
                    (assignment, metadata, [{"role": "user", "content": prompt}])
                )
            except (
                Exception
            ) as exc:  # an individual malformed assignment must not shift IDs
                records[assignment.sample_id] = self._error_record(
                    assignment, metadata, "prompt_construction_error", str(exc)
                )

        if executable:
            backend = self._preflight_continuation_scoring()
            try:
                responses = backend.chat(
                    dialogue_history=[item[2] for item in executable],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=executable[0][0].seed,
                    show_progress=self.show_progress,
                    desc="Step 2: predicted percentage",
                )
                if not isinstance(responses, Sequence) or isinstance(
                    responses, (str, bytes)
                ):
                    raise TypeError("backend batch response must be a sequence")
                if len(responses) != len(executable):
                    raise RuntimeError(
                        f"backend returned {len(responses)} responses for {len(executable)} prompts"
                    )
            except Exception as exc:
                for assignment, metadata, _ in executable:
                    records[assignment.sample_id] = self._error_record(
                        assignment, metadata, "backend_execution_error", str(exc)
                    )
            else:
                for (assignment, metadata, _), response in zip(
                    executable, responses, strict=True
                ):
                    try:
                        text = self._response_text(response)
                    except (TypeError, ValueError) as exc:
                        records[assignment.sample_id] = self._error_record(
                            assignment,
                            metadata,
                            "backend_response_schema_error",
                            str(exc),
                        )
                        continue
                    parsed = parse_percentage_response(text)
                    records[assignment.sample_id] = make_record(
                        "step2", metadata, parsed, sample_id=assignment.sample_id
                    )

        return [records[assignment.sample_id] for assignment in assignments]

    def _execute_binary_chunk(
        self,
        stage: str,
        assignments: Sequence[Assignment],
    ) -> list[ExperimentRecord]:
        records: dict[str, ExperimentRecord] = {}
        phase1_inputs: list[
            tuple[Assignment, dict[str, Any], str, list[dict[str, str]]]
        ] = []

        for assignment in assignments:
            metadata: dict[str, Any] = {}
            try:
                metadata = self._assignment_metadata(assignment)
                if self._simulated_consensus_missing(assignment):
                    records[assignment.sample_id] = self._invalid_record(
                        assignment,
                        metadata,
                        "simulated_consensus_unavailable",
                        "no valid other-persona responses are available; no model call was made",
                    )
                    continue
                phase1_prompt = self._render_phase1_prompt(stage, assignment)
                dialogue = [{"role": "user", "content": phase1_prompt}]
                phase1_inputs.append((assignment, metadata, phase1_prompt, dialogue))
            except Exception as exc:
                records[assignment.sample_id] = self._error_record(
                    assignment, metadata, "prompt_construction_error", str(exc)
                )

        if not phase1_inputs:
            return [records[assignment.sample_id] for assignment in assignments]

        backend = self._preflight_continuation_scoring()
        try:
            phase1_responses = backend.chat(
                dialogue_history=[item[3] for item in phase1_inputs],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=phase1_inputs[0][0].seed,
                show_progress=self.show_progress,
                desc=f"{stage} phase 1: analysis",
            )
            if not isinstance(phase1_responses, Sequence) or isinstance(
                phase1_responses, (str, bytes)
            ):
                raise TypeError("backend batch response must be a sequence")
            if len(phase1_responses) != len(phase1_inputs):
                raise RuntimeError(
                    f"backend returned {len(phase1_responses)} responses for "
                    f"{len(phase1_inputs)} prompts"
                )
        except Exception as exc:
            for assignment, metadata, _, _ in phase1_inputs:
                records[assignment.sample_id] = self._error_record(
                    assignment, metadata, "phase1_backend_error", str(exc)
                )
            return [records[assignment.sample_id] for assignment in assignments]

        phase2_inputs: list[
            tuple[Assignment, dict[str, Any], str, list[dict[str, str]]]
        ] = []
        for (assignment, metadata, phase1_prompt, _), response in zip(
            phase1_inputs, phase1_responses, strict=True
        ):
            try:
                analysis_text = self._response_text(response)
                finish_reason = response.get("finish_reason")
                if finish_reason is not None and not isinstance(finish_reason, str):
                    raise TypeError("phase 1 finish_reason must be a string or null")
            except (TypeError, ValueError) as exc:
                records[assignment.sample_id] = self._error_record(
                    assignment, metadata, "phase1_response_schema_error", str(exc)
                )
                continue
            if finish_reason is not None and finish_reason.casefold() == "length":
                records[assignment.sample_id] = self._invalid_record(
                    assignment,
                    metadata,
                    "phase1_truncated",
                    "phase 1 reached its token limit; continuation was not attempted",
                    raw_response=analysis_text,
                )
                continue
            if not analysis_text.strip():
                records[assignment.sample_id] = self._invalid_record(
                    assignment,
                    metadata,
                    "empty_analysis",
                    "phase 1 returned no analysis; continuation was not attempted",
                    raw_response=analysis_text,
                )
                continue
            if self._phase1_contains_final_answer(analysis_text):
                records[assignment.sample_id] = self._invalid_record(
                    assignment,
                    metadata,
                    "phase1_contains_final_answer",
                    "phase 1 contained an explicit final Yes/No answer; continuation was not attempted",
                    raw_response=analysis_text,
                )
                continue
            try:
                continuation_prefix = self._render_phase2_prompt(
                    stage, assignment, analysis_text
                )
            except Exception as exc:
                records[assignment.sample_id] = self._error_record(
                    assignment, metadata, "continuation_prompt_error", str(exc)
                )
                continue
            phase2_inputs.append(
                (
                    assignment,
                    metadata,
                    analysis_text,
                    [
                        {"role": "user", "content": phase1_prompt},
                        {"role": "assistant", "content": continuation_prefix},
                    ],
                )
            )

        if phase2_inputs:
            try:
                phase2_responses = backend.chat_with_continuation(
                    dialogue_history=[item[3] for item in phase2_inputs],
                    # Keep the scoring estimand invariant when phase-1 generation
                    # temperature changes. Candidate logprobs are always read from
                    # an untempered continuation distribution.
                    temperature=0.0,
                    max_tokens=1,
                    seed=phase2_inputs[0][0].seed,
                    show_progress=self.show_progress,
                    desc=f"{stage} phase 2: bounded Yes/No candidates",
                )
                if not isinstance(phase2_responses, Sequence) or isinstance(
                    phase2_responses, (str, bytes)
                ):
                    raise TypeError("continuation batch response must be a sequence")
                if len(phase2_responses) != len(phase2_inputs):
                    raise RuntimeError(
                        f"backend returned {len(phase2_responses)} responses for "
                        f"{len(phase2_inputs)} continuations"
                    )
            except Exception as exc:
                for assignment, metadata, analysis_text, _ in phase2_inputs:
                    records[assignment.sample_id] = self._error_record(
                        assignment,
                        metadata,
                        "phase2_backend_error",
                        str(exc),
                        value={
                            "analysis_text": analysis_text,
                            "analysis_text_kind": "model_generated_visible_text",
                        },
                    )
            else:
                for (assignment, metadata, analysis_text, _), response in zip(
                    phase2_inputs, phase2_responses, strict=True
                ):
                    if not isinstance(response, Mapping):
                        records[assignment.sample_id] = self._error_record(
                            assignment,
                            metadata,
                            "phase2_response_schema_error",
                            "continuation response must be a mapping",
                            value={
                                "analysis_text": analysis_text,
                                "analysis_text_kind": "model_generated_visible_text",
                            },
                        )
                        continue
                    try:
                        is_valid, invalid_code = self._validate_score_payload(response)
                    except (TypeError, ValueError, OverflowError) as exc:
                        records[assignment.sample_id] = self._error_record(
                            assignment,
                            metadata,
                            "phase2_response_schema_error",
                            str(exc),
                            value={
                                "analysis_text": analysis_text,
                                "analysis_text_kind": "model_generated_visible_text",
                            },
                        )
                        continue
                    if not is_valid:
                        raw = response.get("generated_text")
                        diagnostic_value = self._score_value(response, analysis_text)
                        records[assignment.sample_id] = self._invalid_record(
                            assignment,
                            metadata,
                            invalid_code or "invalid_logprob_score",
                            "bounded Yes/No candidate score is invalid",
                            value=diagnostic_value,
                            raw_response=raw if isinstance(raw, str) else None,
                        )
                        continue
                    candidate_record = self._valid_score_record(
                        assignment, metadata, response, analysis_text
                    )
                    try:
                        validate_logprob_value(
                            candidate_record.value,
                            sample_id=assignment.sample_id,
                        )
                    except CheckpointValidationError as exc:
                        records[assignment.sample_id] = self._error_record(
                            assignment,
                            metadata,
                            "phase2_response_schema_error",
                            str(exc),
                            value={
                                "analysis_text": analysis_text,
                                "analysis_text_kind": "model_generated_visible_text",
                            },
                        )
                        continue
                    records[assignment.sample_id] = candidate_record

        return [records[assignment.sample_id] for assignment in assignments]

    def _execute_stage_chunk(
        self,
        stage: str,
        assignments: Sequence[Assignment],
    ) -> list[ExperimentRecord]:
        if not assignments:
            return []
        if any(assignment.stage != stage for assignment in assignments):
            raise ValueError("every assignment in a stage chunk must match its stage")
        if stage == "step2":
            return self._execute_step2_chunk(assignments)
        if stage in _BINARY_STAGES:
            return self._execute_binary_chunk(stage, assignments)
        raise ValueError(f"unknown experiment stage: {stage}")

    def _backend_request_estimate(
        self, logical_counts: Mapping[str, int]
    ) -> dict[str, int]:
        """Report generated sequences, counting both phases of binary stages."""

        return {
            stage: count if stage == "step2" else 2 * count
            for stage, count in logical_counts.items()
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _percentage(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("percentage must be between 0 and 100")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and at most 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the real command-line parser without initializing any backend."""

    parser = argparse.ArgumentParser(
        description="Run bounded-candidate logprob political-response experiments."
    )
    parser.add_argument("--model", help="Local model name or path")
    parser.add_argument("--model-revision", help="Pinned model branch, tag, or commit")
    parser.add_argument(
        "--tokenizer-revision",
        help="Pinned tokenizer revision (defaults to --model-revision)",
    )
    parser.add_argument(
        "--code-revision",
        help="Pinned remote-code revision (defaults to --model-revision)",
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Directory containing experiment JSON data"
    )
    parser.add_argument(
        "--prompts-dir", type=Path, help="Directory containing logprob prompts"
    )
    parser.add_argument(
        "--results-dir", type=Path, help="Directory for results/checkpoints"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=_positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=_positive_int, default=1)
    parser.add_argument("--chunk-size", type=_positive_int, default=128)
    parser.add_argument(
        "--max-base-units",
        type=_positive_int,
        help="Deterministically cap action-level units before any inference",
    )
    parser.add_argument(
        "--fixed-percentages",
        type=_percentage,
        nargs="+",
        default=[10.0, 30.0, 50.0, 70.0, 90.0],
        metavar="PERCENT",
        help="Fixed hypothetical survey treatment percentages",
    )
    parser.add_argument(
        "--no-simulated-consensus",
        action="store_true",
        help="Disable leave-one-persona-out simulated-consensus treatments",
    )
    parser.add_argument(
        "--no-retest",
        action="store_true",
        help="Disable no-information retest conditions",
    )
    parser.add_argument(
        "--no-placebo",
        action="store_true",
        help="Disable neutral-text placebo conditions",
    )
    parser.add_argument(
        "--persona",
        "--personas",
        nargs="+",
        action="extend",
        dest="personas",
        help="Persona labels; repeat either option as needed (default: all)",
    )
    parser.add_argument(
        "--category",
        "--categories",
        nargs="+",
        action="extend",
        dest="categories",
        help="Policy categories; repeat either option as needed (default: all)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt in to executing model repository code (disabled by default)",
    )
    parser.add_argument("--dtype", default="auto", help="vLLM model dtype")
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Use vLLM eager mode"
    )
    parser.add_argument("--gpu-memory-utilization", type=_unit_interval, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )
    parser.add_argument(
        "--resume-from",
        metavar="RUN_ID",
        help="Resume a versioned checkpoint run",
    )
    parser.add_argument(
        "--resume-step",
        choices=("step1", "step2", "step3", "step4a", "step4b"),
        default="step1",
        help="First logical stage to execute when resuming",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List logprob checkpoint runs and exit without loading a model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/prompts and print the deterministic plan without loading a model",
    )
    return parser


def _default_results_dir() -> Path:
    return default_results_directory()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process status and parses real ``argv``."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.resume_from:
        parser.error("--dry-run and --resume-from are mutually exclusive")

    results_dir = args.results_dir or _default_results_dir()
    if args.list_checkpoints:
        print_json(LogprobExperimentRunner.list_checkpoint_runs(results_dir))
        return 0
    if not args.model and not args.resume_from:
        parser.error("--model is required for a new run or dry-run")

    runner: LogprobExperimentRunner | None = None
    try:
        runner = LogprobExperimentRunner(
            model_name=args.model or "__restored_from_checkpoint_manifest__",
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            results_dir=results_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            replicates=args.replicates,
            fixed_percentages=args.fixed_percentages,
            include_simulated_consensus=not args.no_simulated_consensus,
            include_retest=not args.no_retest,
            include_placebo=not args.no_placebo,
            chunk_size=args.chunk_size,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            code_revision=args.code_revision,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            show_progress=not args.no_progress,
        )
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
                args.resume_from, args.resume_step
            )
        else:
            output_path = runner.run_experiments(
                personas=args.personas,
                categories=args.categories,
                max_base_units=args.max_base_units,
            )
        print(f"Results written to: {output_path}")
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
