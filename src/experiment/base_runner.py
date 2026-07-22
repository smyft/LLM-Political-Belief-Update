"""Shared orchestration for political-response experiments.

The base runner owns deterministic planning, lazy model initialization,
versioned checkpoints, strict sample-ID alignment, and linear compilation.
Concrete runners only build prompts and translate backend responses into
``ExperimentRecord`` values.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from dotenv import load_dotenv

from src.data.data_loader import (
    NAMED_PLACEHOLDER_PATTERN,
    DataLoader,
    default_data_directory,
    load_prompt_template,
)
from src.experiment.checkpoints import (
    CheckpointStore,
    CheckpointValidationError,
    RunManifest,
    atomic_write_json,
    validate_record_ids,
)
from src.experiment.compiler import compile_grouped_results
from src.experiment.core import (
    ExperimentRecord,
    ResultStatus,
    canonical_json,
    hash_files,
    hash_mapping,
    hash_templates,
)
from src.experiment.planning import (
    FIXED_DISTRIBUTION_PERCENTAGES,
    ActionUnit,
    SelectionPlan,
    StageAssignment,
    TreatmentPlan,
    build_baseline_assignments,
    build_selection_plan,
    build_treatment_plan,
)
from src.models.unified_llm_interface import UnifiedLLMInterface


class Assignment(Protocol):
    sample_id: str
    stage: str
    seed: int

    def to_dict(self) -> Mapping[str, Any]:
        """Return the assignment's JSON-serializable representation."""

        ...


PromptSpec = tuple[str, frozenset[str]]


def default_results_directory() -> Path:
    """Return a writable, installation-independent default output directory."""

    return Path.cwd() / "results"


class BaseExperimentRunner:
    """Backend-independent experiment planning and checkpoint lifecycle."""

    pipeline_name = "base"
    prompt_specs: Mapping[str, PromptSpec] = {}

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
        fixed_percentages: Sequence[float] = FIXED_DISTRIBUTION_PERCENTAGES,
        include_simulated_consensus: bool = True,
        include_retest: bool = True,
        include_placebo: bool = True,
        chunk_size: int = 128,
        use_api: bool = False,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        code_revision: str | None = None,
        api_base_url: str | None = None,
        api_timeout: float = 60.0,
        api_max_retries: int = 2,
        api_max_workers: int = 4,
        api_retry_base_delay: float = 0.5,
        api_retry_max_delay: float = 8.0,
        api_retry_total_timeout: float = 180.0,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        enforce_eager: bool = False,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        language_model_only: bool = False,
        enable_thinking: bool | None = None,
        llm_factory: Callable[..., Any] = UnifiedLLMInterface,
        llm_interface: Any | None = None,
        show_progress: bool = True,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or replicates < 1
        ):
            raise ValueError("replicates must be a positive integer")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("temperature must be numeric")
        if not math.isfinite(float(temperature)) or float(temperature) < 0:
            raise ValueError("temperature must be finite and non-negative")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision.strip()
        ):
            raise ValueError("model_revision must be a non-empty string when provided")
        for name, revision_value in (
            ("tokenizer_revision", tokenizer_revision),
            ("code_revision", code_revision),
        ):
            if revision_value is not None and (
                not isinstance(revision_value, str) or not revision_value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if use_api and any(
            revision_value is not None
            for revision_value in (
                model_revision,
                tokenizer_revision,
                code_revision,
            )
        ):
            raise ValueError(
                "model/tokenizer/code revision flags apply only to local vLLM; "
                "hosted API versions must be selected in the model identifier"
            )
        for name, numeric_value in (
            ("api_timeout", api_timeout),
            ("api_retry_base_delay", api_retry_base_delay),
            ("api_retry_max_delay", api_retry_max_delay),
            ("api_retry_total_timeout", api_retry_total_timeout),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(numeric_value)):
                raise ValueError(f"{name} must be finite")
        if api_timeout <= 0:
            raise ValueError("api_timeout must be positive")
        if api_retry_total_timeout <= 0:
            raise ValueError("api_retry_total_timeout must be positive")
        if (
            isinstance(api_max_retries, bool)
            or not isinstance(api_max_retries, int)
            or api_max_retries < 0
        ):
            raise ValueError("api_max_retries must be a non-negative integer")
        if (
            isinstance(api_max_workers, bool)
            or not isinstance(api_max_workers, int)
            or api_max_workers < 1
        ):
            raise ValueError("api_max_workers must be a positive integer")
        if api_retry_base_delay < 0 or api_retry_max_delay < api_retry_base_delay:
            raise ValueError("API retry delays are invalid")
        resolved_api_url: str | None = None
        if use_api:
            # Resolve public endpoint configuration before freezing it into the
            # manifest.  The API client also loads dotenv for the secret, but
            # doing that only during lazy initialization would make a
            # repository-root OPENROUTER_BASE_URL silently ineffective here.
            dotenv_paths = (Path.cwd() / ".env", project_root / ".env")
            for dotenv_path in dict.fromkeys(dotenv_paths):
                if dotenv_path.is_file():
                    load_dotenv(dotenv_path=dotenv_path, override=False)
            resolved_api_url = (
                api_base_url
                or os.getenv("OPENROUTER_BASE_URL")
                or "https://openrouter.ai/api/v1"
            ).rstrip("/")
            parsed_api_url = urlsplit(resolved_api_url)
            if (
                parsed_api_url.scheme not in {"http", "https"}
                or not parsed_api_url.hostname
                or parsed_api_url.username
                or parsed_api_url.password
                or parsed_api_url.query
                or parsed_api_url.fragment
            ):
                raise ValueError(
                    "api_base_url must be an HTTP(S) endpoint without credentials, "
                    "query, or fragment"
                )
            if parsed_api_url.scheme != "https" and parsed_api_url.hostname not in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise ValueError(
                    "api_base_url must use HTTPS except for a local loopback endpoint"
                )
        if isinstance(gpu_memory_utilization, bool) or not isinstance(
            gpu_memory_utilization, (int, float)
        ):
            raise TypeError("gpu_memory_utilization must be numeric")
        if not math.isfinite(float(gpu_memory_utilization)) or not (
            0 < gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if (
            isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size < 1
        ):
            raise ValueError("tensor_parallel_size must be a positive integer")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("dtype must be a non-empty string")
        if not isinstance(enforce_eager, bool):
            raise TypeError("enforce_eager must be a boolean")
        for name, integer_value in (
            ("max_model_len", max_model_len),
            ("max_num_seqs", max_num_seqs),
        ):
            if integer_value is not None and (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < 1
            ):
                raise ValueError(f"{name} must be a positive integer when provided")
        if max_model_len is not None and max_model_len <= max_tokens:
            raise ValueError(
                "max_model_len must exceed max_tokens so the prompt has context space"
            )
        if not isinstance(language_model_only, bool):
            raise TypeError("language_model_only must be a boolean")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError("enable_thinking must be a boolean or None")
        if use_api and (
            float(gpu_memory_utilization) != 0.9
            or tensor_parallel_size != 1
            or dtype.strip() != "auto"
            or enforce_eager
            or max_model_len is not None
            or max_num_seqs is not None
            or language_model_only
            or enable_thinking is not None
        ):
            raise ValueError(
                "gpu_memory_utilization, tensor_parallel_size, dtype, enforce_eager, "
                "max_model_len, max_num_seqs, language_model_only, and "
                "enable_thinking apply only to local vLLM when non-default"
            )

        self.model_name = model_name.strip()
        self.data_dir = (
            Path(data_dir) if data_dir is not None else default_data_directory()
        )
        self.prompts_dir = (
            Path(prompts_dir)
            if prompts_dir is not None
            else project_root / "src" / "prompts" / self.pipeline_name
        )
        self.results_dir = (
            Path(results_dir)
            if results_dir is not None
            else default_results_directory()
        )
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.seed = seed
        self.replicates = replicates
        self.fixed_percentages = tuple(float(value) for value in fixed_percentages)
        self.include_simulated_consensus = include_simulated_consensus
        self.include_retest = include_retest
        self.include_placebo = include_placebo
        self.chunk_size = chunk_size
        self.use_api = use_api
        self.trust_remote_code = trust_remote_code
        self.model_revision = model_revision.strip() if model_revision else None
        self.tokenizer_revision = (
            tokenizer_revision.strip() if tokenizer_revision else self.model_revision
        )
        self.code_revision = (
            code_revision.strip() if code_revision else self.model_revision
        )
        if self.use_api and self.trust_remote_code:
            raise ValueError(
                "trust_remote_code is only valid for the local vLLM backend"
            )
        if not self.use_api and self.trust_remote_code and self.code_revision is None:
            raise ValueError(
                "trust_remote_code requires code_revision or model_revision to pin "
                "the executed remote code"
            )
        self.api_base_url = resolved_api_url
        self.api_timeout = float(api_timeout)
        self.api_max_retries = api_max_retries
        self.api_max_workers = api_max_workers
        self.api_retry_base_delay = float(api_retry_base_delay)
        self.api_retry_max_delay = float(api_retry_max_delay)
        self.api_retry_total_timeout = float(api_retry_total_timeout)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype.strip()
        self.enforce_eager = enforce_eager
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.language_model_only = language_model_only
        self.enable_thinking = enable_thinking
        self.show_progress = show_progress

        self.data_loader = DataLoader(data_dir=str(self.data_dir))
        self.prompt_templates: dict[str, str] = {}
        self._prompt_hashes: dict[str, str] = {}
        self._llm_factory = llm_factory
        self.llm_interface = llm_interface
        self.results: list[dict[str, Any]] = []
        self.last_status_summary: dict[str, dict[str, int | float]] = {}
        self.active_manifest: RunManifest | None = None

    @property
    def checkpoint_root(self) -> Path:
        return self.results_dir / "checkpoints"

    def load_prompt_templates(self) -> Mapping[str, str]:
        """Load every required prompt and validate its placeholder schema."""

        if not self.prompt_specs:
            raise NotImplementedError("concrete runner must declare prompt_specs")
        loaded: dict[str, str] = {}
        problems: list[str] = []
        for name, (filename, expected_placeholders) in self.prompt_specs.items():
            path = self.prompts_dir / filename
            if not path.is_file():
                problems.append(f"missing prompt {name}: {path}")
                continue
            try:
                template = load_prompt_template(str(path))
            except (OSError, ValueError) as exc:
                problems.append(f"invalid prompt {name}: {exc}")
                continue
            actual = frozenset(NAMED_PLACEHOLDER_PATTERN.findall(template))
            if actual != expected_placeholders:
                problems.append(
                    f"prompt {name} placeholders are {sorted(actual)}, "
                    f"expected {sorted(expected_placeholders)}"
                )
                continue
            loaded[name] = template
        if problems:
            raise ValueError("Prompt validation failed:\n- " + "\n- ".join(problems))
        self.prompt_templates = loaded
        self._prompt_hashes = hash_templates(loaded)
        return loaded

    def initialize_llm(self) -> Any:
        """Initialize the backend lazily; planning/help/dry-run never call this."""

        if self.llm_interface is None:
            self.llm_interface = self._llm_factory(
                model_name=self.model_name,
                use_api=self.use_api,
                trust_remote_code=self.trust_remote_code,
                revision=self.model_revision,
                tokenizer_revision=self.tokenizer_revision,
                code_revision=self.code_revision,
                base_url=self.api_base_url,
                timeout=self.api_timeout,
                max_retries=self.api_max_retries,
                max_workers=self.api_max_workers,
                retry_base_delay=self.api_retry_base_delay,
                retry_max_delay=self.api_retry_max_delay,
                retry_total_timeout=self.api_retry_total_timeout,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
                enforce_eager=self.enforce_eager,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                language_model_only=self.language_model_only,
                enable_thinking=self.enable_thinking,
            )
        return self.llm_interface

    def _chat_backend(
        self,
        dialogues: Sequence[list[dict[str, str]]],
        *,
        seed: int,
        max_tokens: int,
        desc: str,
        **kwargs: Any,
    ) -> list[Mapping[str, Any] | BaseException]:
        """Call chat with per-assignment API error isolation.

        The API facade's batch call raises when any future fails, which would
        discard already-paid successful siblings. Calling single dialogues in
        a bounded runner-level pool preserves one output/error per sample. The
        local vLLM path remains a true batch and fails atomically.
        """

        if not dialogues:
            return []
        llm = self.initialize_llm()
        call_kwargs = {
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "show_progress": False if self.use_api else self.show_progress,
            "desc": desc,
            **kwargs,
        }
        if not self.use_api:
            outputs = llm.chat(dialogue_history=list(dialogues), **call_kwargs)
            if not isinstance(outputs, list) or len(outputs) != len(dialogues):
                raise CheckpointValidationError(
                    f"backend returned {len(outputs) if isinstance(outputs, list) else 'non-list'} "
                    f"results for {len(dialogues)} dialogues"
                )
            return outputs

        def call_one(
            dialogue: list[dict[str, str]],
        ) -> Mapping[str, Any] | BaseException:
            try:
                output = llm.chat(dialogue_history=dialogue, **call_kwargs)
                if not isinstance(output, list) or len(output) != 1:
                    raise ValueError("single-dialogue API call must return one result")
                if not isinstance(output[0], Mapping):
                    raise TypeError("API result must be an object")
                return output[0]
            except Exception as exc:  # converted into a per-sample ERROR record
                return exc

        with ThreadPoolExecutor(
            max_workers=min(self.api_max_workers, len(dialogues))
        ) as pool:
            return list(pool.map(call_one, dialogues))

    def cleanup(self) -> None:
        if self.llm_interface is None:
            return
        close = getattr(self.llm_interface, "close", None)
        if callable(close):
            close()
        else:
            free_memory = getattr(self.llm_interface, "free_memory", None)
            if callable(free_memory):
                free_memory()
        self.llm_interface = None

    def get_persona_prompt(self, persona: str) -> str:
        """Return persona context; the ``none`` control is intentionally empty."""

        if persona == "none":
            return ""
        entities = self.data_loader.load_entities()
        politicians = set(entities["politicians"])
        platforms = set(entities["platforms"])
        if persona in platforms:
            return (
                f"Act as a typical voter who supports {persona}. Answer based on "
                f"the political views and values of a typical {persona} supporter."
            )
        if persona in politicians:
            return f"Act as {persona}. Answer based on the political views and values of {persona}."
        raise ValueError(f"unknown persona: {persona}")

    def _select_action_units(
        self,
        *,
        personas: Sequence[str] | None,
        categories: Sequence[str] | None,
        unique_proposals: Sequence[tuple[str, str]] | None,
    ) -> list[ActionUnit]:
        available_personas = self.data_loader.get_personas(include_none=True)
        selected_personas = list(available_personas if personas is None else personas)
        if not selected_personas:
            raise ValueError("at least one persona must be selected")
        unknown_personas = sorted(set(selected_personas) - set(available_personas))
        if unknown_personas:
            raise ValueError(f"unknown persona: {unknown_personas[0]}")
        if len(selected_personas) != len(set(selected_personas)):
            raise ValueError("personas must not contain duplicates")

        available_categories = set(self.data_loader.get_categories())
        selected_categories = (
            available_categories if categories is None else set(categories)
        )
        unknown_categories = sorted(selected_categories - available_categories)
        if unknown_categories:
            raise ValueError(f"unknown category: {unknown_categories[0]}")

        available_proposals = set(self.data_loader.get_unique_proposals())
        selected_proposals = (
            available_proposals if unique_proposals is None else set(unique_proposals)
        )
        unknown_proposals = sorted(selected_proposals - available_proposals)
        if unknown_proposals:
            raise ValueError(f"unknown proposal: {unknown_proposals[0]!r}")

        units: list[ActionUnit] = []
        for persona in selected_personas:
            for (
                category,
                proposal,
                action_type,
                action,
            ) in self.data_loader.get_proposal_action_pairs():
                if (
                    category not in selected_categories
                    or (category, proposal) not in selected_proposals
                ):
                    continue
                units.append(
                    ActionUnit(
                        persona=persona,
                        category=category,
                        proposal=proposal,
                        action_type=action_type,
                        action=action,
                    )
                )
        if not units:
            raise ValueError("selection produced no experiment base units")
        return units

    def _build_plans(
        self,
        *,
        personas: Sequence[str] | None,
        categories: Sequence[str] | None,
        unique_proposals: Sequence[tuple[str, str]] | None,
        max_base_units: int | None,
    ) -> tuple[
        SelectionPlan,
        dict[str, tuple[StageAssignment, ...]],
        TreatmentPlan,
    ]:
        selection = build_selection_plan(
            self._select_action_units(
                personas=personas,
                categories=categories,
                unique_proposals=unique_proposals,
            ),
            max_base_units=max_base_units,
            selection_seed=self.seed,
        )
        baseline = build_baseline_assignments(
            selection,
            replicates=self.replicates,
            master_seed=self.seed,
        )
        # This unresolved skeleton fixes all IDs/order before the first call.
        treatment_skeleton = build_treatment_plan(
            selection,
            valid_binary_beliefs=None,
            step2_predictions={},
            fixed_percentages=self.fixed_percentages,
            include_simulated_consensus=self.include_simulated_consensus,
            include_retest=self.include_retest,
            include_placebo=self.include_placebo,
            replicates=self.replicates,
            master_seed=self.seed,
        )
        return selection, baseline, treatment_skeleton

    def _extra_manifest_config(self) -> Mapping[str, Any]:
        return {}

    def _apply_extra_manifest_config(self, config: Mapping[str, Any]) -> None:
        del config

    def _manifest_config(self, max_base_units: int | None) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline_name,
            "model_name": self.model_name,
            "use_api": self.use_api,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "replicates": self.replicates,
            "fixed_percentages": list(self.fixed_percentages),
            "include_simulated_consensus": self.include_simulated_consensus,
            "include_retest": self.include_retest,
            "include_placebo": self.include_placebo,
            "trust_remote_code": self.trust_remote_code,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "code_revision": self.code_revision,
            "source_tree_sha256": self._source_tree_sha256(),
            "backend": "api" if self.use_api else "vllm",
            "api": {
                "base_url": self.api_base_url if self.use_api else None,
                "timeout": self.api_timeout,
                "max_retries": self.api_max_retries,
                "max_workers": self.api_max_workers,
                "retry_base_delay": self.api_retry_base_delay,
                "retry_max_delay": self.api_retry_max_delay,
                "retry_total_timeout": self.api_retry_total_timeout,
            },
            "vllm": {
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "tensor_parallel_size": self.tensor_parallel_size,
                "dtype": self.dtype,
                "enforce_eager": self.enforce_eager,
                "max_model_len": self.max_model_len,
                "max_num_seqs": self.max_num_seqs,
                "language_model_only": self.language_model_only,
                "enable_thinking": self.enable_thinking,
            },
            "max_base_units": max_base_units,
            **dict(self._extra_manifest_config()),
        }

    def _apply_manifest_config(self, config: Mapping[str, Any]) -> None:
        if config.get("pipeline") != self.pipeline_name:
            raise CheckpointValidationError(
                f"checkpoint pipeline is {config.get('pipeline')!r}, expected {self.pipeline_name!r}"
            )
        self.model_name = config["model_name"]
        self.use_api = config["use_api"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.seed = config["seed"]
        self.replicates = config["replicates"]
        self.fixed_percentages = tuple(config["fixed_percentages"])
        self.include_simulated_consensus = config["include_simulated_consensus"]
        self.include_retest = config["include_retest"]
        self.include_placebo = config["include_placebo"]
        self.trust_remote_code = config["trust_remote_code"]
        self.model_revision = config.get("model_revision")
        self.tokenizer_revision = config.get("tokenizer_revision", self.model_revision)
        self.code_revision = config.get("code_revision", self.model_revision)
        if self.use_api and any(
            revision_value is not None
            for revision_value in (
                self.model_revision,
                self.tokenizer_revision,
                self.code_revision,
            )
        ):
            raise CheckpointValidationError(
                "API checkpoint cannot contain local model revision settings"
            )
        if self.use_api and self.trust_remote_code:
            raise CheckpointValidationError(
                "API checkpoint cannot enable trust_remote_code"
            )
        if not self.use_api and self.trust_remote_code and self.code_revision is None:
            raise CheckpointValidationError(
                "checkpoint enables trust_remote_code without a pinned code revision"
            )
        api_config = config["api"]
        self.api_base_url = api_config.get("base_url") if self.use_api else None
        self.api_timeout = api_config["timeout"]
        self.api_max_retries = api_config["max_retries"]
        self.api_max_workers = api_config["max_workers"]
        self.api_retry_base_delay = api_config["retry_base_delay"]
        self.api_retry_max_delay = api_config["retry_max_delay"]
        self.api_retry_total_timeout = api_config["retry_total_timeout"]
        vllm_config = config["vllm"]
        self.gpu_memory_utilization = vllm_config["gpu_memory_utilization"]
        self.tensor_parallel_size = vllm_config["tensor_parallel_size"]
        self.dtype = vllm_config["dtype"]
        self.enforce_eager = vllm_config["enforce_eager"]
        self.max_model_len = vllm_config.get("max_model_len")
        self.max_num_seqs = vllm_config.get("max_num_seqs")
        self.language_model_only = vllm_config.get("language_model_only", False)
        self.enable_thinking = vllm_config.get("enable_thinking")
        if self.max_model_len is not None and self.max_model_len <= self.max_tokens:
            raise CheckpointValidationError(
                "checkpoint max_model_len must exceed max_tokens"
            )
        self._apply_extra_manifest_config(config)

    def _data_hashes(self) -> dict[str, str]:
        # Hash the same validated byte snapshots that produced the in-memory
        # planning objects. Re-hashing only the current files could falsely bind
        # a manifest to newer data while execution still uses an older cache.
        return self.data_loader.snapshot_hashes(verify_files=True)

    @staticmethod
    def _source_tree_sha256() -> str:
        """Fingerprint Python sources by stable repository-relative names."""

        project_root = Path(__file__).resolve().parents[2]
        source_root = project_root / "src"
        source_files = {
            path.relative_to(project_root).as_posix(): path
            for path in sorted(source_root.rglob("*.py"))
            if path.is_file()
        }
        return hash_mapping(hash_files(source_files))

    @staticmethod
    def _new_run_id(pipeline_name: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"{pipeline_name}-{timestamp}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _detect_code_version(project_root: Path) -> str | None:
        """Best-effort Git commit plus dirty marker, without persisting paths."""

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return f"{commit}-dirty" if dirty else commit

    def _create_manifest(
        self,
        selection: SelectionPlan,
        baseline: Mapping[str, Sequence[StageAssignment]],
        treatment_skeleton: TreatmentPlan,
        *,
        max_base_units: int | None,
        run_id: str | None = None,
        created_at: str | None = None,
        code_version: str | None = None,
    ) -> RunManifest:
        expected = {
            "step1": tuple(item.sample_id for item in baseline["step1"]),
            "step2": tuple(item.sample_id for item in baseline["step2"]),
            "step3": tuple(item.sample_id for item in baseline["step3"]),
            "step4a": tuple(item.sample_id for item in treatment_skeleton.step4a),
            "step4b": tuple(item.sample_id for item in treatment_skeleton.step4b),
        }
        sampling_plan = {
            "selection": selection.to_dict(),
            "treatment_design": {
                "fixed_percentages": list(self.fixed_percentages),
                "include_simulated_consensus": self.include_simulated_consensus,
                "simulated_consensus_method": "leave_one_persona_out_valid_binary",
                "include_retest": self.include_retest,
                "include_placebo": self.include_placebo,
                "replicates": self.replicates,
                "master_seed": self.seed,
                "treatment_order": "deterministic_sha256_seeded_shuffle",
            },
        }
        return RunManifest.create(
            run_id=run_id or self._new_run_id(self.pipeline_name),
            pipeline=self.pipeline_name,
            config=self._manifest_config(max_base_units),
            data_hashes=self._data_hashes(),
            prompt_hashes=self._prompt_hashes,
            sampling_plan=sampling_plan,
            expected_sample_ids=expected,
            code_version=code_version,
            created_at=created_at,
        )

    def _restore_plans(
        self,
        manifest: RunManifest,
    ) -> tuple[
        SelectionPlan,
        dict[str, tuple[StageAssignment, ...]],
        TreatmentPlan,
    ]:
        if self.llm_interface is not None:
            raise CheckpointValidationError(
                "checkpoint restore refuses to reuse an existing or injected backend; "
                "create a fresh runner and provide a lazy llm_factory if a test or "
                "application needs a custom backend"
            )
        self._apply_manifest_config(manifest.config)
        self.load_prompt_templates()
        try:
            selection_payload = manifest.sampling_plan["selection"]
        except KeyError as exc:
            raise CheckpointValidationError("manifest has no selection plan") from exc
        selection = SelectionPlan.from_dict(selection_payload)
        current_code_version = self._detect_code_version(
            Path(__file__).resolve().parents[2]
        )
        if (
            manifest.code_version is not None
            and current_code_version is not None
            and current_code_version != manifest.code_version
        ):
            raise CheckpointValidationError(
                "current Git code version does not match the checkpoint manifest"
            )
        baseline = build_baseline_assignments(
            selection,
            replicates=self.replicates,
            master_seed=self.seed,
        )
        skeleton = build_treatment_plan(
            selection,
            valid_binary_beliefs=None,
            step2_predictions={},
            fixed_percentages=self.fixed_percentages,
            include_simulated_consensus=self.include_simulated_consensus,
            include_retest=self.include_retest,
            include_placebo=self.include_placebo,
            replicates=self.replicates,
            master_seed=self.seed,
        )
        rebuilt = self._create_manifest(
            selection,
            baseline,
            skeleton,
            max_base_units=manifest.config.get("max_base_units"),
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            code_version=manifest.code_version,
        )
        if rebuilt.run_fingerprint != manifest.run_fingerprint:
            raise CheckpointValidationError(
                "current data, prompts, or reconstructed plan do not match the checkpoint manifest"
            )
        return selection, baseline, skeleton

    def _execute_stage_chunk(
        self,
        stage: str,
        assignments: Sequence[Assignment],
    ) -> list[ExperimentRecord]:
        raise NotImplementedError

    def _execute_stage(
        self,
        store: CheckpointStore,
        manifest: RunManifest,
        stage: str,
        assignments: Sequence[Assignment],
    ) -> dict[str, ExperimentRecord]:
        assignment_by_id = {
            assignment.sample_id: assignment for assignment in assignments
        }
        if len(assignment_by_id) != len(assignments):
            raise CheckpointValidationError(
                f"{stage} plan contains duplicate sample IDs"
            )
        validate_record_ids(
            manifest.expected_sample_ids[stage],
            assignment_by_id,
            require_complete=True,
        )
        completed = store.load_stage(manifest, stage, require_complete=False)
        self._validate_completed_assignments(stage, assignment_by_id, completed)
        missing = [
            assignment
            for assignment in assignments
            if assignment.sample_id not in completed
        ]

        # The model API accepts one seed per batch. Grouping by replicate seed
        # preserves the planned seed while retaining batched inference.
        by_seed: dict[int, list[Assignment]] = {}
        for assignment in missing:
            by_seed.setdefault(assignment.seed, []).append(assignment)

        chunk_index = store.next_chunk_index(manifest, stage)
        for seed_assignments in by_seed.values():
            for start in range(0, len(seed_assignments), self.chunk_size):
                chunk = seed_assignments[start : start + self.chunk_size]
                records = self._execute_stage_chunk(stage, chunk)
                validate_record_ids(
                    [assignment.sample_id for assignment in chunk],
                    [record.sample_id for record in records],
                    require_complete=True,
                )
                self._validate_completed_assignments(
                    stage,
                    {assignment.sample_id: assignment for assignment in chunk},
                    {record.sample_id: record for record in records},
                )
                store.write_chunk(manifest, stage, chunk_index, records)
                chunk_index += 1
        return store.load_stage(manifest, stage, require_complete=True)

    @staticmethod
    def _validate_completed_assignments(
        stage: str,
        assignments: Mapping[str, Assignment],
        records: Mapping[str, ExperimentRecord],
    ) -> None:
        """Reject stale/tampered shards whose metadata no longer matches the plan.

        Treatment sample IDs intentionally exclude baseline-derived values such
        as realized consensus and survey surprise so that the complete sample
        set can be fixed before inference.  Exact metadata comparison is
        therefore required when resuming: otherwise regenerated upstream
        records could be silently combined with downstream observations created
        from different treatment values.
        """

        for sample_id, record in records.items():
            assignment = assignments.get(sample_id)
            if assignment is None:
                raise CheckpointValidationError(
                    f"{stage} checkpoint contains unplanned sample ID {sample_id!r}"
                )
            payload = assignment.to_dict()
            expected_metadata = payload.get("metadata")
            if not isinstance(expected_metadata, Mapping):
                raise CheckpointValidationError(
                    f"{stage} assignment {sample_id!r} has invalid metadata"
                )
            if canonical_json(record.metadata) != canonical_json(expected_metadata):
                raise CheckpointValidationError(
                    f"{stage} checkpoint metadata does not match the reconstructed "
                    f"assignment for sample {sample_id!r}; start a new run"
                )

    @staticmethod
    def _proposal_key(record: ExperimentRecord) -> tuple[str, str, str]:
        metadata = record.metadata
        return (metadata["persona"], metadata["category"], metadata["proposal"])

    @staticmethod
    def _binary_decision(record: ExperimentRecord) -> int | None:
        if record.status is not ResultStatus.VALID:
            return None
        if record.value == "Yes":
            return 1
        if record.value == "No":
            return 0
        if isinstance(record.value, Mapping):
            # Logprob records measure an actually sampled, format-valid answer;
            # aggregate candidate mass is diagnostic and must not replace that
            # observed Yes/No token with an argmax-derived pseudo-response.
            if record.value.get("format_valid") is not True:
                return None
            sampled_choice = record.value.get("sampled_choice")
            if sampled_choice == "Yes":
                return 1
            if sampled_choice == "No":
                return 0
        return None

    def _resolve_treatments(
        self,
        selection: SelectionPlan,
        step1_records: Iterable[ExperimentRecord],
        step2_records: Iterable[ExperimentRecord],
    ) -> TreatmentPlan:
        beliefs: dict[tuple[str, str, str], list[int | None]] = defaultdict(list)
        predictions: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for record in step1_records:
            beliefs[self._proposal_key(record)].append(self._binary_decision(record))
        for record in step2_records:
            if record.status is ResultStatus.VALID and isinstance(
                record.value, (int, float)
            ):
                predictions[self._proposal_key(record)].append(float(record.value))
        mean_predictions = {
            key: sum(values) / len(values)
            for key, values in predictions.items()
            if values
        }
        return build_treatment_plan(
            selection,
            valid_binary_beliefs=beliefs,
            step2_predictions=mean_predictions,
            fixed_percentages=self.fixed_percentages,
            include_simulated_consensus=self.include_simulated_consensus,
            include_retest=self.include_retest,
            include_placebo=self.include_placebo,
            replicates=self.replicates,
            master_seed=self.seed,
        )

    def _compile_and_save(
        self,
        manifest: RunManifest,
        selection: SelectionPlan,
        records: Mapping[str, Mapping[str, ExperimentRecord]],
    ) -> Path:
        self.results = compile_grouped_results(
            selection.base_units,
            step1_records=records["step1"].values(),
            step2_records=records["step2"].values(),
            step3_records=records["step3"].values(),
            step4a_records=records["step4a"].values(),
            step4b_records=records["step4b"].values(),
            expected_ids_by_stage=manifest.expected_sample_ids,
            require_complete=True,
        )
        self.last_status_summary = self._status_summary(manifest, records)
        output_path = self.results_dir / f"results_{manifest.run_id}.json"
        atomic_write_json(
            output_path,
            {
                "schema_version": manifest.schema_version,
                "run_id": manifest.run_id,
                "pipeline": self.pipeline_name,
                "model": self.model_name,
                "run_fingerprint": manifest.run_fingerprint,
                "manifest": manifest.to_dict(),
                "status_summary": self.last_status_summary,
                "results": self.results,
            },
        )
        return output_path

    @staticmethod
    def _status_summary(
        manifest: RunManifest,
        records: Mapping[str, Mapping[str, ExperimentRecord]],
    ) -> dict[str, dict[str, int | float]]:
        """Summarize valid, invalid, error, and missing observations per stage."""

        summary: dict[str, dict[str, int | float]] = {}
        for stage, expected_ids in manifest.expected_sample_ids.items():
            stage_records = records.get(stage, {})
            counts = {"valid": 0, "invalid": 0, "error": 0}
            for record in stage_records.values():
                counts[record.status.value] += 1
            expected = len(expected_ids)
            observed = sum(counts.values())
            missing = max(0, expected - observed)
            denominator = expected or 1
            summary[stage] = {
                "expected": expected,
                "observed": observed,
                "valid": counts["valid"],
                "invalid": counts["invalid"],
                "error": counts["error"],
                "missing": missing,
                "valid_rate": counts["valid"] / denominator,
                "invalid_rate": counts["invalid"] / denominator,
                "error_rate": counts["error"] / denominator,
                "missing_rate": missing / denominator,
            }
        return summary

    def has_execution_errors(self) -> bool:
        """Return whether the most recently compiled run contains ERROR records."""

        return any(
            values.get("error", 0) for values in self.last_status_summary.values()
        )

    def _backend_request_estimate(
        self, logical_counts: Mapping[str, int]
    ) -> dict[str, int]:
        return dict(logical_counts)

    def dry_run(
        self,
        *,
        personas: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        unique_proposals: Sequence[tuple[str, str]] | None = None,
        max_base_units: int | None = None,
    ) -> dict[str, Any]:
        self.load_prompt_templates()
        selection, baseline, skeleton = self._build_plans(
            personas=personas,
            categories=categories,
            unique_proposals=unique_proposals,
            max_base_units=max_base_units,
        )
        logical = {
            "step1": len(baseline["step1"]),
            "step2": len(baseline["step2"]),
            "step3": len(baseline["step3"]),
            "step4a": len(skeleton.step4a),
            "step4b": len(skeleton.step4b),
        }
        requests = self._backend_request_estimate(logical)
        return {
            "pipeline": self.pipeline_name,
            "model": self.model_name,
            "base_units": len(selection.base_units),
            "proposal_units": len(selection.proposal_units),
            "replicates": self.replicates,
            "logical_sample_counts": logical,
            "backend_sequence_counts": requests,
            "total_backend_sequences": sum(requests.values()),
            "backend_sequence_count_semantics": "planned upper bound",
            "note": (
                "logical sample counts are exact for the plan; backend sequence "
                "counts are upper bounds because unresolved simulated-consensus "
                "cells and later phases may be skipped after invalid prior outputs"
            ),
        }

    def run_experiments(
        self,
        personas: Sequence[str] | None = None,
        unique_proposals: Sequence[tuple[str, str]] | None = None,
        max_base_units: int | None = None,
        *,
        categories: Sequence[str] | None = None,
    ) -> Path:
        self.load_prompt_templates()
        selection, baseline, skeleton = self._build_plans(
            personas=personas,
            categories=categories,
            unique_proposals=unique_proposals,
            max_base_units=max_base_units,
        )
        manifest = self._create_manifest(
            selection,
            baseline,
            skeleton,
            max_base_units=max_base_units,
            code_version=self._detect_code_version(Path(__file__).resolve().parents[2]),
        )
        self.active_manifest = manifest
        store = CheckpointStore(self.checkpoint_root)
        store.save_manifest(manifest)

        records: dict[str, dict[str, ExperimentRecord]] = {}
        for stage in ("step1", "step2", "step3"):
            records[stage] = self._execute_stage(
                store, manifest, stage, baseline[stage]
            )
        resolved = self._resolve_treatments(
            selection,
            records["step1"].values(),
            records["step2"].values(),
        )
        records["step4a"] = self._execute_stage(
            store, manifest, "step4a", resolved.step4a
        )
        records["step4b"] = self._execute_stage(
            store, manifest, "step4b", resolved.step4b
        )
        return self._compile_and_save(manifest, selection, records)

    def run_experiments_from_step(self, run_id: str, resume_step: str) -> Path:
        if resume_step not in {"step1", "step2", "step3", "step4a", "step4b"}:
            raise ValueError(f"invalid resume step: {resume_step}")
        store = CheckpointStore(self.checkpoint_root)
        manifest = store.load_manifest(run_id)
        selection, baseline, skeleton = self._restore_plans(manifest)
        del skeleton
        self.active_manifest = manifest
        store.validate_resume_dependencies(manifest, resume_step)

        stages = ("step1", "step2", "step3", "step4a", "step4b")
        resume_index = stages.index(resume_step)
        records: dict[str, dict[str, ExperimentRecord]] = {}
        resolved: TreatmentPlan | None = None
        for index, stage in enumerate(stages):
            if stage in {"step4a", "step4b"} and resolved is None:
                resolved = self._resolve_treatments(
                    selection,
                    records["step1"].values(),
                    records["step2"].values(),
                )
            assignments: Sequence[Assignment]
            if stage in baseline:
                assignments = baseline[stage]
            elif stage == "step4a":
                assignments = resolved.step4a
            else:
                assignments = resolved.step4b

            if index < resume_index:
                records[stage] = store.load_stage(
                    manifest, stage, require_complete=True
                )
                self._validate_completed_assignments(
                    stage,
                    {assignment.sample_id: assignment for assignment in assignments},
                    records[stage],
                )
            else:
                records[stage] = self._execute_stage(
                    store, manifest, stage, assignments
                )
        return self._compile_and_save(manifest, selection, records)

    @classmethod
    def list_checkpoint_runs(cls, results_dir: str | Path) -> list[dict[str, Any]]:
        store = CheckpointStore(Path(results_dir) / "checkpoints")
        runs: list[dict[str, Any]] = []
        if not store.root.exists():
            return runs
        for manifest_path in sorted(store.root.glob("*/manifest.json")):
            run_id = manifest_path.parent.name
            try:
                manifest = store.load_manifest(run_id)
                if manifest.pipeline != cls.pipeline_name:
                    continue
                missing = {
                    stage: len(store.missing_sample_ids(manifest, stage))
                    for stage in manifest.expected_sample_ids
                }
                runs.append(
                    {
                        "run_id": run_id,
                        "pipeline": manifest.pipeline,
                        "model": manifest.config.get("model_name"),
                        "created_at": manifest.created_at,
                        "missing_by_stage": missing,
                    }
                )
            except (OSError, ValueError) as exc:
                runs.append({"run_id": run_id, "error": str(exc)})
        return runs


def print_json(value: Any) -> None:
    """Small CLI helper kept separate for easy output capture in tests."""

    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
