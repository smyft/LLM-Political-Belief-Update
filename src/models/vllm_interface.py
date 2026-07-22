"""
vLLM Interface Module.

This module implements a VLLMInterface class for interacting with LLMs via vLLM.
It supports loading models locally, generating responses, and extracting logprob information.
"""

from __future__ import annotations

import gc
import inspect
import re

from src.models.binary_logprob import (
    build_yes_no_candidate_map,
    score_yes_no_candidates,
)

# vLLM imports - these will only work if vllm is installed
try:
    import torch
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def free_gpu_memory():
    """
    Aggressively cleans up GPU memory to allow loading the next model.
    vLLM is persistent, so we must delete the object and run garbage collection.
    """
    print("Cleaning up GPU memory...")

    # Python garbage collection
    gc.collect()

    # PyTorch CUDA cache cleanup
    if VLLM_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("GPU memory cleaned.")


def extract_thinking_process(text: str) -> str:
    """
    Extracts the thinking process from the Model's output.

    Handles:
    - Explicit <think>...</think> tags
    - Answer filtering from beginning, middle, and end of text

    Args:
        text: The raw model output

    Returns:
        The extracted thinking process
    """
    THINK_START_TAG = "<think>"
    THINK_END_TAG = "</think>"

    # Strategy 1: Explicit Tags - Check for complete pair: <think> ...Content... </think>
    complete_pattern = re.compile(
        f"{re.escape(THINK_START_TAG)}(.*?){re.escape(THINK_END_TAG)}",
        re.DOTALL | re.IGNORECASE,
    )
    match_complete = complete_pattern.search(text)
    if match_complete:
        return match_complete.group(1).strip()

    # Strategy 2: Check for unclosed tag: <think> ...Content (End of Text)
    incomplete_pattern = re.compile(
        f"{re.escape(THINK_START_TAG)}(.*)$", re.DOTALL | re.IGNORECASE
    )
    match_incomplete = incomplete_pattern.search(text)
    if match_incomplete:
        return match_incomplete.group(1).strip()

    # Some chat templates prefill the opening tag in the assistant prompt, so
    # the completion contains reasoning followed only by a closing tag.
    closing_only_pattern = re.compile(
        rf"^(.*?){re.escape(THINK_END_TAG)}", re.DOTALL | re.IGNORECASE
    )
    match_closing_only = closing_only_pattern.search(text)
    if match_closing_only:
        return match_closing_only.group(1).strip()

    # No tags - return the original text with answer filtering.
    return filter_answer_from_text(text)


def filter_answer_from_text(text: str) -> str:
    """
    Filter out answer-related content from the text.
    This helps extract pure reasoning without the final answer.

    Args:
        text: The raw text

    Returns:
        Filtered text without answer statements
    """
    # Remove explicit answer patterns at the end
    answer_patterns = [
        # "Yes" or "No" at the end
        r"\s*(?:So,?\s+)?(?:the\s+)?(?:final\s+)?answer(?:\s+is)?(?:\s*:)?\s*(?:Yes|No)\.?\s*$",
        # JSON-like answer at the end
        r'\s*\{[^}]*"answer"\s*:\s*"(?:Yes|No)"[^}]*\}\s*$',
        # "My answer is Yes/No"
        r"\s*(?:My\s+)?answer(?:\s+is)?(?:\s*:)?\s*(?:Yes|No)\.?\s*$",
        # "Therefore, Yes/No"
        r"\s*(?:Therefore|Thus|Hence),?\s*(?:Yes|No)\.?\s*$",
    ]

    result = text.strip()
    for pattern in answer_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    return result.strip()


class VLLMInterface:
    """
    Interface for interacting with LLMs via vLLM.

    This class supports:
    - Loading models locally
    - Generating responses via chat
    - Extracting logprob information for probability belief estimation
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        code_revision: str | None = None,
        dtype: str = "auto",
        enforce_eager: bool = False,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        language_model_only: bool = False,
        enable_thinking: bool | None = None,
        **kwargs,
    ):
        """
        Initialize the vLLM interface.

        Args:
            model_name: Name or path of the model to load
            gpu_memory_utilization: Fraction of GPU memory to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            trust_remote_code: Whether to trust remote code for model loading
            revision: Optional model branch, tag, or commit identifier
            tokenizer_revision: Optional tokenizer branch, tag, or commit identifier
            code_revision: Optional remote-code branch, tag, or commit identifier
            dtype: Data type for model weights
            enforce_eager: Whether to use eager mode (recommended for compatibility)
            max_model_len: Optional context-length cap passed to vLLM
            max_num_seqs: Optional scheduler sequence cap passed to vLLM
            language_model_only: Skip multimodal towers for text-only inference
            enable_thinking: Optional chat-template thinking-mode switch
            **kwargs: Additional arguments for vLLM
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with: pip install vllm"
            )

        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.code_revision = code_revision
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.language_model_only = language_model_only
        self.enable_thinking = enable_thinking
        self.extra_kwargs = kwargs

        self.llm = None
        self.tokenizer = None
        self._yes_no_candidate_map = None
        self._bounded_scoring_preflight_complete = False

    def load_model(self):
        """Load the model into memory."""
        if self.llm is not None:
            if self.tokenizer is None:
                self.tokenizer = self.llm.get_tokenizer()
            print(f"Model {self.model_name} is already loaded.")
            return

        print(f"Loading model: {self.model_name}...")
        engine_kwargs = dict(self.extra_kwargs)
        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len
        if self.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = self.max_num_seqs
        engine_kwargs["language_model_only"] = self.language_model_only

        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            tokenizer_revision=self.tokenizer_revision,
            code_revision=self.code_revision,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
            **engine_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded successfully.")

    def _get_yes_no_candidate_map(self) -> dict[int, str]:
        """Build and cache the bounded single-token candidate set."""

        if self.tokenizer is None:
            raise RuntimeError("The tokenizer is unavailable; load the model first.")
        if self._yes_no_candidate_map is None:
            self._yes_no_candidate_map = build_yes_no_candidate_map(self.tokenizer)
        return self._yes_no_candidate_map

    def _ensure_bounded_scoring_api(self) -> None:
        """Fail clearly instead of changing prompt semantics on old vLLM APIs."""

        try:
            parameters = inspect.signature(self.llm.chat).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to inspect the installed vLLM chat API."
            ) from exc

        required = {
            "use_tqdm",
            "add_generation_prompt",
            "continue_final_message",
            "chat_template_kwargs",
        }
        missing = sorted(required.difference(parameters))
        if missing:
            raise RuntimeError(
                "The installed vLLM is incompatible with bounded scoring; "
                "missing LLM.chat parameters: " + ", ".join(missing)
            )

    def _validate_bounded_scoring_chat_template(self) -> None:
        """Validate the completed-analysis, fresh-assistant scoring contract."""

        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError(
                "The tokenizer is incompatible with bounded scoring: "
                "apply_chat_template() is unavailable."
            )

        user_marker = "LLM_BELIEF_ANALYSIS_REQUEST_MARKER_7F31"
        analysis_marker = "LLM_BELIEF_VISIBLE_ANALYSIS_MARKER_4A92"
        request_marker = "LLM_BELIEF_BINARY_REQUEST_MARKER_9C26"
        answer_marker = "LLM_BELIEF_BINARY_ANSWER_MARKER_2D84"
        user_message = {"role": "user", "content": user_marker}
        assistant_message = {"role": "assistant", "content": analysis_marker}
        scoring_request = {"role": "user", "content": request_marker}
        template_kwargs = self._chat_template_kwargs() or {}

        def render(
            messages: list[dict[str, str]], *, add_generation_prompt: bool
        ) -> str:
            value = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=False,
                **template_kwargs,
            )
            if not isinstance(value, str):
                raise TypeError("chat template must return text when tokenize=False")
            return value

        try:
            phase1_prefix = render([user_message], add_generation_prompt=True)
            completed_analysis = render(
                [user_message, assistant_message], add_generation_prompt=False
            )
            completed_scoring_dialogue = render(
                [user_message, assistant_message, scoring_request],
                add_generation_prompt=False,
            )
            scoring_prefix = render(
                [user_message, assistant_message, scoring_request],
                add_generation_prompt=True,
            )
            completed_answer = render(
                [
                    user_message,
                    assistant_message,
                    scoring_request,
                    {"role": "assistant", "content": answer_marker},
                ],
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "The tokenizer chat template is incompatible with bounded "
                "scoring after a completed analysis turn."
            ) from exc

        if not completed_analysis.startswith(phase1_prefix + analysis_marker):
            raise RuntimeError(
                "The tokenizer chat template cannot reconstruct phase-1 visible "
                "analysis as assistant content."
            )
        marker_indexes = [
            scoring_prefix.find(marker)
            for marker in (user_marker, analysis_marker, request_marker)
        ]
        if (
            any(index < 0 for index in marker_indexes)
            or marker_indexes != sorted(marker_indexes)
            or any(
                scoring_prefix.count(marker) != 1
                for marker in (
                    user_marker,
                    analysis_marker,
                    request_marker,
                )
            )
        ):
            raise RuntimeError(
                "The tokenizer chat template must preserve the analysis dialogue "
                "and final user request in order."
            )
        if not scoring_prefix.startswith(completed_scoring_dialogue) or len(
            scoring_prefix
        ) == len(completed_scoring_dialogue):
            raise RuntimeError(
                "The tokenizer chat template does not append a fresh assistant "
                "generation prompt for bounded scoring."
            )
        if not completed_answer.startswith(scoring_prefix + answer_marker):
            raise RuntimeError(
                "The tokenizer chat template does not place the bounded token at "
                "the start of visible assistant answer content."
            )

    @staticmethod
    def _build_bounded_sampling_params(candidate_token_ids: list[int], **kwargs):
        """Construct the pinned vLLM bounded-scoring parameter contract."""

        try:
            return SamplingParams(
                max_tokens=1,
                logprob_token_ids=candidate_token_ids,
                skip_special_tokens=False,
                **kwargs,
            )
        except TypeError as exc:
            raise RuntimeError(
                "The installed vLLM SamplingParams is incompatible with bounded "
                "scoring; logprob_token_ids support from vLLM 0.24 or a "
                "compatible release is required."
            ) from exc

    def preflight_bounded_scoring(self) -> dict[int, str]:
        """Load once and validate bounded first-token scoring without inference.

        The returned candidate map is detached from the cached tokenizer-specific
        map so callers cannot mutate later scoring behavior.
        """

        if self.llm is None or self.tokenizer is None:
            self.load_model()
        if not self._bounded_scoring_preflight_complete:
            self._ensure_bounded_scoring_api()
            self._validate_bounded_scoring_chat_template()
            candidate_map = self._get_yes_no_candidate_map()
            self._build_bounded_sampling_params(list(candidate_map))
            self._bounded_scoring_preflight_complete = True
        return dict(self._get_yes_no_candidate_map())

    def _normalize_and_validate_dialogues(
        self,
        dialogue_history: list[dict] | list[list[dict]],
    ) -> list[list[dict]]:
        """
        Normalize input to batched OpenAI-style dialogues and validate schema.

        Args:
            dialogue_history: A single dialogue ([{role, content}, ...]) or
                              a batch of dialogues ([[{role, content}, ...], ...])
        Returns:
            Normalized list of dialogues.
        """
        if not isinstance(dialogue_history, list) or not dialogue_history:
            raise TypeError(
                "dialogue_history must be a non-empty list of OpenAI-style messages/dialogues."
            )

        # Single dialogue: List[Dict]
        if isinstance(dialogue_history[0], dict):
            dialogues = [dialogue_history]
        # Batch dialogue: List[List[Dict]]
        elif isinstance(dialogue_history[0], list):
            dialogues = dialogue_history
        else:
            raise TypeError(
                "dialogue_history must be List[Dict] or List[List[Dict]]. "
                "Passing plain string prompts is not supported."
            )

        for d_idx, dialogue in enumerate(dialogues):
            if not isinstance(dialogue, list) or not dialogue:
                raise TypeError(
                    f"Dialogue at index {d_idx} must be a non-empty list of messages."
                )

            for m_idx, msg in enumerate(dialogue):
                if not isinstance(msg, dict):
                    raise TypeError(
                        f"Message at dialogue[{d_idx}][{m_idx}] must be a dict."
                    )
                if "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"Message at dialogue[{d_idx}][{m_idx}] must contain 'role' and 'content'."
                    )
                if not isinstance(msg["role"], str) or not isinstance(
                    msg["content"], str
                ):
                    raise TypeError(
                        f"'role' and 'content' at dialogue[{d_idx}][{m_idx}] must both be strings."
                    )
                if "reasoning_content" in msg:
                    if not isinstance(msg["reasoning_content"], str):
                        raise TypeError(
                            "'reasoning_content' at "
                            f"dialogue[{d_idx}][{m_idx}] must be a string."
                        )
                    if msg["role"].strip().lower() != "assistant":
                        raise ValueError(
                            "'reasoning_content' is only valid on assistant messages; "
                            f"found it at dialogue[{d_idx}][{m_idx}]."
                        )

        return dialogues

    def _chat_template_kwargs(self) -> dict[str, bool] | None:
        """Return a fresh, manifest-controlled chat-template configuration."""

        if self.enable_thinking is None:
            return None
        return {"enable_thinking": self.enable_thinking}

    def chat(
        self,
        dialogue_history: list[dict] | list[list[dict]],
        temperature: float = 0,
        max_tokens: int = 1024,
        seed: int = 42,
        logprobs: int | None = None,
        show_progress: bool = True,
        desc: str = "Processing",
        **kwargs,
    ) -> list[dict]:
        """
        Generate responses using the loaded model.

        Args:
            dialogue_history: Single dialogue or list of dialogues (each dialogue is a list of message dicts)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed
            logprobs: Number of top logprobs to return (None for no logprobs)
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            **kwargs: Additional arguments for SamplingParams

        Returns:
            List of response dictionaries containing generated_text and optionally logprobs
        """
        if self.llm is None:
            self.load_model()

        # Normalize input to list of dialogues and validate OpenAI-style schema
        dialogues = self._normalize_and_validate_dialogues(dialogue_history)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            skip_special_tokens=False,
            **kwargs,
        )

        # Generate responses
        outputs = self.llm.chat(
            dialogues,
            sampling_params=sampling_params,
            use_tqdm=show_progress,
            chat_template_kwargs=self._chat_template_kwargs(),
        )

        # Process outputs
        results = []
        for output in outputs:
            result = {
                "generated_text": output.outputs[0].text,
                "finish_reason": (
                    output.outputs[0].finish_reason
                    if hasattr(output.outputs[0], "finish_reason")
                    else None
                ),
            }

            # Include logprobs if requested
            if logprobs and output.outputs[0].logprobs:
                result["logprobs"] = output.outputs[0].logprobs

            results.append(result)

        return results

    def _chat_with_bounded_candidate_scores(
        self,
        dialogue_history: list[dict] | list[list[dict]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        logprobs: int | None,
        show_progress: bool,
        last_role: str,
        add_generation_prompt: bool,
        continue_final_message: bool,
        kwargs: dict,
    ) -> list[dict]:
        """Run the shared finite-candidate, one-token scoring implementation."""

        candidate_map = self.preflight_bounded_scoring()

        dialogues = self._normalize_and_validate_dialogues(dialogue_history)
        for index, dialogue in enumerate(dialogues):
            actual_role = dialogue[-1]["role"].strip().lower()
            if actual_role != last_role:
                raise ValueError(
                    f"Dialogue at index {index} must end with a {last_role} "
                    "message for bounded scoring."
                )

        if max_tokens != 1:
            raise ValueError("Bounded probability extraction requires max_tokens=1.")
        if logprobs is not None and (not isinstance(logprobs, int) or logprobs < 1):
            raise ValueError("logprobs must be a positive integer when provided.")

        candidate_token_ids = list(candidate_map)

        reserved_sampling_keys = {
            "temperature",
            "max_tokens",
            "seed",
            "logprobs",
            "prompt_logprobs",
            "logprob_token_ids",
            "skip_special_tokens",
        }
        duplicate_keys = reserved_sampling_keys.intersection(kwargs)
        if duplicate_keys:
            raise ValueError(
                "bounded-scoring kwargs cannot override: "
                + ", ".join(sorted(duplicate_keys))
            )

        # vLLM 0.24 returns exactly these bounded candidate logprobs plus the
        # sampled token. Do not set `logprobs`, especially not -1, because that
        # would request a top-K or full-vocabulary distribution.
        sampling_params = self._build_bounded_sampling_params(
            candidate_token_ids,
            temperature=temperature,
            seed=seed,
            **kwargs,
        )

        outputs = self.llm.chat(
            dialogues,
            sampling_params=sampling_params,
            use_tqdm=show_progress,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            chat_template_kwargs=self._chat_template_kwargs(),
        )

        # Process outputs
        results = []
        for output in outputs:
            if not output.outputs:
                score = score_yes_no_candidates(None, candidate_map)
                results.append(
                    {
                        "generated_text": "",
                        "finish_reason": None,
                        "logprobs_raw_vllm": {},
                        "logprobs_raw": {},
                        **score,
                    }
                )
                continue

            completion = output.outputs[0]
            result = {
                "generated_text": completion.text,
                "finish_reason": getattr(completion, "finish_reason", None),
            }

            token_ids = list(getattr(completion, "token_ids", ()) or ())
            sampled_token_id = int(token_ids[0]) if token_ids else None
            all_positions = getattr(completion, "logprobs", None)
            first_token_logprobs = all_positions[0] if all_positions else None
            score = score_yes_no_candidates(
                first_token_logprobs,
                candidate_map,
                sampled_token_id=sampled_token_id,
            )

            # Keep the legacy aggregate key while exposing the complete,
            # explicit validity and truncation metadata returned by the scorer.
            result["logprobs_raw_vllm"] = first_token_logprobs or {}
            result["logprobs_raw"] = score["label_logprobs"] or {}
            result.update(score)
            results.append(result)

        return results

    def chat_with_bounded_candidates(
        self,
        dialogue_history: list[dict] | list[list[dict]],
        temperature: float = 0,
        max_tokens: int = 1,
        seed: int = 42,
        logprobs: int | None = 20,
        show_progress: bool = True,
        desc: str = "Processing",
        **kwargs,
    ) -> list[dict]:
        """Score the first token of a fresh assistant turn.

        The dialogue must end with the user request that follows a completed
        assistant analysis turn. ``desc`` is accepted for the common backend
        interface; progress rendering is delegated to vLLM.
        """

        del desc
        return self._chat_with_bounded_candidate_scores(
            dialogue_history,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            show_progress=show_progress,
            last_role="user",
            add_generation_prompt=True,
            continue_final_message=False,
            kwargs=kwargs,
        )

    def extract_thinking(self, response_text: str) -> str:
        """
        Extract the thinking process from a response.

        Args:
            response_text: The raw response text

        Returns:
            The extracted thinking process
        """
        return extract_thinking_process(response_text)

    def free_memory(self):
        """Free GPU memory by deleting the model."""
        llm = self.llm
        # Break owner references before collection/cache cleanup.  The old
        # implementation collected while self.llm still retained the model.
        self.llm = None
        self.tokenizer = None
        self._yes_no_candidate_map = None
        self._bounded_scoring_preflight_complete = False
        try:
            if llm is not None:
                # Use a public close hook when a vLLM release provides one;
                # avoid reaching into unstable private engine internals.
                close = getattr(llm, "close", None)
                try:
                    if callable(close):
                        close()
                finally:
                    del close
        finally:
            # Drop the last owner-local model reference *before* collection and
            # CUDA cache cleanup.  Keeping it as a helper argument would retain
            # the model until after gc.collect(), defeating cleanup.
            del llm
            free_gpu_memory()

    def close(self):
        """Idempotent resource cleanup alias shared with the API backend."""
        self.free_memory()


if __name__ == "__main__":
    # Test the vLLM interface (only works if vLLM is installed)
    if VLLM_AVAILABLE:
        print("Testing VLLMInterface...")

        # Test with a small model
        model_name = "Qwen/Qwen3-0.6B"

        try:
            interface = VLLMInterface(model_name)
            interface.load_model()

            # Test chat
            dialogues = [
                [
                    {"role": "user", "content": "Is 2+2=4? Answer Yes or No."},
                ],
            ]

            results = interface.chat(dialogues, max_tokens=1024, desc="Testing")
            print("Chat result:", results)

            # Test chat with logprobs
            results_with_logprobs = interface.chat(
                dialogues,
                max_tokens=1024,
                logprobs=None,
                desc="Testing Phase 1 thinking",
            )
            print("Phase 1 thinking:", results_with_logprobs)

            # Test bounded first-token scoring on a fresh assistant turn
            scoring_dialogues = [
                [
                    {"role": "user", "content": "Is 2+2=4? Answer Yes or No."},
                    {
                        "role": "assistant",
                        "content": "2+2 equals 4.",
                    },
                    {"role": "user", "content": "Answer exactly Yes or No."},
                ],
            ]

            scoring_results = interface.chat_with_bounded_candidates(
                scoring_dialogues,
                max_tokens=1,
                logprobs=20,
                desc="Testing bounded scoring",
            )
            print("Bounded-scoring result:", scoring_results)

            interface.free_memory()
            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("vLLM not available. Skipping tests.")
