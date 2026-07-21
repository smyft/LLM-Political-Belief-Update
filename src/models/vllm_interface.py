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

    # Strategy 3: No tags - return the original text with answer filtering
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
        self.extra_kwargs = kwargs

        self.llm = None
        self.tokenizer = None
        self._yes_no_candidate_map = None
        self._continuation_preflight_complete = False

    def load_model(self):
        """Load the model into memory."""
        if self.llm is not None:
            if self.tokenizer is None:
                self.tokenizer = self.llm.get_tokenizer()
            print(f"Model {self.model_name} is already loaded.")
            return

        print(f"Loading model: {self.model_name}...")
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
            **self.extra_kwargs,
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

    def _ensure_continuation_api(self) -> None:
        """Fail clearly instead of changing prompt semantics on old vLLM APIs."""

        try:
            parameters = inspect.signature(self.llm.chat).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to inspect the installed vLLM chat API."
            ) from exc

        required = {"use_tqdm", "add_generation_prompt", "continue_final_message"}
        missing = sorted(required.difference(parameters))
        if missing:
            raise RuntimeError(
                "The installed vLLM is incompatible with continuation scoring; "
                "missing LLM.chat parameters: " + ", ".join(missing)
            )

    @staticmethod
    def _build_continuation_sampling_params(candidate_token_ids: list[int], **kwargs):
        """Construct the pinned vLLM continuation-scoring parameter contract."""

        try:
            return SamplingParams(
                max_tokens=1,
                logprob_token_ids=candidate_token_ids,
                skip_special_tokens=False,
                **kwargs,
            )
        except TypeError as exc:
            raise RuntimeError(
                "The installed vLLM SamplingParams is incompatible with continuation "
                "scoring; logprob_token_ids support from vLLM 0.24 or a "
                "compatible release is required."
            ) from exc

    def preflight_continuation_scoring(self) -> dict[int, str]:
        """Load once and validate bounded continuation scoring without inference.

        The returned candidate map is detached from the cached tokenizer-specific
        map so callers cannot mutate later scoring behavior.
        """

        if self.llm is None or self.tokenizer is None:
            self.load_model()
        if not self._continuation_preflight_complete:
            self._ensure_continuation_api()
            candidate_map = self._get_yes_no_candidate_map()
            self._build_continuation_sampling_params(list(candidate_map))
            self._continuation_preflight_complete = True
        return dict(self._get_yes_no_candidate_map())

    def _normalize_and_validate_dialogues(
        self,
        dialogue_history: list[dict] | list[list[dict]],
        require_assistant_last: bool = False,
    ) -> list[list[dict]]:
        """
        Normalize input to batched OpenAI-style dialogues and validate schema.

        Args:
            dialogue_history: A single dialogue ([{role, content}, ...]) or
                              a batch of dialogues ([[{role, content}, ...], ...])
            require_assistant_last: Whether the final message must be assistant

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

            if (
                require_assistant_last
                and dialogue[-1]["role"].strip().lower() != "assistant"
            ):
                raise ValueError(
                    f"Dialogue at index {d_idx} must end with an assistant message when "
                    "using continuation mode."
                )

        return dialogues

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

    def chat_with_continuation(
        self,
        dialogue_history: list[dict] | list[list[dict]],
        temperature: float = 0,
        max_tokens: int = 1,
        seed: int = 42,
        logprobs: int = 20,
        show_progress: bool = True,
        desc: str = "Processing",
        **kwargs,
    ) -> list[dict]:
        """
        Generate responses by continuing the last assistant message.

        This is used for Phase 2 of the logprob experiment where we want to
        extract the next token probabilities after the thinking process.

        Args:
            dialogue_history: List of dialogues where the last message is from assistant
                             and should be continued
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (usually 1 for logprob extraction)
            seed: Random seed
            logprobs: Retained for runner compatibility. Continuation scoring
                      requests the finite candidate IDs directly instead of a
                      top-K or full-vocabulary distribution.
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            **kwargs: Additional arguments for SamplingParams

        Returns:
            List of response dictionaries containing generated_text, logprobs, and probabilities
        """
        candidate_map = self.preflight_continuation_scoring()

        # Normalize input to list of dialogues and validate OpenAI-style schema
        dialogues = self._normalize_and_validate_dialogues(
            dialogue_history, require_assistant_last=True
        )

        if max_tokens != 1:
            raise ValueError(
                "Continuation probability extraction requires max_tokens=1."
            )
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
                "continuation kwargs cannot override: "
                + ", ".join(sorted(duplicate_keys))
            )

        # vLLM 0.24 returns exactly these bounded candidate logprobs plus the
        # sampled token. Do not set `logprobs`, especially not -1, because that
        # would request a top-K or full-vocabulary distribution.
        sampling_params = self._build_continuation_sampling_params(
            candidate_token_ids,
            temperature=temperature,
            seed=seed,
            **kwargs,
        )

        outputs = self.llm.chat(
            dialogues,
            sampling_params=sampling_params,
            use_tqdm=show_progress,
            add_generation_prompt=False,
            continue_final_message=True,
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
        self._continuation_preflight_complete = False
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

            # Test continuation for logprob extraction
            continuation_dialogues = [
                [
                    {"role": "user", "content": "Is 2+2=4? Answer Yes or No."},
                    {
                        "role": "assistant",
                        "content": "Let me think. 2+2 equals 4. So my answer is",
                    },
                ],
            ]

            continuation_results = interface.chat_with_continuation(
                continuation_dialogues,
                max_tokens=1,
                logprobs=20,
                desc="Testing continuation",
            )
            print("Continuation result:", continuation_results)

            interface.free_memory()
            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("vLLM not available. Skipping tests.")
