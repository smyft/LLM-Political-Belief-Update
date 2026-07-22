"""Unified API and vLLM model interfaces.

The API backend speaks the OpenAI-compatible HTTP protocol directly and
defaults to OpenRouter.  Credentials are validated before constructing the
HTTP client, and ``OPENAI_API_KEY`` is never consulted.
"""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from typing import Any, TypeAlias

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from src.models.vllm_interface import VLLMInterface, VLLM_AVAILABLE
except ImportError:  # pragma: no cover - protects API-only installations
    VLLMInterface = None
    VLLM_AVAILABLE = False


Dialogue: TypeAlias = list[dict[str, str]]
DialogueInput: TypeAlias = Dialogue | list[Dialogue]


def _normalize_dialogues(
    dialogue_history: DialogueInput,
) -> tuple[list[Dialogue], bool]:
    """Validate dialogue input and return ``(batch, was_batch)``."""

    if not isinstance(dialogue_history, list) or not dialogue_history:
        raise TypeError("dialogue_history must be a non-empty list")

    if isinstance(dialogue_history[0], dict):
        dialogues = [dialogue_history]
        was_batch = False
    elif isinstance(dialogue_history[0], list):
        dialogues = dialogue_history
        was_batch = True
    else:
        raise TypeError("dialogue_history must be List[Dict] or List[List[Dict]]")

    for dialogue_index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, list) or not dialogue:
            raise TypeError(f"dialogue at index {dialogue_index} must be non-empty")
        for message_index, message in enumerate(dialogue):
            if not isinstance(message, dict):
                raise TypeError(
                    f"message at dialogue[{dialogue_index}][{message_index}] must be a dict"
                )
            if {"role", "content"}.difference(message):
                raise ValueError(
                    f"message at dialogue[{dialogue_index}][{message_index}] "
                    "must contain role and content"
                )
            if not isinstance(message["role"], str) or not isinstance(
                message["content"], str
            ):
                raise TypeError("message role and content must be strings")

    return dialogues, was_batch


class APIInterface:
    """OpenAI-compatible HTTP interface with explicit, bounded retries."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_workers: int = 4,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        retry_total_timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
        **_: Any,
    ) -> None:
        load_dotenv()
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).strip()

        try:
            parsed_base_url = httpx.URL(self.base_url)
        except (TypeError, ValueError) as exc:
            raise ValueError("base_url must be a valid absolute HTTP(S) URL") from exc
        if (
            not parsed_base_url.is_absolute_url
            or parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.host
            or bool(parsed_base_url.username)
            or bool(parsed_base_url.password)
            or parsed_base_url.query
            or parsed_base_url.fragment
        ):
            raise ValueError(
                "base_url must be an absolute HTTP(S) URL without credentials, "
                "query parameters, or fragments"
            )
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if (
            parsed_base_url.scheme != "https"
            and parsed_base_url.host not in local_hosts
        ):
            raise ValueError(
                "base_url must use HTTPS except for a local loopback endpoint"
            )

        is_openrouter = parsed_base_url.host == "openrouter.ai" or str(
            parsed_base_url.host
        ).endswith(".openrouter.ai")

        # OPENAI_API_KEY is intentionally never consulted: sending it to an
        # OpenRouter or other compatible endpoint would leak the wrong secret.
        if api_key is None and not is_openrouter:
            raise ValueError(
                "A custom base_url requires an explicitly paired api_key; "
                "OPENROUTER_API_KEY is never forwarded to a non-OpenRouter host."
            )
        resolved_key = (
            api_key
            or (os.getenv("OPENROUTER_API_KEY") if is_openrouter else None)
            or ""
        ).strip()
        if not resolved_key:
            raise ValueError(
                "An OpenRouter API key is required. Pass api_key or set "
                "OPENROUTER_API_KEY; OPENAI_API_KEY is intentionally not used."
            )
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if (
            not isinstance(max_workers, int)
            or isinstance(max_workers, bool)
            or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        for name, numeric_value in (
            ("timeout", timeout),
            ("retry_base_delay", retry_base_delay),
            ("retry_max_delay", retry_max_delay),
            ("retry_total_timeout", retry_total_timeout),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(numeric_value)):
                raise ValueError(f"{name} must be finite")
        if timeout <= 0 or retry_total_timeout <= 0:
            raise ValueError("timeout and retry_total_timeout must be positive")
        if retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("retry delays must be non-negative")
        if retry_max_delay < retry_base_delay:
            raise ValueError("retry_max_delay must be at least retry_base_delay")

        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.request_timeout = float(timeout)
        self.retry_total_timeout = float(retry_total_timeout)
        self._closed = False
        self._completion_url = f"{self.base_url.rstrip('/')}/chat/completions"
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """Parse Retry-After in either delta-seconds or HTTP-date form."""

        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    return None
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {408, 409, 429} or status_code >= 500

    def _retry_delay(self, retry_index: int, response: httpx.Response | None) -> float:
        server_delay = (
            self._retry_after_seconds(response) if response is not None else None
        )
        base_delay = (
            server_delay
            if server_delay is not None
            else min(self.retry_max_delay, self.retry_base_delay * (2**retry_index))
        )
        # Small positive jitter avoids synchronized retry waves while keeping a
        # server-provided Retry-After as the minimum wait.
        jitter_ceiling = min(1.0, base_delay * 0.1)
        return base_delay + random.uniform(0.0, jitter_ceiling)

    def _chat_single(
        self,
        dialogue: Dialogue,
        temperature: float,
        max_tokens: int,
        seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._closed or self.client is None:
            raise RuntimeError("APIInterface is closed")

        protected = {"model", "messages", "temperature", "seed", "max_tokens"}
        duplicate = protected.intersection(kwargs)
        if duplicate:
            raise ValueError(
                "request kwargs cannot override: " + ", ".join(sorted(duplicate))
            )
        payload = {
            "model": self.model_name,
            "messages": dialogue,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            remaining = self.retry_total_timeout - (time.monotonic() - started_at)
            if remaining <= 0:
                raise httpx.TimeoutException("total API retry deadline exceeded")
            try:
                response = self.client.post(
                    self._completion_url,
                    json=payload,
                    timeout=min(self.request_timeout, remaining),
                )
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(attempt, None)
                remaining = self.retry_total_timeout - (time.monotonic() - started_at)
                if delay >= remaining:
                    raise
                time.sleep(delay)
                continue

            if (
                self._is_retryable_status(response.status_code)
                and attempt < self.max_retries
            ):
                delay = self._retry_delay(attempt, response)
                remaining = self.retry_total_timeout - (time.monotonic() - started_at)
                if delay >= remaining:
                    response.raise_for_status()
                time.sleep(delay)
                continue
            response.raise_for_status()
            break

        if response is None:  # defensive; every loop path either sets or raises
            raise RuntimeError("request completed without an HTTP response")
        try:
            completion = response.json()
            choice = completion["choices"][0]
            generated_text = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
            if not isinstance(generated_text, str):
                raise TypeError("message content must be a string")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "API response does not match the chat completions schema"
            ) from exc
        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
        }

    def chat(
        self,
        dialogue_history: DialogueInput,
        temperature: float = 0,
        max_tokens: int = 1000,
        seed: int = 42,
        show_progress: bool = True,
        desc: str = "Processing",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate responses concurrently while retaining input order."""

        dialogues, was_batch = _normalize_dialogues(dialogue_history)
        if not was_batch:
            return [
                self._chat_single(dialogues[0], temperature, max_tokens, seed, **kwargs)
            ]

        workers = min(self.max_workers, len(dialogues))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map yields results in input order even when requests
            # complete out of order.
            ordered_results = executor.map(
                lambda dialogue: self._chat_single(
                    dialogue, temperature, max_tokens, seed, **kwargs
                ),
                dialogues,
            )
            if show_progress:
                ordered_results = tqdm(
                    ordered_results,
                    total=len(dialogues),
                    desc=desc,
                    unit="dialogue",
                )
            return list(ordered_results)

    def close(self) -> None:
        """Close the underlying HTTP client; safe to call repeatedly."""

        if self._closed:
            return
        self._closed = True
        client, self.client = self.client, None
        if client is not None:
            client.close()

    def free_memory(self) -> None:
        """Backward-compatible cleanup alias used by existing runners."""

        self.close()

    def __enter__(self) -> APIInterface:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class UnifiedLLMInterface:
    """Uniform facade for API and local vLLM backends."""

    _API_KWARGS = {
        "api_key",
        "base_url",
        "timeout",
        "max_retries",
        "max_workers",
        "retry_base_delay",
        "retry_max_delay",
        "retry_total_timeout",
        "transport",
    }
    _VLLM_KWARGS = {
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "trust_remote_code",
        "revision",
        "tokenizer_revision",
        "code_revision",
        "dtype",
        "enforce_eager",
        "max_model_len",
        "max_num_seqs",
        "language_model_only",
        "enable_thinking",
    }

    def __init__(self, model_name: str, use_api: bool = False, **kwargs: Any) -> None:
        self.use_api = use_api
        self.model_name = model_name
        # Expose the configured template contract so experiment runners can
        # verify injected/factory-created backends before making model calls.
        self.enable_thinking = kwargs.get("enable_thinking")

        unknown_kwargs = set(kwargs).difference(self._API_KWARGS | self._VLLM_KWARGS)
        if unknown_kwargs:
            raise TypeError(
                "unsupported model-interface argument(s): "
                + ", ".join(sorted(unknown_kwargs))
            )

        if use_api:
            api_kwargs = {
                key: value for key, value in kwargs.items() if key in self._API_KWARGS
            }
            self.interface = APIInterface(model_name, **api_kwargs)
        else:
            if VLLMInterface is None or not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is unavailable. Install the repository's vLLM "
                    "dependencies on a supported GPU host."
                )
            vllm_kwargs = {
                key: value for key, value in kwargs.items() if key in self._VLLM_KWARGS
            }
            self.interface = VLLMInterface(model_name, **vllm_kwargs)
            self.interface.load_model()

    def chat(
        self,
        dialogue_history: DialogueInput,
        show_progress: bool = True,
        desc: str = "Processing",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return self.interface.chat(
            dialogue_history,
            show_progress=show_progress,
            desc=desc,
            **kwargs,
        )

    def chat_with_bounded_candidates(
        self,
        dialogue_history: DialogueInput,
        show_progress: bool = True,
        desc: str = "Processing",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.use_api:
            raise NotImplementedError(
                "chat_with_bounded_candidates is only available with the vLLM backend"
            )
        return self.interface.chat_with_bounded_candidates(
            dialogue_history,
            show_progress=show_progress,
            desc=desc,
            **kwargs,
        )

    def preflight_bounded_scoring(self) -> dict[int, str]:
        """Validate local bounded first-token scoring without inference."""

        if self.use_api:
            raise NotImplementedError(
                "preflight_bounded_scoring is only available with the vLLM backend"
            )
        return self.interface.preflight_bounded_scoring()

    def extract_thinking(self, response_text: str) -> str:
        if self.use_api:
            return response_text
        return self.interface.extract_thinking(response_text)

    def close(self) -> None:
        if self.interface is not None:
            close = getattr(self.interface, "close", None)
            if callable(close):
                close()
            else:
                self.interface.free_memory()
            self.interface = None

    def free_memory(self) -> None:
        """Backward-compatible cleanup alias used by BaseExperimentRunner."""

        self.close()

    def __enter__(self) -> UnifiedLLMInterface:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["APIInterface", "UnifiedLLMInterface"]
