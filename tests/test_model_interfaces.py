import json
import math
import time
from types import SimpleNamespace

import httpx
import pytest

import src.models.unified_llm_interface as unified_module
import src.models.vllm_interface as vllm_module
from src.models.unified_llm_interface import APIInterface, UnifiedLLMInterface
from src.models.vllm_interface import VLLMInterface, extract_thinking_process


def completion_json(text, finish_reason="stop"):
    return {
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": finish_reason,
            }
        ]
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reasoning</think>answer", "reasoning"),
        ("<think>reasoning", "reasoning"),
        ("reasoning</think>answer", "reasoning"),
    ],
)
def test_extract_thinking_handles_both_full_and_template_prefilled_tags(text, expected):
    assert extract_thinking_process(text) == expected


class CountingTransport(httpx.BaseTransport):
    def __init__(self, handler):
        self.inner = httpx.MockTransport(handler)
        self.close_calls = 0

    def handle_request(self, request):
        return self.inner.handle_request(request)

    def close(self):
        self.close_calls += 1
        self.inner.close()


def prepare_api(monkeypatch):
    monkeypatch.setattr(unified_module, "load_dotenv", lambda: False)


def test_api_requires_openrouter_key_and_never_falls_back_to_openai_key(monkeypatch):
    prepare_api(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        APIInterface("provider/model")


def test_api_never_forwards_openrouter_key_to_custom_host(monkeypatch):
    prepare_api(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-leak")

    with pytest.raises(ValueError, match="explicitly paired api_key"):
        APIInterface("provider/model", base_url="https://example.test/v1")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.test/v1",
        "https://user:password@example.test/v1",
        "https://example.test/v1?secret=value",
    ],
)
def test_api_rejects_unsafe_base_urls(base_url):
    with pytest.raises(ValueError, match="base_url"):
        APIInterface("provider/model", api_key="explicit", base_url=base_url)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": math.nan},
        {"retry_total_timeout": math.inf},
        {"retry_base_delay": math.nan},
        {"retry_max_delay": math.inf},
    ],
)
def test_api_rejects_nonfinite_timeouts_and_delays(kwargs):
    with pytest.raises(ValueError, match="finite"):
        APIInterface("provider/model", api_key="explicit", **kwargs)


def test_api_sends_explicit_key_and_openai_compatible_payload(monkeypatch):
    prepare_api(monkeypatch)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, request=request, json=completion_json("ok"))

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        timeout=12.5,
        max_retries=3,
        transport=httpx.MockTransport(handler),
    )

    result = interface.chat([{"role": "user", "content": "hello"}])[0]

    assert result["generated_text"] == "ok"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer router-secret"
    payload = json.loads(request.content)
    assert payload == {
        "model": "provider/model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1000,
    }
    assert interface.max_retries == 3
    assert interface.client.timeout.read == 12.5
    interface.close()


def test_api_retries_only_retryable_statuses_with_retry_after_and_jitter(monkeypatch):
    prepare_api(monkeypatch)
    statuses = iter((429, 500, 200))
    requests = []
    sleeps = []

    monkeypatch.setattr(unified_module.random, "uniform", lambda _a, _b: 0.25)
    monkeypatch.setattr(unified_module.time, "sleep", sleeps.append)

    def handler(request):
        requests.append(request)
        status = next(statuses)
        if status == 429:
            return httpx.Response(
                status,
                request=request,
                headers={"Retry-After": "2"},
                json={"error": "rate limited"},
            )
        if status == 500:
            return httpx.Response(status, request=request, json={"error": "server"})
        return httpx.Response(status, request=request, json=completion_json("ok"))

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_retries=2,
        retry_base_delay=0.5,
        transport=httpx.MockTransport(handler),
    )

    result = interface.chat([{"role": "user", "content": "hello"}])[0]

    assert result["generated_text"] == "ok"
    assert len(requests) == 3
    assert sleeps == pytest.approx([2.25, 1.25])
    interface.close()


def test_api_does_not_retry_non_retryable_http_errors(monkeypatch):
    prepare_api(monkeypatch)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, request=request, json={"error": "bad request"})

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_retries=5,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.HTTPStatusError):
        interface.chat([{"role": "user", "content": "hello"}])

    assert len(calls) == 1
    interface.close()


@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ReadError,
        httpx.RemoteProtocolError,
    ],
)
def test_api_retries_transient_transport_errors(monkeypatch, error_type):
    prepare_api(monkeypatch)
    calls = []
    sleeps = []
    monkeypatch.setattr(unified_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(unified_module.time, "sleep", sleeps.append)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise error_type("temporary network error", request=request)
        return httpx.Response(200, request=request, json=completion_json("ok"))

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_retries=1,
        retry_base_delay=0.5,
        transport=httpx.MockTransport(handler),
    )

    result = interface.chat([{"role": "user", "content": "hello"}])[0]

    assert result["generated_text"] == "ok"
    assert len(calls) == 2
    assert sleeps == [0.5]
    interface.close()


def test_api_respects_server_retry_after_beyond_local_backoff_cap(monkeypatch):
    prepare_api(monkeypatch)
    calls = []
    sleeps = []
    monkeypatch.setattr(unified_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(unified_module.time, "sleep", sleeps.append)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "60"},
                json={"error": "rate limited"},
            )
        return httpx.Response(200, request=request, json=completion_json("ok"))

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_retries=1,
        retry_max_delay=8.0,
        retry_total_timeout=120.0,
        transport=httpx.MockTransport(handler),
    )

    result = interface.chat([{"role": "user", "content": "hello"}])[0]

    assert result["generated_text"] == "ok"
    assert sleeps == [60.0]
    interface.close()


def test_api_retry_budget_is_finite(monkeypatch):
    prepare_api(monkeypatch)
    calls = []
    monkeypatch.setattr(unified_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(unified_module.time, "sleep", lambda _delay: None)

    def handler(request):
        calls.append(request)
        return httpx.Response(503, request=request, json={"error": "unavailable"})

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_retries=2,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.HTTPStatusError):
        interface.chat([{"role": "user", "content": "hello"}])

    assert len(calls) == 3
    interface.close()


def test_api_total_retry_deadline_prevents_an_unsafe_sleep(monkeypatch):
    prepare_api(monkeypatch)
    calls = []
    clock = [0.0]
    monkeypatch.setattr(unified_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        unified_module.time,
        "sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    monkeypatch.setattr(unified_module.random, "uniform", lambda _a, _b: 0.0)

    def handler(request):
        calls.append(request)
        return httpx.Response(503, request=request, json={"error": "unavailable"})

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_retries=5,
        retry_base_delay=0.5,
        retry_total_timeout=0.25,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.HTTPStatusError):
        interface.chat([{"role": "user", "content": "hello"}])

    assert len(calls) == 1
    assert clock[0] == 0.0
    interface.close()


def test_api_batch_is_concurrent_but_preserves_input_order(monkeypatch):
    prepare_api(monkeypatch)

    def delayed(request):
        payload = json.loads(request.content)
        value = int(payload["messages"][-1]["content"])
        time.sleep((3 - value) * 0.01)
        return httpx.Response(200, request=request, json=completion_json(str(value)))

    interface = APIInterface(
        "provider/model",
        api_key="router-secret",
        max_workers=3,
        transport=httpx.MockTransport(delayed),
    )
    dialogues = [[{"role": "user", "content": str(index)}] for index in range(3)]

    results = interface.chat(dialogues, show_progress=False)

    assert [item["generated_text"] for item in results] == ["0", "1", "2"]
    interface.close()


def test_api_close_is_idempotent_and_closed_client_cannot_be_used(monkeypatch):
    prepare_api(monkeypatch)
    transport = CountingTransport(
        lambda request: httpx.Response(200, request=request, json=completion_json("ok"))
    )
    interface = APIInterface(
        "provider/model", api_key="router-secret", transport=transport
    )

    interface.close()
    interface.close()

    assert transport.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        interface.chat([{"role": "user", "content": "hello"}])


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        normalized = text.strip().rstrip(".,!?").casefold()
        if normalized == "yes":
            return [11]
        if normalized == "no":
            return [22]
        return [100, 101]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        continue_final_message,
        **kwargs,
    ):
        assert tokenize is False
        assert continue_final_message is False
        del kwargs
        rendered = "".join(
            f"{message['role']}:{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            rendered += "assistant:"
        return rendered


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCompletion:
    def __init__(self, token_id=11, logprobs=None, text="Yes"):
        self.text = text
        self.finish_reason = "length"
        self.token_ids = [token_id]
        self.logprobs = logprobs


class FakeRequestOutput:
    def __init__(self, completion):
        self.outputs = [completion]


class CompatibleFakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def chat(
        self,
        messages,
        sampling_params=None,
        use_tqdm=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )
        return self.outputs


def make_vllm_interface(monkeypatch):
    monkeypatch.setattr(vllm_module, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(
        vllm_module, "SamplingParams", FakeSamplingParams, raising=False
    )
    interface = VLLMInterface("fake/model")
    interface.tokenizer = FakeTokenizer()
    return interface


def test_vllm_defaults_do_not_trust_remote_code(monkeypatch):
    interface = make_vllm_interface(monkeypatch)

    assert interface.trust_remote_code is False
    assert interface.enforce_eager is False


def test_vllm_load_pins_model_tokenizer_and_code_revisions(monkeypatch):
    monkeypatch.setattr(vllm_module, "VLLM_AVAILABLE", True)
    captured = {}

    class LoadableLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_tokenizer(self):
            return object()

    monkeypatch.setattr(vllm_module, "LLM", LoadableLLM, raising=False)
    interface = VLLMInterface(
        "fake/model",
        revision="model-commit",
        tokenizer_revision="tokenizer-commit",
        code_revision="code-commit",
        max_model_len=4096,
        max_num_seqs=16,
        language_model_only=True,
        enable_thinking=False,
    )

    interface.load_model()

    assert captured["model"] == "fake/model"
    assert captured["revision"] == "model-commit"
    assert captured["tokenizer_revision"] == "tokenizer-commit"
    assert captured["code_revision"] == "code-commit"
    assert captured["trust_remote_code"] is False
    assert captured["max_model_len"] == 4096
    assert captured["max_num_seqs"] == 16
    assert captured["language_model_only"] is True


def test_fresh_turn_scoring_requests_only_candidate_ids_and_first_token(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    interface.enable_thinking = False
    first_position = {
        11: SimpleNamespace(logprob=math.log(0.6), decoded_token=" Yes"),
        22: SimpleNamespace(logprob=math.log(0.2), decoded_token=" No"),
    }
    # A second position with the reverse distribution must never be read.
    second_position = {
        11: SimpleNamespace(logprob=math.log(0.01), decoded_token="Yes"),
        22: SimpleNamespace(logprob=math.log(0.99), decoded_token="No"),
    }
    fake_llm = CompatibleFakeLLM(
        [FakeRequestOutput(FakeCompletion(logprobs=[first_position, second_position]))]
    )
    interface.llm = fake_llm
    dialogue = [
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "analysis"},
            {"role": "user", "content": "Answer exactly Yes or No."},
        ]
    ]

    results = interface.chat_with_bounded_candidates(
        dialogue,
        max_tokens=1,
        logprobs=20,  # legacy runner argument is accepted but not forwarded
        show_progress=False,
    )

    call = fake_llm.calls[0]
    sampling_kwargs = call["sampling_params"].kwargs
    assert sampling_kwargs["max_tokens"] == 1
    assert sampling_kwargs["logprob_token_ids"] == [11, 22]
    assert "logprobs" not in sampling_kwargs
    assert "prompt_logprobs" not in sampling_kwargs
    assert call["use_tqdm"] is False
    assert call["add_generation_prompt"] is True
    assert call["continue_final_message"] is False
    assert call["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["messages"] == dialogue
    assert "enable_thinking" not in sampling_kwargs
    assert results[0]["valid"] is True
    assert results[0]["format_valid"] is True
    assert results[0]["sampled_choice"] == "Yes"
    assert results[0]["candidate_mass"] == pytest.approx(0.8)
    assert results[0]["residual_mass"] == pytest.approx(0.2)
    assert results[0]["probabilities"]["Yes"] == pytest.approx(0.75)
    assert results[0]["logprobs_raw"] == results[0]["label_logprobs"]


def test_fresh_turn_scoring_requires_user_last_before_inference(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    interface.llm = CompatibleFakeLLM([])

    with pytest.raises(ValueError, match="must end with a user message"):
        interface.chat_with_bounded_candidates(
            [[{"role": "assistant", "content": "analysis"}]],
            show_progress=False,
        )

    assert interface.llm.calls == []


def test_bounded_scoring_preflight_loads_once_caches_and_never_infers(monkeypatch):
    monkeypatch.setattr(vllm_module, "VLLM_AVAILABLE", True)
    events = {"loads": 0, "tokenizers": 0, "chat": 0, "generate": 0}
    sampling_calls = []

    class RecordingSamplingParams:
        def __init__(self, **kwargs):
            sampling_calls.append(kwargs)

    class PreflightLLM:
        def __init__(self, **kwargs):
            del kwargs
            events["loads"] += 1

        def get_tokenizer(self):
            events["tokenizers"] += 1
            return FakeTokenizer()

        def chat(
            self,
            messages,
            sampling_params=None,
            use_tqdm=True,
            add_generation_prompt=True,
            continue_final_message=False,
            chat_template_kwargs=None,
        ):
            del (
                messages,
                sampling_params,
                use_tqdm,
                add_generation_prompt,
                continue_final_message,
                chat_template_kwargs,
            )
            events["chat"] += 1
            raise AssertionError("preflight must not call chat")

        def generate(self, *args, **kwargs):
            del args, kwargs
            events["generate"] += 1
            raise AssertionError("preflight must not call generate")

    monkeypatch.setattr(vllm_module, "LLM", PreflightLLM, raising=False)
    monkeypatch.setattr(
        vllm_module, "SamplingParams", RecordingSamplingParams, raising=False
    )
    interface = VLLMInterface("fake/model")

    first = interface.preflight_bounded_scoring()
    first[11] = "No"
    first[999] = "Yes"
    second = interface.preflight_bounded_scoring()
    interface.load_model()

    assert second == {11: "Yes", 22: "No"}
    assert events == {"loads": 1, "tokenizers": 1, "chat": 0, "generate": 0}
    assert sampling_calls == [
        {
            "max_tokens": 1,
            "logprob_token_ids": [11, 22],
            "skip_special_tokens": False,
        }
    ]


def test_bounded_scoring_preflight_rejects_incompatible_sampling_params(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    interface.llm = CompatibleFakeLLM([])

    class OldSamplingParams:
        def __init__(self, max_tokens, skip_special_tokens):
            del max_tokens, skip_special_tokens

    monkeypatch.setattr(vllm_module, "SamplingParams", OldSamplingParams)

    with pytest.raises(RuntimeError, match="logprob_token_ids"):
        interface.preflight_bounded_scoring()
    assert interface.llm.calls == []


def test_bounded_scoring_preflight_rejects_template_that_drops_analysis(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    interface.llm = CompatibleFakeLLM([])

    class DroppingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return messages[-1]["content"]

    interface.tokenizer = DroppingTokenizer()

    with pytest.raises(RuntimeError, match="reconstruct phase-1 visible analysis"):
        interface.preflight_bounded_scoring()
    assert interface.llm.calls == []


def test_bounded_scoring_preflight_requires_fresh_assistant_prefix(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    interface.llm = CompatibleFakeLLM([])

    class NoFreshAssistantPrefixTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            continue_final_message,
            **kwargs,
        ):
            if len(messages) == 3:
                add_generation_prompt = False
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                **kwargs,
            )

    interface.tokenizer = NoFreshAssistantPrefixTokenizer()

    with pytest.raises(RuntimeError, match="fresh assistant generation prompt"):
        interface.preflight_bounded_scoring()
    assert interface.llm.calls == []


def test_vllm_dialogue_validates_optional_reasoning_content(monkeypatch):
    interface = make_vllm_interface(monkeypatch)

    with pytest.raises(TypeError, match="reasoning_content.*must be a string"):
        interface._normalize_and_validate_dialogues(
            [{"role": "assistant", "content": "Answer:", "reasoning_content": None}]
        )
    with pytest.raises(ValueError, match="only valid on assistant messages"):
        interface._normalize_and_validate_dialogues(
            [{"role": "user", "content": "question", "reasoning_content": "x"}]
        )


def test_unified_bounded_scoring_forwards_locally_and_rejects_api():
    local = object.__new__(UnifiedLLMInterface)
    local.use_api = False
    local.interface = SimpleNamespace(
        preflight_bounded_scoring=lambda: {11: "Yes", 22: "No"}
    )
    assert local.preflight_bounded_scoring() == {11: "Yes", 22: "No"}

    api = object.__new__(UnifiedLLMInterface)
    api.use_api = True
    api.interface = object()
    with pytest.raises(NotImplementedError, match="only available with the vLLM"):
        api.preflight_bounded_scoring()


def test_bounded_scoring_missing_candidate_is_explicitly_invalid(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    fake_llm = CompatibleFakeLLM(
        [
            FakeRequestOutput(
                FakeCompletion(
                    logprobs=[
                        {
                            11: SimpleNamespace(
                                logprob=math.log(0.6), decoded_token="Yes"
                            )
                        }
                    ]
                )
            )
        ]
    )
    interface.llm = fake_llm

    result = interface.chat_with_bounded_candidates(
        [[{"role": "user", "content": "Answer exactly Yes or No."}]],
        show_progress=False,
    )[0]

    assert result["valid"] is False
    assert result["error"] == "missing_candidate_logprobs:22"
    assert result["probabilities"] is None
    assert result["logprobs_raw"] == {}


def test_bounded_scoring_api_mismatch_fails_instead_of_falling_back(monkeypatch):
    interface = make_vllm_interface(monkeypatch)

    class OldLLM:
        def chat(self, messages, sampling_params=None):
            raise AssertionError("old standard-chat fallback must not run")

    interface.llm = OldLLM()

    with pytest.raises(RuntimeError, match="incompatible with bounded scoring"):
        interface.chat_with_bounded_candidates(
            [[{"role": "user", "content": "Answer exactly Yes or No."}]],
            show_progress=False,
        )


def test_bounded_scoring_preflight_requires_chat_template_kwargs(monkeypatch):
    interface = make_vllm_interface(monkeypatch)

    class LLMWithoutTemplateKwargs:
        def chat(
            self,
            messages,
            sampling_params=None,
            use_tqdm=True,
            add_generation_prompt=True,
            continue_final_message=False,
        ):
            del (
                messages,
                sampling_params,
                use_tqdm,
                add_generation_prompt,
                continue_final_message,
            )

    interface.llm = LLMWithoutTemplateKwargs()

    with pytest.raises(RuntimeError, match="chat_template_kwargs"):
        interface.preflight_bounded_scoring()


def test_normal_vllm_chat_delegates_progress_to_engine(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    interface.enable_thinking = False
    fake_llm = CompatibleFakeLLM(
        [FakeRequestOutput(FakeCompletion(logprobs=None, text="answer"))]
    )
    interface.llm = fake_llm

    result = interface.chat(
        [[{"role": "user", "content": "question"}]],
        show_progress=False,
    )

    assert result[0]["generated_text"] == "answer"
    assert fake_llm.calls[0]["use_tqdm"] is False
    assert fake_llm.calls[0]["add_generation_prompt"] is True
    assert fake_llm.calls[0]["continue_final_message"] is False
    assert fake_llm.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "logprob_token_ids" not in fake_llm.calls[0]["sampling_params"].kwargs
    assert "enable_thinking" not in fake_llm.calls[0]["sampling_params"].kwargs


def test_unified_interface_rejects_unknown_arguments_before_backend_creation():
    with pytest.raises(
        TypeError, match="unsupported model-interface.*max_model_lenght"
    ):
        UnifiedLLMInterface("fake/model", max_model_lenght=4096)


def test_vllm_cleanup_detaches_owner_reference_before_release(monkeypatch):
    interface = make_vllm_interface(monkeypatch)
    events = []

    class Model:
        def close(self):
            events.append(("close", interface.llm, interface.tokenizer))

    model = Model()
    interface.llm = model

    def observe_release():
        events.append(("cleanup", interface.llm, interface.tokenizer))

    monkeypatch.setattr(vllm_module, "free_gpu_memory", observe_release)

    interface.free_memory()

    assert events == [("close", None, None), ("cleanup", None, None)]
    assert interface._yes_no_candidate_map is None
