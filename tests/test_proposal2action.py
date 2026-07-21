from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from data import proposal2action as p2a


PROPOSALS = {
    "Category A": ["Proposal one", "Proposal two"],
    "Category B": ["Proposal three"],
}
PROMPT = "Create actions for {POLICY_PROPOSAL}"


def valid_result(proposal: str) -> dict:
    return {
        "political_proposal": proposal,
        "actions": [
            {
                "action_type": action_type,
                "action_description": f"Concrete action for {action_type}",
            }
            for action_type in p2a.ACTION_TYPES
        ],
    }


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def chat(self, dialogue, **kwargs):
        self.calls.append((dialogue, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, dict) and "generated_text" in response:
            return [response]
        return [{"generated_text": json.dumps(response)}]

    def close(self):
        self.closed = True


def write_inputs(tmp_path: Path, proposals=PROPOSALS):
    policy_path = tmp_path / "policy_options.json"
    prompt_path = tmp_path / "prompt.txt"
    policy_path.write_text(json.dumps(proposals), encoding="utf-8")
    prompt_path.write_text(PROMPT, encoding="utf-8")
    return policy_path, prompt_path


def cli_args(policy_path: Path, prompt_path: Path, output: Path, *extra: str):
    return [
        "--model",
        "test/model",
        "--policy-options",
        str(policy_path),
        "--prompt-template",
        str(prompt_path),
        "--output",
        str(output),
        "--delay",
        "0",
        *extra,
    ]


def install_fake_factory(monkeypatch, apis):
    queue = list(apis)
    models = []

    def factory(model, **_kwargs):
        models.append(model)
        return queue.pop(0)

    monkeypatch.setattr(p2a, "create_api_interface", factory)
    return models


def test_output_is_required_without_initializing_api(monkeypatch):
    monkeypatch.setattr(
        p2a,
        "create_api_interface",
        lambda model, **kwargs: pytest.fail(
            "API must not initialize during argument parsing"
        ),
    )
    with pytest.raises(SystemExit) as exc_info:
        p2a.main(["--model", "test/model"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_delay_must_be_finite(value):
    with pytest.raises(SystemExit) as exc_info:
        p2a.build_parser().parse_args(["--output", "generated.json", "--delay", value])
    assert exc_info.value.code == 2


def test_main_uses_model_writes_output_and_closes_api(tmp_path, monkeypatch):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One"]})
    output = tmp_path / "generated.json"
    fake = FakeAPI([valid_result("One")])
    models = install_fake_factory(monkeypatch, [fake])

    status = p2a.main(cli_args(policy_path, prompt_path, output))

    assert status == 0
    assert models == ["test/model"]
    assert fake.closed is True
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "Only": [valid_result("One")]
    }
    assert not p2a.partial_path_for(output).exists()


def test_debug_cannot_target_tracked_canonical(monkeypatch):
    initialized = False

    def factory(model, **_kwargs):
        nonlocal initialized
        initialized = True
        return FakeAPI([])

    monkeypatch.setattr(p2a, "create_api_interface", factory)
    status = p2a.main(
        [
            "--output",
            str(p2a.CANONICAL_OUTPUT),
            "--debug",
            "1",
            "--overwrite",
        ]
    )
    assert status != 0
    assert initialized is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update(extra=True),
        lambda result: result.update(political_proposal="different"),
        lambda result: result.update(actions=result["actions"][:2]),
        lambda result: result["actions"][0].update(extra=True),
        lambda result: result["actions"][0].update(action_description="  "),
        lambda result: result["actions"][1].update(action_type="Personal Commitment"),
    ],
)
def test_strict_result_validation_rejects_malformed_schema(mutation):
    result = valid_result("Proposal")
    mutation(result)
    with pytest.raises(p2a.ResponseValidationError):
        p2a.validate_action_result(result, "Proposal")


def test_parser_rejects_markdown_wrapped_json():
    response = f"```json\n{json.dumps(valid_result('Proposal'))}\n```"
    with pytest.raises(p2a.ResponseValidationError):
        p2a.parse_llm_response(response, "Proposal")


@pytest.mark.parametrize(
    "response",
    [
        '{"political_proposal":"Proposal","political_proposal":"Other","actions":[]}',
        '{"political_proposal":"Proposal","actions":NaN}',
    ],
)
def test_parser_rejects_duplicate_keys_and_nonstandard_constants(response):
    with pytest.raises(p2a.ResponseValidationError):
        p2a.parse_llm_response(response, "Proposal")


def test_failure_keeps_only_valid_partial_and_never_publishes_final(
    tmp_path, monkeypatch
):
    policy_path, prompt_path = write_inputs(tmp_path)
    output = tmp_path / "generated.json"
    fake = FakeAPI([valid_result("Proposal one"), {"generated_text": "not JSON"}])
    install_fake_factory(monkeypatch, [fake])

    status = p2a.main(cli_args(policy_path, prompt_path, output))

    assert status != 0
    assert fake.closed is True
    assert not output.exists()
    partial = json.loads(p2a.partial_path_for(output).read_text(encoding="utf-8"))
    assert len(partial["completed"]) == 1
    assert partial["completed"][0]["result"] == valid_result("Proposal one")


def test_failed_full_run_preserves_existing_canonical_data(tmp_path, monkeypatch):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One", "Two"]})
    canonical = tmp_path / "proposal_actions.json"
    original = {"tracked": "unchanged"}
    canonical.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(p2a, "CANONICAL_OUTPUT", canonical)
    monkeypatch.setattr(p2a, "DEFAULT_POLICY_OPTIONS", policy_path)
    monkeypatch.setattr(p2a, "DEFAULT_PROMPT_TEMPLATE", prompt_path)
    fake = FakeAPI([valid_result("One"), {"generated_text": "not JSON"}])
    install_fake_factory(monkeypatch, [fake])

    status = p2a.main(cli_args(policy_path, prompt_path, canonical, "--overwrite"))

    assert status != 0
    assert fake.closed is True
    assert json.loads(canonical.read_text(encoding="utf-8")) == original
    partial = json.loads(p2a.partial_path_for(canonical).read_text(encoding="utf-8"))
    assert [
        entry["result"]["political_proposal"] for entry in partial["completed"]
    ] == ["One"]


def test_resume_skips_completed_prefix_and_atomically_finishes(tmp_path, monkeypatch):
    policy_path, prompt_path = write_inputs(tmp_path)
    output = tmp_path / "generated.json"
    first_api = FakeAPI([valid_result("Proposal one"), {"generated_text": "not JSON"}])
    second_api = FakeAPI([valid_result("Proposal two"), valid_result("Proposal three")])
    install_fake_factory(monkeypatch, [first_api, second_api])

    first_status = p2a.main(cli_args(policy_path, prompt_path, output))
    second_status = p2a.main(cli_args(policy_path, prompt_path, output, "--resume"))

    assert first_status != 0
    assert second_status == 0
    assert first_api.closed and second_api.closed
    assert len(first_api.calls) == 2
    assert len(second_api.calls) == 2
    assert not p2a.partial_path_for(output).exists()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "Category A": [
            valid_result("Proposal one"),
            valid_result("Proposal two"),
        ],
        "Category B": [valid_result("Proposal three")],
    }


def test_resume_rejects_changed_model_without_new_api_call(tmp_path, monkeypatch):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One", "Two"]})
    output = tmp_path / "generated.json"
    first_api = FakeAPI([valid_result("One"), RuntimeError("stop")])
    models = install_fake_factory(monkeypatch, [first_api, FakeAPI([])])
    assert p2a.main(cli_args(policy_path, prompt_path, output)) != 0

    args = cli_args(policy_path, prompt_path, output, "--resume")
    args[1] = "different/model"
    assert p2a.main(args) != 0
    assert models == ["test/model"]


def test_resume_rejects_changed_generation_regime_without_new_api_call(
    tmp_path, monkeypatch
):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One", "Two"]})
    output = tmp_path / "generated.json"
    first_api = FakeAPI([valid_result("One"), RuntimeError("stop")])
    models = install_fake_factory(monkeypatch, [first_api, FakeAPI([])])
    assert p2a.main(cli_args(policy_path, prompt_path, output)) != 0

    monkeypatch.setattr(p2a, "GENERATION_TEMPERATURE", 0.2)
    assert p2a.main(cli_args(policy_path, prompt_path, output, "--resume")) != 0
    assert models == ["test/model"]


def test_resume_rejects_changed_openrouter_endpoint_without_new_api_call(
    tmp_path, monkeypatch
):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One", "Two"]})
    output = tmp_path / "generated.json"
    first_api = FakeAPI([valid_result("One"), RuntimeError("stop")])
    models = install_fake_factory(monkeypatch, [first_api, FakeAPI([])])
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    assert p2a.main(cli_args(policy_path, prompt_path, output)) != 0

    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v2")
    assert p2a.main(cli_args(policy_path, prompt_path, output, "--resume")) != 0
    assert models == ["test/model"]


def test_existing_output_requires_explicit_overwrite(tmp_path, monkeypatch):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One"]})
    output = tmp_path / "generated.json"
    output.write_text('{"existing": true}', encoding="utf-8")
    monkeypatch.setattr(
        p2a,
        "create_api_interface",
        lambda model, **kwargs: pytest.fail(
            "API must not initialize for unsafe output"
        ),
    )

    assert p2a.main(cli_args(policy_path, prompt_path, output)) != 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"existing": True}


def test_output_and_partial_symlinks_are_rejected_before_api_initialization(
    tmp_path, monkeypatch
):
    policy_path, prompt_path = write_inputs(tmp_path, {"Only": ["One"]})
    external = tmp_path / "external.json"
    external.write_text('{"unchanged": true}', encoding="utf-8")
    output_link = tmp_path / "output-link.json"
    try:
        output_link.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    monkeypatch.setattr(
        p2a,
        "create_api_interface",
        lambda model, **kwargs: pytest.fail("API must not initialize for a symlink"),
    )
    assert p2a.main(cli_args(policy_path, prompt_path, output_link, "--overwrite")) != 0
    assert json.loads(external.read_text(encoding="utf-8")) == {"unchanged": True}

    output = tmp_path / "generated.json"
    partial_link = p2a.partial_path_for(output)
    partial_link.symlink_to(external)
    assert p2a.main(cli_args(policy_path, prompt_path, output, "--overwrite")) != 0
    assert json.loads(external.read_text(encoding="utf-8")) == {"unchanged": True}


def test_atomic_write_uses_same_directory_fsync_and_replace(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "output.json"
    calls = []
    original_replace = os.replace
    original_fsync = os.fsync

    def replace(source, target):
        calls.append((Path(source), Path(target)))
        original_replace(source, target)

    fsync_count = 0

    def fsync(descriptor):
        nonlocal fsync_count
        fsync_count += 1
        original_fsync(descriptor)

    monkeypatch.setattr(p2a.os, "replace", replace)
    monkeypatch.setattr(p2a.os, "fsync", fsync)
    p2a.atomic_write_json({"ok": True}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"ok": True}
    assert calls and calls[0][0].parent == destination.parent
    assert calls[0][1] == destination
    assert fsync_count >= 1
    assert not list(destination.parent.glob("*.tmp"))


def test_client_fallback_is_closed_when_interface_has_no_close():
    class Client:
        closed = False

        def close(self):
            self.closed = True

    class Interface:
        client = Client()

    interface = Interface()
    p2a.close_api_interface(interface)
    assert interface.client.closed is True
