"""Generate validated proposal-to-action data through an OpenAI-compatible API.

The final output is published only after every requested proposal has produced a
valid result. Valid progress is saved to an adjacent ``.partial.json`` file and
can be continued with ``--resume``.

Run from the repository root so package imports resolve normally::

    python -m data.proposal2action \
        --model google/gemini-3-pro-preview \
        --output results/proposal_actions.generated.json

The tracked ``data/proposal_actions.json`` file is deliberately protected from
debug runs and from implicit overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_OPTIONS = REPOSITORY_ROOT / "data" / "policy_options.json"
DEFAULT_PROMPT_TEMPLATE = REPOSITORY_ROOT / "data" / "proposal2action.txt"
CANONICAL_OUTPUT = REPOSITORY_ROOT / "data" / "proposal_actions.json"

ACTION_TYPES = (
    "Personal Commitment",
    "Public Advocacy",
    "Strategic Support",
)
PARTIAL_SCHEMA_VERSION = 2
GENERATION_TEMPERATURE = 0.7
GENERATION_SEED = 42
GENERATION_MAX_TOKENS = 2000
PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
REQUIRED_PROMPT_PLACEHOLDERS = frozenset({"POLICY_PROPOSAL"})


class ProposalActionError(RuntimeError):
    """Base class for expected proposal-to-action failures."""


class ResponseValidationError(ProposalActionError):
    """Raised when an LLM response does not match the required schema."""


class ResumeValidationError(ProposalActionError):
    """Raised when a partial checkpoint cannot safely be resumed."""


class _DuplicateJsonKey(ValueError):
    pass


class _UnpairedUnicodeSurrogate(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _reject_unpaired_unicode_surrogates(value: Any, path: str = "$") -> None:
    """Reject surrogate code points left after strict JSON decoding."""
    if isinstance(value, str):
        if any("\ud800" <= character <= "\udfff" for character in value):
            raise _UnpairedUnicodeSurrogate(f"unpaired Unicode surrogate at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_unpaired_unicode_surrogates(key, f"{path}.<key>")
            _reject_unpaired_unicode_surrogates(child, f"{path}[{key!r}]")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_unpaired_unicode_surrogates(child, f"{path}[{index}]")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without initializing an API client."""
    parser = argparse.ArgumentParser(
        description="Convert political proposals into three validated actions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Final JSON path. Existing files require --overwrite.",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-3-pro-preview",
        help="OpenRouter/OpenAI-compatible model identifier.",
    )
    parser.add_argument(
        "--policy-options",
        type=Path,
        default=DEFAULT_POLICY_OPTIONS,
        help="Policy proposal JSON file.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template containing {POLICY_PROPOSAL}.",
    )
    parser.add_argument(
        "--delay",
        type=_non_negative_float,
        default=0.5,
        help="Delay in seconds between successful API calls (default: 0.5).",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=3,
        type=_positive_int,
        metavar="N",
        help="Process only the first N proposals (default: 3).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the output's adjacent .partial.json checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing final output or stale checkpoint.",
    )
    return parser


def load_json_file(file_path: Path) -> Any:
    """Load UTF-8 JSON from ``file_path``."""
    with file_path.open("r", encoding="utf-8") as handle:
        parsed = json.load(
            handle,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    try:
        _reject_unpaired_unicode_surrogates(parsed)
    except _UnpairedUnicodeSurrogate as exc:
        raise ProposalActionError(f"JSON input contains an {exc}: {file_path}") from exc
    return parsed


def load_prompt_template(file_path: Path) -> str:
    """Load and validate the proposal prompt template."""
    template = file_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ProposalActionError(f"Prompt template is empty: {file_path}")
    placeholders = frozenset(PROMPT_PLACEHOLDER_PATTERN.findall(template))
    if placeholders != REQUIRED_PROMPT_PLACEHOLDERS:
        missing = sorted(REQUIRED_PROMPT_PLACEHOLDERS - placeholders)
        unknown = sorted(placeholders - REQUIRED_PROMPT_PLACEHOLDERS)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ProposalActionError(
            "Prompt template placeholders must be exactly {POLICY_PROPOSAL} "
            f"({'; '.join(details)}): {file_path}"
        )
    return template


def validate_policy_options(data: Any) -> dict[str, list[str]]:
    """Validate the policy input and return a normalized copy."""
    if not isinstance(data, dict) or not data:
        raise ProposalActionError("Policy options must be a non-empty JSON object.")

    normalized: dict[str, list[str]] = {}
    for category, proposals in data.items():
        if not isinstance(category, str) or not category.strip():
            raise ProposalActionError(
                "Every policy category must be a non-empty string."
            )
        if not isinstance(proposals, list) or not proposals:
            raise ProposalActionError(
                f"Policy category {category!r} must contain a non-empty list."
            )

        clean_proposals: list[str] = []
        seen: set[str] = set()
        for proposal in proposals:
            if not isinstance(proposal, str) or not proposal.strip():
                raise ProposalActionError(
                    f"Every proposal in {category!r} must be a non-empty string."
                )
            if proposal in seen:
                raise ProposalActionError(
                    f"Duplicate proposal in category {category!r}: {proposal!r}"
                )
            seen.add(proposal)
            clean_proposals.append(proposal)
        normalized[category] = clean_proposals

    return normalized


def build_plan(
    policy_options: Mapping[str, Sequence[str]], max_items: int | None = None
) -> list[tuple[str, str]]:
    """Build the deterministic ordered proposal plan."""
    plan = [
        (category, proposal)
        for category, proposals in policy_options.items()
        for proposal in proposals
    ]
    return plan if max_items is None else plan[:max_items]


def validate_action_result(data: Any, expected_proposal: str) -> dict[str, Any]:
    """Validate one response and normalize action ordering.

    The schema is intentionally strict: no extra fields are accepted, all three
    action types must occur exactly once, and every description must be non-empty.
    """
    try:
        _reject_unpaired_unicode_surrogates(data)
        _reject_unpaired_unicode_surrogates(expected_proposal, "expected_proposal")
    except _UnpairedUnicodeSurrogate as exc:
        raise ResponseValidationError(
            f"The action response contains an {exc}."
        ) from exc
    if not isinstance(expected_proposal, str) or not expected_proposal.strip():
        raise ResponseValidationError(
            "The expected proposal must be a non-empty string."
        )
    if not isinstance(data, dict):
        raise ResponseValidationError("The response must be a JSON object.")
    if set(data) != {"political_proposal", "actions"}:
        raise ResponseValidationError(
            "The response must contain exactly political_proposal and actions."
        )
    if (
        not isinstance(data["political_proposal"], str)
        or data["political_proposal"] != expected_proposal
    ):
        raise ResponseValidationError(
            "The response political_proposal does not exactly match the input."
        )

    actions = data["actions"]
    if not isinstance(actions, list) or len(actions) != len(ACTION_TYPES):
        raise ResponseValidationError(
            f"actions must contain exactly {len(ACTION_TYPES)} entries."
        )

    by_type: dict[str, dict[str, str]] = {}
    for action in actions:
        if not isinstance(action, dict):
            raise ResponseValidationError("Every action must be a JSON object.")
        if set(action) != {"action_type", "action_description"}:
            raise ResponseValidationError(
                "Every action must contain exactly action_type and action_description."
            )
        action_type = action["action_type"]
        description = action["action_description"]
        if action_type not in ACTION_TYPES:
            raise ResponseValidationError(f"Unknown action type: {action_type!r}")
        if action_type in by_type:
            raise ResponseValidationError(f"Duplicate action type: {action_type!r}")
        if not isinstance(description, str) or not description.strip():
            raise ResponseValidationError(
                f"Action description for {action_type!r} must be non-empty."
            )
        by_type[action_type] = {
            "action_type": action_type,
            "action_description": description.strip(),
        }

    missing = set(ACTION_TYPES) - set(by_type)
    if missing:
        raise ResponseValidationError(
            f"Missing required action types: {', '.join(sorted(missing))}"
        )

    return {
        "political_proposal": expected_proposal,
        "actions": [by_type[action_type] for action_type in ACTION_TYPES],
    }


def parse_llm_response(response_content: str, expected_proposal: str) -> dict[str, Any]:
    """Parse a strict JSON-only model response and validate its schema."""
    if not isinstance(response_content, str) or not response_content.strip():
        raise ResponseValidationError("The model returned an empty response.")
    try:
        parsed = json.loads(
            response_content,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJsonKey, ValueError) as exc:
        raise ResponseValidationError(
            f"The model response is not valid standalone JSON: {exc}"
        ) from exc
    return validate_action_result(parsed, expected_proposal)


def convert_proposal_to_action(
    policy_proposal: str,
    prompt_template: str,
    model: str,
    api_interface: Any,
    debug: bool = False,
) -> dict[str, Any]:
    """Generate and strictly validate the actions for one proposal."""
    user_message = prompt_template.replace("{POLICY_PROPOSAL}", policy_proposal)
    responses = api_interface.chat(
        [{"role": "user", "content": user_message}],
        temperature=GENERATION_TEMPERATURE,
        seed=GENERATION_SEED,
        max_tokens=GENERATION_MAX_TOKENS,
        show_progress=False,
    )
    if not isinstance(responses, list) or len(responses) != 1:
        raise ProposalActionError(
            f"Model {model!r} returned an invalid response container."
        )
    response = responses[0]
    if not isinstance(response, dict):
        raise ProposalActionError(f"Model {model!r} returned a non-object response.")
    if response.get("error"):
        raise ProposalActionError(
            f"Model {model!r} request failed: {response['error']}"
        )

    response_content = response.get("generated_text")
    if debug and isinstance(response_content, str):
        print(f"    Raw response (first 500 chars): {response_content[:500]}")
    return parse_llm_response(response_content, policy_proposal)


def partial_path_for(output_path: Path) -> Path:
    """Return ``name.partial.json`` next to a final JSON output."""
    suffix = output_path.suffix or ".json"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.partial{suffix}")


def _canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def generator_source_sha256() -> str:
    """Hash this implementation without persisting a machine-specific path."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def model_interface_source_sha256() -> str:
    """Hash the HTTP adapter used by the generator."""

    interface_path = REPOSITORY_ROOT / "src" / "models" / "unified_llm_interface.py"
    return hashlib.sha256(interface_path.read_bytes()).hexdigest()


def resolve_openrouter_base_url() -> str:
    """Resolve and validate the non-secret endpoint included in resume identity."""

    for dotenv_path in dict.fromkeys((Path.cwd() / ".env", REPOSITORY_ROOT / ".env")):
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)
    base_url = (
        (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1")
        .strip()
        .rstrip("/")
    )
    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not (host == "openrouter.ai" or host.endswith(".openrouter.ai"))
    ):
        raise ProposalActionError(
            "OPENROUTER_BASE_URL must be a credential-free OpenRouter HTTPS endpoint"
        )
    return base_url


def build_run_fingerprint(
    *,
    model: str,
    policy_options: Mapping[str, Sequence[str]],
    prompt_template: str,
    plan: Sequence[tuple[str, str]],
    api_base_url: str,
) -> str:
    """Fingerprint all inputs that affect resumable generation."""
    payload = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "model": model,
        "generation": {
            "temperature": GENERATION_TEMPERATURE,
            "seed": GENERATION_SEED,
            "max_tokens": GENERATION_MAX_TOKENS,
        },
        "generator_source_sha256": generator_source_sha256(),
        "model_interface_source_sha256": model_interface_source_sha256(),
        "api_base_url": api_base_url,
        "policy_options": policy_options,
        "prompt_template_sha256": hashlib.sha256(
            prompt_template.encode("utf-8")
        ).hexdigest(),
        "plan": list(plan),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def atomic_write_json(data: Any, file_path: Path) -> None:
    """Atomically write JSON using a same-directory temporary file and fsync."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.", suffix=".tmp", dir=file_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, file_path)
        _fsync_directory(file_path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory after a replace/unlink operation."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_partial(fingerprint: str, model: str) -> dict[str, Any]:
    return {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "model": model,
        "completed": [],
    }


def validate_partial(
    data: Any,
    *,
    fingerprint: str,
    model: str,
    plan: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Validate checkpoint identity, schema, and completed prefix ordering."""
    if not isinstance(data, dict) or set(data) != {
        "schema_version",
        "run_fingerprint",
        "model",
        "completed",
    }:
        raise ResumeValidationError("Partial checkpoint has an invalid schema.")
    if data["schema_version"] != PARTIAL_SCHEMA_VERSION:
        raise ResumeValidationError("Partial checkpoint schema version is unsupported.")
    if data["run_fingerprint"] != fingerprint or data["model"] != model:
        raise ResumeValidationError(
            "Partial checkpoint does not match the current model/input/prompt/plan."
        )
    completed = data["completed"]
    if not isinstance(completed, list) or len(completed) > len(plan):
        raise ResumeValidationError("Partial checkpoint has an invalid completed list.")

    normalized_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(completed):
        if not isinstance(entry, dict) or set(entry) != {"category", "result"}:
            raise ResumeValidationError(
                f"Partial checkpoint entry {index} has an invalid schema."
            )
        expected_category, expected_proposal = plan[index]
        if entry["category"] != expected_category:
            raise ResumeValidationError(
                f"Partial checkpoint entry {index} is not the expected plan prefix."
            )
        try:
            result = validate_action_result(entry["result"], expected_proposal)
        except ResponseValidationError as exc:
            raise ResumeValidationError(
                f"Partial checkpoint entry {index} is invalid: {exc}"
            ) from exc
        normalized_entries.append({"category": expected_category, "result": result})

    return {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "model": model,
        "completed": normalized_entries,
    }


def build_final_results(
    completed: Sequence[Mapping[str, Any]], plan: Sequence[tuple[str, str]]
) -> dict[str, list[dict[str, Any]]]:
    """Build and fully validate the public output schema from completed entries."""
    if len(completed) != len(plan):
        raise ProposalActionError(
            f"Cannot publish an incomplete run ({len(completed)}/{len(plan)} complete)."
        )

    results: dict[str, list[dict[str, Any]]] = {}
    for index, (category, proposal) in enumerate(plan):
        entry = completed[index]
        if entry.get("category") != category:
            raise ProposalActionError(
                "Completed results are not aligned with the plan."
            )
        result = validate_action_result(entry.get("result"), proposal)
        results.setdefault(category, []).append(result)

    validate_final_results(results, plan)
    return results


def validate_final_results(results: Any, plan: Sequence[tuple[str, str]]) -> None:
    """Validate that final results match every planned proposal exactly once."""
    if not isinstance(results, dict):
        raise ProposalActionError("Final results must be a JSON object.")

    flattened: list[tuple[str, str]] = []
    for category, entries in results.items():
        if not isinstance(category, str) or not isinstance(entries, list):
            raise ProposalActionError("Final results contain an invalid category.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProposalActionError("Final results contain a non-object entry.")
            proposal = entry.get("political_proposal")
            validate_action_result(entry, proposal)
            flattened.append((category, proposal))
    if flattened != list(plan):
        raise ProposalActionError(
            "Final results do not exactly match the generation plan."
        )


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ensure_output_safety(
    *,
    output_path: Path,
    partial_path: Path,
    policy_options_path: Path,
    prompt_template_path: Path,
    debug_limit: int | None,
    resume: bool,
    overwrite: bool,
) -> None:
    """Reject ambiguous or potentially destructive output configurations."""
    output_resolved = _resolved(output_path)
    canonical_resolved = _resolved(CANONICAL_OUTPUT)
    policy_resolved = _resolved(policy_options_path)
    prompt_resolved = _resolved(prompt_template_path)
    partial_resolved = _resolved(partial_path)

    if output_resolved in {policy_resolved, prompt_resolved}:
        raise ProposalActionError("Output must not replace an input or prompt file.")
    if partial_resolved in {
        output_resolved,
        policy_resolved,
        prompt_resolved,
        canonical_resolved,
    }:
        raise ProposalActionError(
            "Partial checkpoint must not replace an input, prompt, final output, "
            "or tracked canonical data file."
        )
    if output_path.is_symlink():
        raise ProposalActionError("Refusing to replace a symbolic-link output.")
    if partial_path.is_symlink():
        raise ProposalActionError("Refusing to use a symbolic-link partial checkpoint.")
    if output_path.exists() and not output_path.is_file():
        raise ProposalActionError("Output path must be a regular file when it exists.")
    if partial_path.exists() and not partial_path.is_file():
        raise ProposalActionError(
            "Partial checkpoint path must be a regular file when it exists."
        )
    if output_path.exists() and not overwrite:
        raise ProposalActionError(
            f"Output already exists; pass --overwrite to replace it: {output_path}"
        )
    if resume and not partial_path.exists():
        raise ResumeValidationError(
            f"--resume requires an existing partial checkpoint: {partial_path}"
        )
    if not resume and partial_path.exists() and not overwrite:
        raise ProposalActionError(
            "A partial checkpoint already exists; use --resume or --overwrite: "
            f"{partial_path}"
        )

    if output_resolved == canonical_resolved:
        if debug_limit is not None:
            raise ProposalActionError(
                "Debug mode is forbidden for tracked data/proposal_actions.json. "
                "Choose a non-canonical --output path."
            )
        if policy_resolved != _resolved(DEFAULT_POLICY_OPTIONS):
            raise ProposalActionError(
                "Canonical output requires the canonical policy_options.json input."
            )
        if prompt_resolved != _resolved(DEFAULT_PROMPT_TEMPLATE):
            raise ProposalActionError(
                "Canonical output requires the canonical proposal2action.txt prompt."
            )


def create_api_interface(model: str, *, base_url: str) -> Any:
    """Create the API backend lazily so offline tooling can import this module."""
    from src.models.unified_llm_interface import APIInterface

    return APIInterface(model_name=model, base_url=base_url)


def close_api_interface(api_interface: Any) -> None:
    """Close an API interface (or its underlying SDK client) exactly once."""
    close = getattr(api_interface, "close", None)
    if callable(close):
        close()
        return
    client = getattr(api_interface, "client", None)
    client_close = getattr(client, "close", None)
    if callable(client_close):
        client_close()


def process_plan(
    *,
    plan: Sequence[tuple[str, str]],
    policy_options: Mapping[str, Sequence[str]],
    prompt_template: str,
    model: str,
    output_path: Path,
    api_interface: Any,
    delay: float,
    debug: bool,
    resume: bool,
    api_base_url: str,
) -> dict[str, list[dict[str, Any]]]:
    """Generate a plan with atomic progress checkpoints and final publication."""
    partial_path = partial_path_for(output_path)
    fingerprint = build_run_fingerprint(
        model=model,
        policy_options=policy_options,
        prompt_template=prompt_template,
        plan=plan,
        api_base_url=api_base_url,
    )

    if resume:
        partial = validate_partial(
            load_json_file(partial_path),
            fingerprint=fingerprint,
            model=model,
            plan=plan,
        )
    else:
        partial = _new_partial(fingerprint, model)
        atomic_write_json(partial, partial_path)

    start_index = len(partial["completed"])
    print(f"Processing {len(plan)} policy proposals with model {model!r}.")
    if start_index:
        print(f"Resuming after {start_index} validated proposals.")

    for index in range(start_index, len(plan)):
        category, proposal = plan[index]
        print(f"  [{index + 1}/{len(plan)}] {category}: {proposal[:70]}")
        result = convert_proposal_to_action(
            policy_proposal=proposal,
            prompt_template=prompt_template,
            model=model,
            api_interface=api_interface,
            debug=debug,
        )
        partial["completed"].append({"category": category, "result": result})
        atomic_write_json(partial, partial_path)
        if delay and index + 1 < len(plan):
            time.sleep(delay)

    final_results = build_final_results(partial["completed"], plan)
    atomic_write_json(final_results, output_path)
    partial_path.unlink(missing_ok=True)
    _fsync_directory(partial_path.parent)
    return final_results


def run_from_args(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Validate configuration, execute generation, and always close the API."""
    requested_output_path = Path(args.output).expanduser()
    output_path = _resolved(requested_output_path)
    policy_options_path = _resolved(args.policy_options)
    prompt_template_path = _resolved(args.prompt_template)
    partial_path = partial_path_for(output_path)

    ensure_output_safety(
        output_path=requested_output_path,
        partial_path=partial_path,
        policy_options_path=policy_options_path,
        prompt_template_path=prompt_template_path,
        debug_limit=args.debug,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    policy_options = validate_policy_options(load_json_file(policy_options_path))
    prompt_template = load_prompt_template(prompt_template_path)
    plan = build_plan(policy_options, args.debug)
    if not plan:
        raise ProposalActionError("The selected generation plan is empty.")

    api_base_url = resolve_openrouter_base_url()

    # Validate resume identity before constructing a potentially costly API client.
    validated_partial: dict[str, Any] | None = None
    if args.resume:
        fingerprint = build_run_fingerprint(
            model=args.model,
            policy_options=policy_options,
            prompt_template=prompt_template,
            plan=plan,
            api_base_url=api_base_url,
        )
        validated_partial = validate_partial(
            load_json_file(partial_path),
            fingerprint=fingerprint,
            model=args.model,
            plan=plan,
        )

    # A crash may occur after the last partial checkpoint is committed but
    # before final publication. Finishing that transaction is entirely offline;
    # do not require credentials or initialize a paid API client.
    if validated_partial is not None and len(validated_partial["completed"]) == len(
        plan
    ):
        final_results = build_final_results(validated_partial["completed"], plan)
        atomic_write_json(final_results, output_path)
        partial_path.unlink(missing_ok=True)
        _fsync_directory(partial_path.parent)
        return final_results

    api_interface = create_api_interface(args.model, base_url=api_base_url)
    try:
        return process_plan(
            plan=plan,
            policy_options=policy_options,
            prompt_template=prompt_template,
            model=args.model,
            output_path=output_path,
            api_interface=api_interface,
            delay=args.delay,
            debug=args.debug is not None,
            resume=args.resume,
            api_base_url=api_base_url,
        )
    finally:
        close_api_interface(api_interface)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Return zero on success and non-zero on any failure."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = run_from_args(args)
    except (ProposalActionError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # API/provider errors must also produce a non-zero exit.
        print(f"Unexpected error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    total = sum(len(entries) for entries in results.values())
    print(f"Completed {total} proposals. Results saved to: {_resolved(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
