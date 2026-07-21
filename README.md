# LLM Political Belief Update

This repository studies how persona-conditioned LLM responses change after the model is shown hypothetical social-opinion cues. It measures policy-benefit judgments, predictions about public opinion, and support for concrete political actions through two pipelines:

- `verbalize`: strict JSON answers, using either OpenRouter-compatible HTTP or local vLLM;
- `logprob`: local vLLM only, using a deliberately bounded single-token Yes/No estimator.

The experiment measures prompt-conditioned response shifts. It does **not** demonstrate a persistent change to an LLM's internal beliefs or state.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── data/
│   ├── entities.json
│   ├── policy_options.json
│   ├── proposal_actions.json
│   ├── proposal2action.py
│   └── proposal2action.txt
├── src/
│   ├── data/data_loader.py
│   ├── experiment/
│   │   ├── base_runner.py
│   │   ├── checkpoints.py
│   │   ├── compiler.py
│   │   ├── core.py
│   │   ├── planning.py
│   │   ├── logprob_experiment_runner.py
│   │   └── verbalize_experiment_runner.py
│   ├── models/
│   │   ├── binary_logprob.py
│   │   ├── unified_llm_interface.py
│   │   └── vllm_interface.py
│   └── prompts/
│       ├── logprob/
│       └── verbalize/
├── tests/
├── .env.example
├── .gitignore
├── CHECKPOINT_RESUME_GUIDE.md
├── environment.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── requirements-vllm.txt
```

## Experimental design

The five logical stages are:

| Stage | Unit | Measurement |
| --- | --- | --- |
| Step 1 | `(persona, proposal, replicate)` | Baseline Yes/No judgment: is the policy beneficial to the United States? |
| Step 2 | `(persona, proposal, replicate)` | Predicted percentage of the U.S. population that would answer Yes to the Step 1 question. |
| Step 3 | `(persona, proposal, action, replicate)` | Baseline Yes/No support for a concrete action. |
| Step 4a | `(persona, proposal, treatment, replicate)` | Policy-benefit judgment after a treatment. This is proposal-level and is not redundantly inferred once per action. |
| Step 4b | `(persona, proposal, action, treatment, replicate)` | Action support after the same treatment design. |

The default Step 4 treatment set contains eight distinct conditions:

1. five explicitly hypothetical survey values: `10`, `30`, `50`, `70`, and `90` percent;
2. one `simulated_persona_consensus` value derived from other persona conditions that have at least one valid Step 1 replicate from the same model;
3. one no-information retest that reuses the baseline prompt;
4. one neutral-text placebo.

The simulated consensus is leave-one-persona-out: the target persona is excluded, valid replicates are averaged within each remaining persona, and remaining personas receive equal weight. Its metadata fields `consensus_persona_n` and compatibility alias `consensus_n` both count contributing personas, not individual response records. It is a summary of simulated model outputs, not a population sample or real poll. A fixed and simulated condition remain distinct even if their percentages happen to be equal.

Each assignment records a stable `sample_id`, `replicate_id`, generation seed, and treatment `order_index`. Treatment order is reproducibly shuffled for each unit and replicate. The Step 2 prediction is used to record:

```text
survey_surprise = treatment_percentage - step2_predicted_percentage
```

For retest and placebo conditions, which have no percentage, `survey_surprise` is `null`. If no valid other-persona Step 1 observations exist, the simulated-consensus cell is recorded as `INVALID` without making a model call.

The special persona label `none` adds no persona instruction. It is a genuine no-persona-prompt control, not an “objective analyst” persona.

## Measurement pipelines

### Verbalize

Prompts request a standalone JSON object. The parser accepts either that object or exactly one JSON code fence containing it, but never surrounding prose. A Yes/No `answer` is trimmed and case-normalized to the two-value enum; Step 2 accepts a finite number or complete percentage string in `[0, 100]`. The parser does not clamp out-of-range values or extract unrelated numbers.

Verbalize can use:

- OpenRouter-compatible HTTP with `--use-api`; or
- local vLLM when `--use-api` is omitted.

### Logprob and the deliberate truncation

The logprob pipeline uses local vLLM only. Every binary stage has two generated sequences:

1. generate visible analysis text without a final answer;
2. continue the assistant message for exactly one token at scoring temperature `0.0`.

Before scoring, the tokenizer is used to discover a finite set of supported single-token spellings of Yes and No, including selected whitespace, case, and punctuation variants. Multi-token variants are excluded, and at most 128 token IDs are requested through vLLM's `logprob_token_ids` interface. The runner never requests the full vocabulary.

The reported `probabilities.Yes` and `probabilities.No` are therefore conditional on this finite candidate set. This truncation is intentional: it reduces computation and memory at the cost of incomplete answer-form coverage. Every valid score also reports:

- `candidate_mass`: full next-token probability mass captured by the requested candidate IDs;
- `residual_mass`: `1 - candidate_mass`;
- the candidate token IDs and log probabilities;
- `sampled_choice` and `format_valid`.

A missing candidate score, incompatible tokenizer/API, malformed mass, or sampled token outside the Yes/No candidate set is `INVALID`; the code never fills in `0.5/0.5`. Phase-1 `analysis_text` is visible model-generated text, not hidden chain-of-thought.

## Data

The tracked data currently contain:

- 32 politician labels and 7 platform labels;
- 18 policy categories and 136 proposals;
- 408 actions: three per proposal (`Personal Commitment`, `Public Advocacy`, and `Strategic Support`).

`entities.json` and `policy_options.json` were adapted from [Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs](https://arxiv.org/abs/2502.08640). The action text was generated by an LLM and has not received the human validation needed to establish cost, feasibility, ideological neutrality, or cross-persona comparability.

## Installation

Project metadata supports Python 3.11, 3.12, and 3.13. Local CPU tests were run with Python 3.11 and 3.12, and CI runs both versions. Accelerator-specific vLLM compatibility must be checked separately for the target operating system, driver, CUDA/ROCm stack, and GPU.

### Runtime only

```bash
python -m pip install -r requirements.txt
```

This installs the CPU/API dependencies. It does not install vLLM.

The repository can also be installed as a wheel or editable project with `python -m pip install .`; the wheel includes the tracked experiment data and prompt resources. Running directly from a clone remains the primary development workflow.

### Development and offline tests

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

### Conda CPU/API environment

```bash
conda env create -f environment.yml
conda activate llm-belief-cpu
pytest
```

`environment.yml` is intended for CPU/API development and testing. It does not install vLLM.

### Local vLLM backend

Install vLLM only in a compatible accelerator environment, preferably separate from the CPU/API environment:

```bash
python -m pip install -r requirements-vllm.txt
```

The repository pins `vllm==0.24.0`. Follow the [official vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/) for the correct platform-specific stack. Neither the logprob pipeline nor local verbalize inference is expected to work on an ordinary CPU-only environment.

## OpenRouter configuration and endpoint safety

Copy the placeholder file and add your own key locally:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Security behavior is deliberate:

- `OPENROUTER_API_KEY` is required for OpenRouter API calls;
- `OPENAI_API_KEY` is never consulted or forwarded;
- the default endpoint is `https://openrouter.ai/api/v1`;
- public endpoints must use HTTPS; plain HTTP is accepted only for loopback hosts;
- credentials, query parameters, and fragments are rejected in endpoint URLs;
- an OpenRouter key is never forwarded to a non-OpenRouter host;
- custom non-OpenRouter hosts require an explicitly paired key in application code. The experiment CLI does not infer such a key from the environment;
- keys are not written into manifests or result files.

Do not commit `.env` or print real credentials. `--trust-remote-code` is disabled by default for local models. Enabling it executes code from the model repository and is rejected unless `--code-revision` or `--model-revision` pins that code.

## Plan before spending money or GPU time

Both runners support `--dry-run`. It validates data and prompt contracts and prints deterministic planned counts without loading a model or creating checkpoints:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model openai/gpt-4.1-mini \
  --use-api \
  --max-base-units 12 \
  --replicates 2 \
  --dry-run

python -m src.experiment.logprob_experiment_runner \
  --model Qwen/Qwen3-0.6B \
  --max-base-units 12 \
  --dry-run
```

`--max-base-units` is a positive action-level budget applied before any Step 1 or Step 2 call. The selected subset is deterministic for the master `--seed`. Logical sample counts are exact for that plan. Backend sequence counts are conservative upper bounds: simulated-consensus cells or a logprob continuation can be skipped when prerequisite outputs are invalid. Use dry-run output, not intuition, to estimate maximum work.

## Run the verbalize pipeline

OpenRouter example:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model openai/gpt-4.1-mini \
  --use-api \
  --max-base-units 12 \
  --results-dir results
```

Local vLLM example:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model Qwen/Qwen3-0.6B \
  --max-base-units 12 \
  --results-dir results
```

Important options include:

- `--persona/--personas ...` and `--category/--categories ...` for filtering;
- `--max-base-units N` (`--max-experiments` is a compatibility alias in this runner);
- `--replicates N`, `--seed`, `--temperature`, and `--max-tokens`;
- `--fixed-percentages ...`;
- `--no-simulated-consensus`, `--no-retest`, and `--no-placebo` to disable default conditions;
- `--chunk-size N` for checkpoint shard size;
- `--model-revision`, `--tokenizer-revision`, and `--code-revision` for reproducible local artifacts;
- API retry/concurrency controls such as `--api-max-workers` and `--api-retry-total-timeout`;
- local vLLM controls such as `--tensor-parallel-size`, `--gpu-memory-utilization`, and the explicit `--trust-remote-code` opt-in.

These revision flags apply only to local vLLM and are rejected with `--use-api`; hosted versions must be selected through the provider's model identifier. For local models, when the same immutable revision applies to all three artifacts, `--model-revision` is sufficient: tokenizer and remote-code revisions default to it. Pass the two more specific flags when they differ. The logprob runner exposes the same revision controls.

Run `python -m src.experiment.verbalize_experiment_runner --help` for the authoritative CLI.

## Run the logprob pipeline

```bash
python -m src.experiment.logprob_experiment_runner \
  --model Qwen/Qwen3-0.6B \
  --model-revision REVISION_OR_COMMIT \
  --max-base-units 12 \
  --results-dir results
```

This command requires the pinned vLLM dependency and compatible GPU resources. It has the same selection, treatment, replicate, chunk, dry-run, and resume concepts as the verbalize runner. It uses `--no-simulated-consensus`, `--no-retest`, and `--no-placebo` to disable conditions. API mode is intentionally unavailable.

Run `python -m src.experiment.logprob_experiment_runner --help` for the authoritative CLI.

## Checkpoints and resume

Each real run creates:

```text
results/
├── checkpoints/<run_id>/
│   ├── manifest.json
│   ├── step1/
│   │   ├── .sample-index.sqlite3
│   │   └── chunk_00000000.json
│   ├── step2/
│   │   ├── .sample-index.sqlite3
│   │   └── chunk_00000000.json
│   └── ...
└── results_<run_id>.json
```

Each stage's `.sample-index.sqlite3` is a disposable performance index used to detect duplicate sample IDs without rereading every historical shard before each new chunk. The immutable JSON chunks remain the sole source of sample records: a missing, stale, or damaged index is rebuilt from them and may be deleted safely when no writer is active.

List compatible runs without loading a model:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --results-dir results \
  --list-checkpoints

python -m src.experiment.logprob_experiment_runner \
  --results-dir results \
  --list-checkpoints
```

Resume from the first logical stage that should be checked/executed:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --results-dir results \
  --resume-from RUN_ID \
  --resume-step step4a
```

Both runners allow `--model` to be omitted with `--resume-from`; it remains required for a new run or dry-run. The stored manifest restores the model, backend, revisions, and experimental configuration, overriding newly supplied experimental flags. Restore validates the pipeline, Git/source version, prompt and data hashes, configuration, exact sampling plan, expected sample IDs, reconstructed assignment metadata, and dependency DAG. Existing stage chunks are immutable and completed IDs are skipped.

An `ERROR` record is itself a completed immutable observation, so resume will not replace it. Start a new run to retry backend errors. See [CHECKPOINT_RESUME_GUIDE.md](CHECKPOINT_RESUME_GUIDE.md) for the full format and recovery rules.

## Output and status semantics

The final file is `results/results_<run_id>.json`. It contains the full manifest, a per-stage `status_summary`, and action-level compiled rows. Every stage observation is a self-contained record with:

```text
sample_id, stage, metadata, status, value,
error_code, error_message, raw_response
```

Statuses mean:

- `VALID`: passed the stage-specific schema and estimator checks;
- `INVALID`: a response was obtained or a treatment cell was planned, but it did not satisfy the measurement contract. It is never imputed as Yes, No, or 50%;
- `ERROR`: execution failed, for example because of a backend or response-container exception.

Invalid and error observations do not enter the denominator when simulated binary consensus is derived. A run with only `INVALID` observations can finish with a warning; a compiled run containing any `ERROR` observation returns a non-zero process status.

Compilation joins by stable sample ID and validates duplicate, unknown, missing, and wrong-stage records. Step 4a is inferred once per proposal unit and then referenced in the denormalized action-level output; it is not requested three times.

## Workload and resource warning

With all 40 persona conditions, 136 proposals, 408 actions, one replicate, and all eight default Step 4 conditions, dry-run reports the following planned logical samples and backend-sequence ceilings:

| Quantity | Count |
| --- | ---: |
| proposal-level units | 5,440 |
| action-level base units | 16,320 |
| Step 4a logical samples | 43,520 |
| Step 4b logical samples | 130,560 |
| total verbalize backend sequences | 201,280 |
| total logprob backend sequences | 397,120 |

Logprob counts each two-phase binary measurement as up to two generated sequences. Invalid prerequisite output can reduce actual calls below these ceilings; replicates multiply the plan. Full runs can incur substantial API charges, GPU time, wall time, memory use, and checkpoint storage. Start with `--dry-run` and a very small `--max-base-units`; do not launch a full run merely to test installation.

## Generate proposal-to-action data

The repository already tracks `data/proposal_actions.json`. Regeneration is optional, makes paid API calls, and requires an explicit output path:

```bash
python -m data.proposal2action \
  --model google/gemini-3-pro-preview \
  --output results/proposal_actions.generated.json \
  --debug 3
```

Behavior and safety rules:

- `--output` is required;
- `--model` is honored;
- an existing final output requires `--overwrite`;
- progress is atomically saved next to the target as `NAME.partial.json`;
- `--resume` requires that partial file and strictly matches the model, input data, prompt, selected plan, fixed generation settings, normalized OpenRouter endpoint, generator source, and unified model-interface source fingerprint;
- responses must be standalone JSON with exactly the three required action types;
- the final file is atomically published only after every selected proposal validates;
- failures preserve valid partial progress and do not publish an incomplete final file;
- debug mode is forbidden when the output is the tracked `data/proposal_actions.json`.

Generator partial checkpoints currently use schema version 2. Older partial schemas, or partials created under a different endpoint, generation regime, or generator/interface source, are intentionally not resumable. Preserve an old partial for audit, then start with a new output path or explicitly use `--overwrite` to create a fresh partial.

To deliberately regenerate the tracked canonical file, run the complete plan without `--debug`, use the canonical input and prompt, review the generated data, and explicitly pass:

```bash
python -m data.proposal2action \
  --model MODEL_ID \
  --output data/proposal_actions.json \
  --overwrite
```

## Verification

The CPU-only suite covers strict parsing, data and prompt schemas, planning, leave-one-out consensus, treatment randomization, checkpoints, linear compilation, API safety and retries, model cleanup, generator safety, both CLIs, and fake-backend end-to-end runs:

```bash
pytest
```

At the time of this revision, all 165 offline tests pass under both Python 3.11 and 3.12, and `pip check` reports no broken requirements in a fresh CPU environment. A wheel built from the source tree also installs in an isolated Python 3.11 environment, exposes all tracked prompts/data, and completes a dry-run. No paid API request, real vLLM model load, CUDA/GPU run, or full experiment is part of that verification.

## Interpretation limits

Even with the engineering safeguards, important research limitations remain:

- independent prompts measure contextual response changes, not persistent belief updates;
- simulated consensus is endogenous to the same model and is not population-representative;
- API providers and hosted models may drift during long runs;
- comparing verbalize/API with logprob/vLLM can confound extraction method and backend;
- binary Yes/No outcomes compress uncertainty, neutrality, and refusal;
- replicates are repeated model calls rather than independent respondents, and at deterministic temperature they may be identical;
- repeated personas are not independent human respondents;
- valid inference requires a pre-specified multilevel/clustered analysis and multiple-comparison plan;
- action text still needs human review and source-credibility conditions remain limited.

Treat the framework as an exploratory instrument for persona-conditioned social-information sensitivity, not as evidence of real public opinion or durable internal belief change.
