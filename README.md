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
│   ├── __init__.py
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

The simulated consensus is leave-one-persona-out: the target persona is excluded, valid replicates are averaged within each remaining persona, and remaining personas receive equal weight. For logprob records, a replicate contributes only when `format_valid` is true and `sampled_choice` is exactly `Yes` or `No`; consensus follows that observed sampled choice, including when the conditional Yes/No probabilities are tied, and never substitutes an argmax. Its metadata fields `consensus_persona_n` and compatibility alias `consensus_n` both count contributing personas, not individual response records. It is a summary of simulated model outputs, not a population sample or real poll. A fixed and simulated condition remain distinct even if their percentages happen to be equal.

Each assignment records a stable, SHA-256-derived `sample_id`, `replicate_id`, generation seed, and treatment `order_index`. These IDs are collision-resistant engineering identifiers, not a mathematical guarantee that collisions are impossible. Treatment order is reproducibly shuffled for each unit and replicate. The Step 2 prediction is used to record:

```text
survey_surprise = treatment_percentage - step2_predicted_percentage
```

Treatment-planning percentages use one six-decimal-place contract: fixed values are quantized before uniqueness checks, treatment conditions store the quantized value, simulated consensus and `survey_surprise` are quantized to the same precision, and prompt rendering uses fixed-point text with trailing zeros removed rather than scientific notation. For retest and placebo conditions, which have no percentage, `survey_surprise` is `null`. If no valid other-persona Step 1 observations exist, the simulated-consensus cell is recorded as `INVALID` without making a model call.

The special persona label `none` adds no persona instruction. It is a genuine no-persona-prompt control, not an “objective analyst” persona.

## Measurement pipelines

### Verbalize

Prompts request a standalone JSON object. The parser accepts either that object or exactly one JSON code fence containing it, but never surrounding prose. A Yes/No `answer` is trimmed and case-normalized to the two-value enum; Step 2 accepts a finite number or complete percentage string in `[0, 100]`. The parser does not clamp out-of-range values or extract unrelated numbers.

Verbalize can use:

- OpenRouter-compatible HTTP with `--use-api`; or
- local vLLM when `--use-api` is omitted.

Local vLLM verbalize runs disable chat-template thinking so the model cannot
wrap the required standalone JSON object in a reasoning preamble. Hosted API
reasoning behavior remains provider-managed.

### Logprob and the deliberate truncation

The logprob pipeline uses local vLLM only. It explicitly disables model chat-template thinking (`enable_thinking=False`) so Phase 1 contains only the requested visible analysis and Step 2 can satisfy its strict JSON contract. Phase 1 and Step 2 prompts bound the requested analysis to 100 words; `--max-tokens` remains the hard generation ceiling. Each executable binary observation follows a two-phase protocol and generates at most two sequences:

1. generate visible analysis text without a final answer;
2. close that assistant analysis turn, add a new user request for exactly `Yes` or `No`, and score the first token of the fresh assistant response at temperature `0.0`.

The Phase-2 dialogue is therefore `user analysis request → assistant visible analysis → user binary-answer request`; it does not reclassify visible analysis as hidden `reasoning_content` and does not continue an open assistant message. The manifest pins this versioned contract as `bounded_scoring_protocol=completed_analysis_new_user_fresh_assistant_v1`, so checkpoints created under older scoring dialogue layouts cannot be mixed with or resumed under this protocol.

Before the first executable chat, including a Step 2 chat, the runner initializes the backend and performs `preflight_bounded_scoring()` exactly once. The preflight checks the vLLM API, tokenizer-specific Yes/No candidates, and chat-template boundaries for the completed analysis, new user request, fresh assistant generation prompt, and visible answer token. An incompatible API, tokenizer, model, or disabled-thinking contract is a run-level fatal error. A whole-batch local vLLM failure, including CUDA OOM, is also fatal: the current chunk is not published and its IDs remain missing. The run can be safely resumed after correcting the cause; a smaller operational `--chunk-size` may resolve batch-size or OOM failures, while a change to manifest-bound configuration requires a new run. Planning-only operations and chunks with no executable assignments do not initialize the backend. Once a backend returns an aligned batch, malformed individual response containers or score payloads remain auditable per-sample `ERROR` records.

Before scoring, the tokenizer is used to discover a finite set of supported single-token spellings of Yes and No, including selected whitespace, case, and punctuation variants. Multi-token variants are excluded, and at most 128 token IDs are requested through vLLM's `logprob_token_ids` interface. The runner never requests the full vocabulary.

The reported `probabilities.Yes` and `probabilities.No` are therefore conditional on this finite candidate set. This truncation is intentional: it reduces computation and memory at the cost of incomplete answer-form coverage. Every valid score also reports:

- `candidate_mass`: full next-token probability mass captured by the requested candidate IDs;
- `residual_mass`: `1 - candidate_mass`;
- the candidate token IDs and log probabilities;
- `sampled_choice` and `format_valid`.

After preflight, a missing candidate score or sampled token outside the Yes/No candidate set is `INVALID`; malformed response containers or inconsistent score fields are explicit `ERROR` records. The code never fills in `0.5/0.5`. Phase-1 `analysis_text` is visible model-generated text, not hidden chain-of-thought. Its `finish_reason` must be a string or null; `length` becomes `INVALID/phase1_truncated`, while a conservatively recognized explicit final Yes/No answer becomes `INVALID/phase1_contains_final_answer`. Neither case requests Phase-2 bounded scoring.

## Data

The tracked data currently contain:

- 32 politician labels and 7 platform labels;
- 18 policy categories and 136 proposals;
- 408 actions: three per proposal (`Personal Commitment`, `Public Advocacy`, and `Strategic Support`).

`entities.json` and `policy_options.json` were adapted from [Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs](https://arxiv.org/abs/2502.08640). The action text was generated by an LLM and has not received the human validation needed to establish cost, feasibility, ideological neutrality, or cross-persona comparability.

The loader reads, validates, and hashes each JSON file from the same exact byte snapshot, then serves defensive copies from its validated cache. Manifest creation verifies that the files have not drifted since that snapshot was loaded; if they changed, the run fails and must be started or resumed with a fresh runner rather than binding cached planning data to newer on-disk hashes.

## Installation

Project metadata supports Python 3.11, 3.12, and 3.13, while CI targets Python 3.11 and 3.12. Accelerator-specific vLLM compatibility must be checked separately for the target operating system, driver, CUDA/ROCm stack, and GPU.

### Runtime only

```bash
python -m pip install -r requirements.txt
```

This installs the CPU/API dependencies. It does not install vLLM.

The repository can also be installed as a wheel or editable project with `python -m pip install .`; the wheel includes the tracked experiment data, prompt resources, and the `data.proposal2action` generator module. Running directly from a clone remains the primary development workflow.

By default, both runners write to `results/` under the current working directory, independent of whether they run from a clone or an installed wheel. The real-run examples below still pass an explicit `--results-dir` so the output location is deliberate.

The exact versions in `pyproject.toml` and `requirements*.txt` pin the project's direct, top-level dependencies. They are not a complete transitive lock file; the installer still resolves indirect dependencies for the target platform.

### Development and offline tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Pytest also has `pythonpath = ["."]` in `pyproject.toml`, so bare `pytest` from the repository root resolves the same package imports. CI deliberately keeps `python -m pytest` as the canonical invocation.

### Conda CPU/API environment

```bash
conda env create -f environment.yml
conda activate llm-belief-cpu
python -m pytest
```

`environment.yml` is intended for CPU/API development and testing. It does not install vLLM.

### Local vLLM backend

Install the runtime and vLLM accelerator supplement in a compatible accelerator environment, preferably separate from the CPU/API environment:

```bash
python -m pip install -r requirements.txt -r requirements-vllm.txt
```

For GPU-backed repository development and offline tests, install both dependency layers in the same environment; `requirements-dev.txt` already includes `requirements.txt`:

```bash
python -m pip install -r requirements-vllm.txt -r requirements-dev.txt
```

The repository pins `vllm==0.24.0`. Follow the [version-matched vLLM 0.24 GPU installation guide](https://docs.vllm.ai/en/v0.24.0/getting_started/installation/gpu/) for the correct platform-specific stack. The local paths documented here target a compatible accelerator environment; a vLLM CPU build has not been validated for this repository.

A Linux x86-64 compatibility check has passed with Python 3.11.15, `vllm==0.24.0`, a PyTorch 2.11.0 CUDA 13.0 build (`torch==2.11.0+cu130`), and an NVIDIA RTX PRO 6000 Blackwell GPU (compute capability 12.0, driver 580.82.09). In addition to dependency, import, CUDA, offline-test, and Ruff checks, original-BF16 Qwen3 and Qwen3.5 checkpoints completed real single-GPU engine initialization and backend-compatibility smoke checks. Current-protocol pipeline validation is specific to Qwen3.5: an original-BF16 Qwen3.5-9B checkpoint completed both a pilot and the default full plan under `completed_analysis_new_user_fresh_assistant_v1`. All 201,280 full-run records were `VALID`, with no missing, `INVALID`, or `ERROR` records, and an independent audit reproduced the manifest, assignments, checkpoint digests, final compilation, and derived indexes. The generated multi-gigabyte result and checkpoints remain outside the repository. This is validation on one host and one full-run model, not a multi-GPU or general performance guarantee.

On that host, `flashinfer==0.6.12` misidentified Blackwell `sm_120` during sampler JIT warm-up. Setting `VLLM_USE_FLASHINFER_SAMPLER=0` selected vLLM's native sampler and allowed both model families to run; use this fallback only when the FlashInfer sampler fails on the installed CUDA/GPU combination.

Model identifiers in the examples are illustrative. Select a local checkpoint from the required model family only after budgeting model weights (roughly two bytes per parameter for BF16), KV cache, CUDA graphs, scheduler concurrency, runtime overhead, download storage, and result storage against the actual host. A smaller official checkpoint is preferable when the larger one would leave inadequate safety margin; use a dry-run and a small `--max-base-units` pilot before any full plan.

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

`--max-base-units` is a positive action-level budget applied before any Step 1 or Step 2 call. The selected subset is deterministic for the master `--seed`. Logical sample counts are exact for that plan. Backend sequence counts are conservative upper bounds: simulated-consensus cells or a Phase-2 bounded-score request can be skipped when prerequisite outputs are invalid. Use dry-run output, not intuition, to estimate maximum work.

`--dry-run` and `--resume-from` are mutually exclusive in both CLIs. A dry-run always plans a new configuration and never silently ignores a resume request.

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
  --max-tokens 256 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
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
- local vLLM controls such as `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--language-model-only`, and the explicit `--trust-remote-code` opt-in.

These revision and resource flags apply only to local vLLM and non-default local resource values are rejected with `--use-api`; hosted versions must be selected through the provider's model identifier. For local models, when the same immutable revision applies to all three artifacts, `--model-revision` is sufficient: tokenizer and remote-code revisions default to it. Pass the two more specific flags when they differ. The logprob runner exposes the same revision and resource controls. Add `--language-model-only` for text-only use of a multimodal Qwen3.5 checkpoint.

Run `python -m src.experiment.verbalize_experiment_runner --help` for the authoritative CLI.

## Run the logprob pipeline

```bash
python -m src.experiment.logprob_experiment_runner \
  --model Qwen/Qwen3-0.6B \
  --model-revision REVISION_OR_COMMIT \
  --max-tokens 256 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-base-units 12 \
  --results-dir results
```

This command requires the pinned vLLM dependency and compatible GPU resources. It has the same selection, treatment, replicate, chunk, dry-run, and resume concepts as the verbalize runner. It uses `--no-simulated-consensus`, `--no-retest`, and `--no-placebo` to disable conditions. API mode is intentionally unavailable. `--max-model-len` must exceed `--max-tokens`; use it to avoid reserving a model's full long-context capacity when the experiment needs much less. `--max-num-seqs` bounds scheduler concurrency. For a multimodal Qwen3.5 checkpoint used only with these text prompts, add `--language-model-only` to skip its vision tower.

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

`run_id` combines the pipeline name, a UTC microsecond timestamp, and a random UUID fragment. This is designed to make accidental collisions extremely unlikely, not to prove absolute uniqueness; existing-run checks remain authoritative.

Experiment manifests and stage chunks currently use checkpoint schema version 2. Version 1 manifests/chunks are explicitly incompatible and must be preserved for audit rather than resumed as version 2. Every v2 shard stores `records_sha256`; loading and writing also perform stage- and pipeline-specific semantic validation of record metadata, statuses, and values rather than relying on a digest alone.

Storage schema and measurement protocol are separate compatibility layers. Two runs can both use checkpoint schema v2 while remaining scientifically and operationally incompatible. In particular, a logprob run must contain the exact current `bounded_scoring_protocol`; a run created with an older open-assistant or `reasoning_content` layout must be preserved for audit and restarted under a new run ID, never upgraded by editing its manifest.

Each stage's `.sample-index.sqlite3` is a disposable duplicate-ID index. The immutable JSON chunks remain the sole authority for records. On a cold open, or after any shard/index signature changes outside the current `CheckpointStore`, all existing shards are reread and validated, compared with SQLite, and used to rebuild a missing, stale, inconsistent, or damaged index. Within the same `CheckpointStore` instance, an unchanged warm authority cache uses file signatures plus compact index metadata to avoid rereading historical shard contents on each normal append; any mismatch falls back to the full JSON-authority scan. The index may be deleted safely when no writer is active.

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

Both runners allow `--model` to be omitted with `--resume-from`; it remains required for a new run or dry-run. The stored manifest restores the model, backend, revisions, and experimental configuration, overriding newly supplied experimental flags. Restore must use a fresh runner whose backend has not already been initialized or injected; tests and embedding applications that need a custom backend must provide it through a lazy factory. Restore validates the pipeline, Git/source version, exact validated data snapshots and hashes, prompt hashes, configuration, exact scoring protocol, sampling plan, expected sample IDs, reconstructed assignment metadata, record digests/semantics, and dependency DAG. Existing stage chunks are immutable and completed IDs are skipped. Changing the scoring dialogue or protocol requires a fresh run; manually adding a new protocol field to an old manifest would both break its fingerprint and misdescribe how its logits were conditioned.

An already-published `ERROR` record is a completed immutable observation, so resume will not replace it; start a new run to retry that observation. In contrast, a logprob local-vLLM batch failure publishes no chunk and is explicitly resumable because those IDs remain missing. The fatal error message includes the checkpoint run ID. See [CHECKPOINT_RESUME_GUIDE.md](CHECKPOINT_RESUME_GUIDE.md) for the full format and recovery rules.

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

For a binary record, `value.finish_reason` belongs to the one-token fresh-assistant scoring request. Because that request fixes `max_tokens=1`, `finish_reason=length` is normally expected there and is not evidence that Phase 1 was truncated; actual analysis truncation is recorded separately as `INVALID/phase1_truncated`.

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

Logprob counts each two-phase binary measurement as up to two generated sequences. Invalid prerequisite output can reduce actual calls below these ceilings; replicates multiply the plan. Full runs can incur substantial API charges, GPU time, wall time, memory use, and checkpoint storage. A full fake-backend logprob run with only 16-character analyses already produced about 502 MB of checkpoint JSON and a 758 MB compiled JSON; real model text can require several GiB. The compiled action rows intentionally repeat proposal-level Step 1, Step 2, and Step 4a records, so downstream analysis must deduplicate those observations by `sample_id`. Start with `--dry-run` and a very small `--max-base-units`; do not launch a full run merely to test installation.

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
python -m pytest -q
```

The authoritative test count is the output of the current command and is intentionally not hard-coded here as the suite evolves. Bare `pytest -q` is also configured from the repository root, while CI uses `python -m pytest`. A wheel built from the source tree has been checked in an isolated Python 3.11 environment: it exposes the tracked prompts/data and generator module and completes installed-package dry-runs. The real single-GPU coverage described above includes one full Qwen3.5-9B logprob validation, but its bulk artifacts are not distributed with the repository. No paid API request or multi-GPU execution is part of the recorded compatibility check.

## Interpretation limits

Even with the engineering safeguards, important research limitations remain:

- independent prompts measure contextual response changes, not persistent belief updates;
- simulated consensus is endogenous to the same model and is not population-representative;
- API providers and hosted models may drift during long runs;
- comparing verbalize/API with logprob/vLLM can confound extraction method and backend;
- cross-run comparisons must verify the same `bounded_scoring_protocol`, prompt/data hashes, and model/tokenizer revisions; matching `sample_id` values do not make observations from different scoring protocols interchangeable;
- binary Yes/No outcomes compress uncertainty, neutrality, and refusal;
- replicates are repeated model calls rather than independent respondents, and at deterministic temperature they may be identical;
- repeated personas are not independent human respondents;
- valid inference requires a pre-specified multilevel/clustered analysis and multiple-comparison plan;
- action text still needs human review and source-credibility conditions remain limited.

Treat the framework as an exploratory instrument for persona-conditioned social-information sensitivity, not as evidence of real public opinion or durable internal belief change.
