# LLM Political Belief Update

This repository studies how persona-conditioned LLM responses change after the model is shown hypothetical social-opinion cues. It measures policy-benefit judgments, predictions about public opinion, and support for concrete political actions.

Two experiment pipelines implement the same design:

- `verbalize` requests strict JSON answers from either local vLLM or an OpenRouter-compatible API;
- `logprob` uses local vLLM to estimate Yes/No probabilities from a bounded set of tokenizer-specific first-token candidates.

## Repository layout

```text
.
├── data/                       # Tracked entities, proposals, and actions
├── src/
│   ├── data/                  # Data loading and validation
│   ├── experiment/            # Planning, execution, checkpoints, and compilation
│   ├── models/                # API and local vLLM interfaces
│   └── prompts/               # Verbalize and logprob prompt templates
├── tests/                     # Offline tests with fake backends
├── CHECKPOINT_RESUME_GUIDE.md
└── requirements.txt
```

## Experimental design

The experiment has five logical stages:

| Stage | Unit | Measurement |
| --- | --- | --- |
| Step 1 | `(persona, proposal, replicate)` | Baseline Yes/No judgment of whether a policy is beneficial to the United States. |
| Step 2 | `(persona, proposal, replicate)` | Predicted percentage of the U.S. population that would answer Yes in Step 1. |
| Step 3 | `(persona, proposal, action, replicate)` | Baseline Yes/No support for a concrete action. |
| Step 4a | `(persona, proposal, treatment, replicate)` | Policy-benefit judgment after a treatment. |
| Step 4b | `(persona, proposal, action, treatment, replicate)` | Action support after a treatment. |

Step 4a is proposal-level. It is measured once for each persona, proposal, treatment, and replicate rather than once per action.

### Treatments

The default Step 4 design has eight conditions:

1. five hypothetical survey values: `10`, `30`, `50`, `70`, and `90` percent;
2. one leave-one-persona-out `simulated_persona_consensus` value;
3. one no-information retest using the baseline prompt;
4. one neutral-text placebo.

For simulated consensus, the target persona is excluded. Valid Step 1 replicates are averaged within each remaining persona, and the resulting persona means receive equal weight.

For logprob records, consensus uses only a format-valid observed `sampled_choice` of `Yes` or `No`, not the conditional Yes probability or an argmax.

If no other persona has a valid Step 1 observation, the simulated-consensus cell is recorded as `INVALID` without a model call. Fixed and simulated conditions remain distinct even when their percentages are equal.

The mean of the valid Step 2 replicates is used to calculate:

```text
survey_surprise = treatment_percentage - step2_predicted_percentage
```

Retest and placebo conditions have no treatment percentage, so their `survey_surprise` is `null`. It is also `null` when no valid Step 2 prediction is available.

The special persona label `none` adds no persona instruction and serves as a no-persona-prompt control.

### Replicates

`--replicates N` is a run-level positive integer with a default of `1`. It assigns each selected measurement unit zero-based replicate IDs from `0` through `N - 1`.

For example, `--replicates 3` creates replicate IDs `0`, `1`, and `2` for every Step 1–3 unit and every Step 4 unit-treatment pair. Even a default one-replicate run records `replicate_id: 0`.

Use more than one replicate when repeated stochastic generations are needed. Replicates repeat the same experimental cell with the same model, prompt, and decoding parameters but replicate-specific derived seeds.

Replicates are not additional personas or independent respondents. Deterministic decoding may produce identical responses.

`replicate_id` values are fixed before inference. The replicate index contributes to the derived generation seed and stable `sample_id`; for Step 4, it also contributes to the reproducibly shuffled treatment order.

Within a stage, assignments with the same replicate index share a seed derived from the run-level `--seed`. This permits batching and is intended to support comparisons using common random seeds while each cell retains its own `sample_id`.

Step 4 replicate IDs are not paired one-to-one with baseline replicate IDs. After Steps 1 and 2 run, valid Step 1 decisions are aggregated to resolve simulated consensus and valid Step 2 predictions are averaged.

These aggregates resolve the corresponding Step 4 metadata; baseline and Step 4 replicates are not matched by ID.

Replication is separate from the no-information retest, which is a Step 4 treatment condition. Increasing `N` multiplies each stage's planned logical-assignment count.

## Measurement pipelines

### Verbalize

The verbalize pipeline requests a standalone JSON object. It accepts either the object itself or exactly one JSON code fence containing it, without surrounding prose.

Yes/No answers are normalized to the two-value enum. Step 2 accepts a finite number or complete percentage string in `[0, 100]`; out-of-range values are not clamped.

The pipeline can use local vLLM or an OpenRouter-compatible API. Local vLLM runs disable chat-template thinking so the response can satisfy the standalone JSON contract.

### Logprob

The logprob pipeline uses local vLLM only. Each executable binary observation follows a two-phase protocol:

1. generate visible analysis without a final answer;
2. close that assistant turn, add a new user request for exactly `Yes` or `No`, and score the first token of the fresh assistant response.

The manifest identifies this protocol as `completed_analysis_new_user_fresh_assistant_v1`. Runs created with a different bounded-scoring protocol are not interchangeable or resumable under this protocol.

The tokenizer is used to discover supported single-token Yes/No spellings. Multi-token variants are excluded, and the runner requests scores only for the bounded candidate set rather than the full vocabulary.

Reported `probabilities.Yes` and `probabilities.No` are conditional on that candidate set. Each valid score also reports `candidate_mass`, `residual_mass`, candidate token scores, `sampled_choice`, and `format_valid`.

Finite-precision candidate-mass overshoot up to the manifest-recorded absolute tolerance of `1e-6` is treated as a numeric boundary effect and clamped to one. Larger overshoot remains `INVALID`; the scorer and checkpoint validator use the same tolerance.

A missing candidate score, an unsupported sampled token, or truncated Phase 1 analysis is recorded as `INVALID`; malformed backend responses are recorded as `ERROR`. The code never substitutes `0.5/0.5` for an invalid observation.

## Data

The tracked data contain:

- 32 politician labels and 7 platform labels;
- 18 policy categories and 136 proposals;
- 408 actions, with `Personal Commitment`, `Public Advocacy`, and `Strategic Support` for each proposal.

`entities.json` and `policy_options.json` were adapted from [Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs](https://arxiv.org/abs/2502.08640).

The action text was generated by an LLM and has not been human-validated for cost, feasibility, ideological neutrality, or cross-persona comparability.

## Installation

The project requires Python `>=3.11,<3.14`. The installation command below targets Linux with an NVIDIA GPU, driver, and CUDA stack supported by the pinned `vllm==0.24.0` release.

Use the [vLLM 0.24 GPU installation guide](https://docs.vllm.ai/en/v0.24.0/getting_started/installation/gpu/) to verify platform compatibility. Then install all runtime dependencies from the single requirements file:

```bash
git clone https://github.com/smyft/LLM-Political-Belief-Update.git
cd LLM-Political-Belief-Update

conda create --name llm-belief-update python=3.11 -y
conda activate llm-belief-update
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Model weights, KV cache, CUDA graphs, scheduler concurrency, and runtime overhead must all fit in available GPU memory. Choose `--max-model-len`, `--max-num-seqs`, tensor parallelism, and the model size accordingly.

## API configuration

API mode is optional and applies only to the verbalize pipeline and proposal-to-action generator. Copy the placeholder environment file and add your OpenRouter key locally:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Do not commit `.env` or print real credentials. Keys are not written to manifests or result files.

For local models, `--trust-remote-code` is disabled by default. Enabling it requires a pinned `--code-revision` or `--model-revision`.

## Dry run

Both runners support `--dry-run`. It validates the data, prompts, and deterministic sampling plan without loading a model or creating checkpoints.

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model MODEL_ID \
  --max-base-units 12 \
  --replicates 2 \
  --dry-run

python -m src.experiment.logprob_experiment_runner \
  --model MODEL_ID \
  --max-base-units 12 \
  --replicates 2 \
  --dry-run
```

`--max-base-units` is an action-level budget applied before Step 1 or Step 2 execution. The selected subset is deterministic for the run-level `--seed`.

Dry-run logical sample counts are exact. Backend sequence counts are upper bounds because invalid prerequisites can skip some simulated-consensus or Phase 2 calls.

## Run experiments

Replace `MODEL_ID` with a vLLM-compatible local model identifier.

### Verbalize with local vLLM

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model MODEL_ID \
  --max-tokens 256 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-base-units 12 \
  --results-dir results
```

### Verbalize with an API

Replace `API_MODEL_ID` with a model identifier accepted by the configured provider.

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model API_MODEL_ID \
  --use-api \
  --max-base-units 12 \
  --results-dir results
```

### Logprob with local vLLM

```bash
python -m src.experiment.logprob_experiment_runner \
  --model MODEL_ID \
  --max-tokens 256 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-base-units 12 \
  --results-dir results
```

Both runners support filters such as `--personas` and `--categories`, treatment switches, `--replicates`, `--seed`, checkpoint chunk size, model revisions, and local vLLM resource controls. Use `--help` for the authoritative option list.

For reproducible local runs, pin `--model-revision`; tokenizer and remote-code revisions inherit it unless their specific revision flags are supplied.

For text-only use of a supported multimodal checkpoint, add `--language-model-only`.

## Checkpoints and output

Each real run creates immutable checkpoint shards and one compiled result:

```text
results/
├── checkpoints/<run_id>/
│   ├── manifest.json
│   ├── step1/chunk_00000000.json
│   ├── step2/chunk_00000000.json
│   └── ...
└── results_<run_id>.json
```

The manifest binds the model, experimental configuration, sampling plan, prompts, validated data hashes, source version, and expected sample IDs. Resume uses the stored manifest configuration rather than replacing it with new experimental flags.

List runs without loading a model:

```bash
python -m src.experiment.logprob_experiment_runner \
  --results-dir results \
  --list-checkpoints
```

Resume from a logical stage:

```bash
python -m src.experiment.logprob_experiment_runner \
  --results-dir results \
  --resume-from RUN_ID \
  --resume-step step4a
```

Use a new run when changing a manifest-bound setting such as the model, seed, replicate count, treatments, prompts, data, or scoring protocol. See [CHECKPOINT_RESUME_GUIDE.md](CHECKPOINT_RESUME_GUIDE.md) for checkpoint format and recovery rules.

Every observation has a `VALID`, `INVALID`, or `ERROR` status. Invalid and error observations remain explicit and are not imputed as Yes, No, or 50 percent.

The compiled file contains the manifest, per-stage status summaries, and action-level rows. Proposal-level records may appear in multiple action rows; use `sample_id` to identify the same underlying observation.

## Default experiment scale

The complete default plan uses all 40 persona conditions, 136 proposals, 408 actions, eight Step 4 treatments, and one replicate.

| Quantity | Count |
| --- | ---: |
| Proposal-level units | 5,440 |
| Action-level units | 16,320 |
| Step 1 logical samples | 5,440 |
| Step 2 logical samples | 5,440 |
| Step 3 logical samples | 16,320 |
| Step 4a logical samples | 43,520 |
| Step 4b logical samples | 130,560 |
| Total logical samples | 201,280 |
| Verbalize backend-sequence ceiling | 201,280 |
| Logprob backend-sequence ceiling | 397,120 |

The logprob ceiling counts up to two generated sequences for each binary observation. Additional replicates multiply all logical sample and sequence counts.

## Optional action-data regeneration

The repository already includes `data/proposal_actions.json`. Regeneration is optional, uses a paid API, and requires an explicit output path.

```bash
python -m data.proposal2action \
  --model MODEL_ID \
  --output results/proposal_actions.generated.json \
  --debug 3
```

Existing output requires `--overwrite`. Interrupted runs preserve validated progress in a partial checkpoint that can be continued with `--resume` when its configuration still matches.

Do not overwrite the tracked canonical file unless you intend to regenerate the complete dataset and review the result.
