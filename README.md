# LLM Political Belief Update

This repository investigates three connected questions: (1) what political beliefs LLMs express about policy proposals, (2) whether those beliefs translate into support for concrete political actions, and (3) how both beliefs and action support change after introducing social-opinion distributions. The framework also injects persona prompts so the same model can role-play different political entities and be evaluated under consistent conditions.

## Project Structure

```text
.
├── data/
│   ├── entities.json
│   ├── policy_options.json
│   ├── proposal2action.py
│   ├── proposal2action.txt
│   └── proposal_actions.json
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── experiment/
│   │   ├── base_runner.py
│   │   ├── logprob_experiment_runner.py
│   │   └── verbalize_experiment_runner.py
│   ├── models/
│   │   ├── unified_llm_interface.py
│   │   └── vllm_interface.py
│   └── prompts/
│       ├── logprob/
│       └── verbalize/
├── .gitignore
├── README.md
└── requirements.txt
```

## Experimental Design

The experiment is organized into five steps. We provide two extraction pipelines:
- **Logprob runner**: uses token-level log probabilities from open-source models through vLLM.
- **Verbalize runner**: uses direct prompted answers (JSON-style outputs) from API-accessed models.
Both pipelines follow the same task structure.

### Step 1: First-order Belief (Direct Policy Judgment)
- Question: Does the model believe a policy is beneficial to the U.S.?
- Unit: one run per `(persona, proposal)`.
- Output:
  - Logprob runner: `P(Yes)` / `P(No)`.
  - Verbalize runner: parsed Yes/No answer from JSON-like output.

### Step 2: Second-order Belief (Population Prediction)
- Question: What percentage of the U.S. population would support that policy?
- Unit: one run per `(persona, proposal)`.
- Output: predicted percentage in `[0, 100]`.

### Step 3: Action Support Without Distribution
- Question: Will the model support a specific action tied to that policy?
- Unit: one run per `(persona, proposal, action)`.
- Output:
  - Logprob runner: `P(Yes)` / `P(No)`.
  - Verbalize runner: parsed Yes/No answer.

### Step 4a: First-order Belief With Distribution
- Question: Does direct policy judgment shift after adding population-opinion distribution information?
- Unit: one run per `(persona, proposal, action, distribution)`.
- Distribution set: fixed `{10, 30, 50, 70, 90}` plus one inferred percentage from Step 1.

### Step 4b: Action Support With Distribution
- Question: Does action support shift after adding population-opinion distribution information?
- Unit: one run per `(persona, proposal, action, distribution)`.
- Distribution set: same as Step 4a.

## Data Overview

From the current `data/` files:
- `entities.json`: 32 politicians and 7 political platforms.
- `policy_options.json`: 18 categories and 136 policy proposals.
- `entities.json` and `policy_options.json` are adapted from [Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs](https://arxiv.org/abs/2502.08640) (NeurIPS 2025)
- `proposal_actions.json`: 136 proposals expanded into 408 actions.

## Installation

Use Python 3.8+.

```bash
pip install -r requirements.txt
```

## Environment Setup

Create `.env` in project root:

```bash
OPENROUTER_API_KEY=your_key
```

`OPENROUTER_API_KEY` is required whenever API backend is used.

## Usage

### 1) Generate Proposal-to-Action Data (Optional)

```bash
python data/proposal2action.py
```

This script generates actions from policy proposals through API calls, and outputs to `data/proposal_actions.json`.

We already include a `data/proposal_actions.json` file generated with `google/gemini-3-pro-preview`.

## 2) Run Experiments

### A. Logprob Runner
Require **vLLM** installed and compatible CUDA/driver/GPU resources

Run:

```bash
python -m src/experiment/logprob_experiment_runner --prompt-type logprob
```

Key arguments:
- `--model`: LLM name compatible with vLLM.
- `--personas`: subset of personas, `default=None` for all personas.
- `--categories`: subset of policy categories, `default=None` for all categories.
- `--max-experiments`: maximum number of data item to run.
- `--temperature`, `--max-tokens`, `--logprobs`, `--seed`: vllm sampling controls.
- `--results-dir`: output directory.
- `--prompt-type`, `--prompts-dir`: prompt controls, set `--prompt-type` to `logprob` for Logprob Runner.
- `--debug`: small-scale run.
- `--resume-from`, `--resume-step`, `--list-checkpoints`: checkpoint resume controls.

### B. Verbalize Runner
Require `.env` contains a valid `OPENROUTER_API_KEY`.

Run:

```bash
python -m src/experiment/verbalize_experiment_runner --use_api --prompt-type verbalize
```

Arguments same as Logprob Runner.

### Implementation Note

In the current codebase, both runner `main()` functions include preset local test arguments via `parse_args(cmd)`. If you want normal CLI behavior, change them to `parse_args()`.
