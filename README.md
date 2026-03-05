# LLM Political Belief Update

This repository studies how Large Language Models (LLMs) update political beliefs and action support when exposed to population-level opinion distributions.

## Repository Scope

This repo intentionally keeps only:
- `data/`
- `src/`
- `README.md`
- `requirements.txt`
- `.gitignore`

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

## What This Project Does

- Runs multi-step political belief experiments (Step 1 to Step 4b)
- Supports persona-based prompting (politicians, platforms, or `none`)
- Supports logprob-based probability extraction with vLLM
- Supports API-based inference through an OpenAI-compatible interface (OpenRouter)
- Provides a data pipeline to convert policy proposals into concrete political actions

## Data Files

### `data/entities.json`
Persona definitions used in experiments (politicians and platforms).

### `data/policy_options.json`
Policy proposals grouped by category.

### `data/proposal2action.txt`
Prompt template for generating action candidates from policy proposals.

### `data/proposal_actions.json`
Expanded dataset where each proposal is paired with action options.

### `data/proposal2action.py`
Generates `proposal_actions.json` from `policy_options.json` using API inference.
Current default model in this script is `google/gemini-3-pro-preview`.

## Installation

Use Python 3.8+.

```bash
pip install -r requirements.txt
```

If you use conda (example env name: `LLM`):

```bash
conda activate LLM
pip install -r requirements.txt
```

## Environment Variables

Create `.env` in project root:

```bash
OPENROUTER_API_KEY=your_key
# optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Usage

### 1) Generate Proposal-to-Action Data

```bash
python data/proposal2action.py
```

Debug mode:

```bash
python data/proposal2action.py --debug 1
python data/proposal2action.py --debug 5
```

### 2) Run Logprob Experiments

```bash
python -m src.experiment.logprob_experiment_runner
```

Key arguments in source:
- `--model`
- `--personas`
- `--categories`
- `--max-experiments`
- `--temperature`
- `--max-tokens`
- `--logprobs`
- `--seed`
- `--results-dir`
- `--debug`
- `--use-api`
- `--prompt-type`
- `--prompts-dir`
- `--resume-from`
- `--resume-step`
- `--list-checkpoints`

### 3) Run Verbalize Experiments

```bash
python -m src.experiment.verbalize_experiment_runner
```

## Checkpoint Resume Guide

This section integrates the previous checkpoint resume instructions directly into the README.

### Why Resume Exists

If Step 1 to Step 3 are already completed, you can resume from Step 4a or Step 4b without re-running earlier steps.

Intermediate files are typically saved under:

```text
results/intermediate/
```

Checkpoint file names follow:

```text
{model_name}_{timestamp}_stepX_*.json
```

Example experiment prefix:

```text
meta-llama_Llama-3.1-8B-Instruct_20260303_095200
```

### List Available Checkpoints

```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200 \
  --list-checkpoints
```

### Resume from Step 4a

```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200 \
  --resume-step step4a
```

Supported `--resume-step` values:
- `step1`
- `step2`
- `step3`
- `step4a`
- `step4b`

### Python API Resume Example

```python
from src.experiment.logprob_experiment_runner import LogprobExperimentRunner

runner = LogprobExperimentRunner(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    results_dir="./results",
    resume_from_checkpoint="meta-llama_Llama-3.1-8B-Instruct_20260303_095200"
)

runner.run_experiments_from_step("step4a")
runner.cleanup()
```

### Minimum Required Checkpoints for Step 4 Resume

At minimum, resuming into Step 4 requires:
- `step1_phase2`
- `step3_phase2`

The runner validates dependencies and reports missing files.

### Troubleshooting

1. `Checkpoint file not found`
- Verify `--resume-from` value matches existing file prefixes under `results/intermediate/`.

2. `Missing required checkpoints`
- Ensure required Step 1/3 files exist and are valid JSON.

3. Unexpected resume behavior
- Confirm you are using the same model and compatible prompts as the original run.

## Dependencies

Main dependencies are listed in `requirements.txt`:
- `torch`, `numpy`
- `vllm`
- `openai`, `requests`, `python-dotenv`
- `tqdm`

## Notes

- API mode requires a valid `OPENROUTER_API_KEY`.
- vLLM mode requires compatible CUDA/driver/GPU resources.
