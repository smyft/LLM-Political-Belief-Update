# Political Polarization Experiment

本项目用于研究大语言模型（LLM）在面对“人群意见分布”信息时，如何更新其政治立场与行动支持倾向。

当前仓库的核心代码与数据仅保留在：
- `data/`
- `src/`
- `README.md`
- `requirements.txt`

## 1. 项目结构

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
├── README.md
└── requirements.txt
```

## 2. 核心能力

- 多步骤实验流程（Step1~Step4b）
- Persona 注入（政治人物/平台/none）
- logprob 概率提取（vLLM）
- API 模式调用（OpenRouter 兼容接口）
- Proposal -> Action 数据生成脚本（`data/proposal2action.py`）

## 3. 数据说明

### `data/entities.json`
定义实验 persona（politicians/platforms）。

### `data/policy_options.json`
原始政策提案，按 category 组织。

### `data/proposal2action.txt`
将 proposal 转换为 action 的提示词模板。

### `data/proposal_actions.json`
每条 policy proposal 对应 3 类 action：
- Personal Commitment
- Public Advocacy
- Strategic Support

### `data/proposal2action.py`
读取 `policy_options.json`，调用 API 生成 action，并写回 `proposal_actions.json`（当前默认模型：`google/gemini-3-pro-preview`）。

## 4. 环境安装

推荐 Python 3.8+。

```bash
pip install -r requirements.txt
```

如果你使用 conda（例如环境名 `LLM`）：

```bash
conda activate LLM
pip install -r requirements.txt
```

### API 环境变量

在项目根目录 `.env` 中配置：

```bash
OPENROUTER_API_KEY=your_key
# 可选
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## 5. 使用方式

## 5.1 生成 proposal_actions 数据

```bash
python data/proposal2action.py
```

调试模式（只跑前 N 条）：

```bash
python data/proposal2action.py --debug 1
python data/proposal2action.py --debug 5
```

可选参数：
- `--model`（当前脚本中固定使用 `google/gemini-3-pro-preview`）
- `--delay`（API 调用间隔秒数）

## 5.2 运行 logprob 实验

```bash
python -m src.experiment.logprob_experiment_runner
```

常用参数（见源码 argparse）：
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

## 5.3 运行 verbalize 实验

```bash
python -m src.experiment.verbalize_experiment_runner
```

参数与 logprob runner 类似（不含 logprob/resume 相关参数）。

## 6. 断点继续运行（Checkpoint Resume Guide）

本节整合原 `CHECKPOINT_RESUME_GUIDE.md` 内容。

## 6.1 场景

当 Step1~Step3 已完成，可直接从 Step4a 或 Step4b 继续，无需重跑前序步骤。

中间结果默认位于：

```text
results/intermediate/
```

典型文件名：

```text
{model_name}_{timestamp}_step1_phase2.json
{model_name}_{timestamp}_step3_phase2.json
```

其中 `experiment_prefix` 形如：

```text
meta-llama_Llama-3.1-8B-Instruct_20260303_095200
```

## 6.2 查看可用 checkpoint

```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200 \
  --list-checkpoints
```

## 6.3 从 Step4a 继续运行

```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200 \
  --resume-step step4a
```

`--resume-step` 支持：
- `step1`
- `step2`
- `step3`
- `step4a`
- `step4b`

## 6.4 Python 方式恢复

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

## 6.5 恢复前置条件

恢复到 Step4a/Step4b 时，需要至少具备：
- `step1_phase2` checkpoint
- `step3_phase2` checkpoint

程序会自动检查依赖并报错提示缺失项。

## 6.6 常见问题

1. 找不到 `experiment_prefix`：
   到 `results/intermediate/` 查看文件名前缀。
2. 提示缺 checkpoint：
   先确认 `step1_phase2`、`step3_phase2` 文件存在且未损坏。
3. 最终结果保存位置：
   位于 `results/`（文件名含 model 与 timestamp）。

## 7. 依赖说明

- `torch`, `numpy`: 数值与模型推理基础
- `vllm`: 本地 logprob 推理
- `openai`, `requests`, `python-dotenv`: API 模式与配置加载
- `tqdm`: 进度显示

## 8. 备注

- 若使用 API 模式，请确保 `.env` 中 `OPENROUTER_API_KEY` 可用。
- 若使用 vLLM，请确保 CUDA / 驱动 / 显存满足模型要求。
