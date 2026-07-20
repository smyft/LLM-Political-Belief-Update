# 断点继续运行指南 (Checkpoint Resume Guide)

## 概述

本指南说明如何在现有的中间结果基础上，直接从 Step 4a 或 Step 4b 继续运行实验，而无需重新运行 Step 1-3。

## 工作原理

### 中间结果保存
在运行实验时，每一步的结果都会自动保存到 `results/intermediate/` 目录中：

```
results/intermediate/
├── meta-llama_Llama-3.1-8B-Instruct_20260303_095200_step1_phase1.json
├── meta-llama_Llama-3.1-8B-Instruct_20260303_095200_step1_phase2.json
├── meta-llama_Llama-3.1-8B-Instruct_20260303_095200_step2.json
├── meta-llama_Llama-3.1-8B-Instruct_20260303_095200_step3_phase1.json
├── meta-llama_Llama-3.1-8B-Instruct_20260303_095200_step3_phase2.json
└── ...（Step 4a/4b 会在继续运行时生成）
```

**experiment_prefix 格式**: `{model_name}_{timestamp}`
- 例如: `meta-llama_Llama-3.1-8B-Instruct_20260303_095200`

### 断点继续工作流

#### Step 1-3 完成时的状态
```
Step 1: First-order Belief ✓ (phase1 + phase2 完成)
Step 2: Second-order Belief ✓ (完成)
Step 3: Action Support (No Distribution) ✓ (phase1 + phase2 完成)
Step 4a: First-order Belief with Distribution ⏸ (未运行)
Step 4b: Action Support with Distribution ⏸ (未运行)
```

#### 从Step 4a继续
现有的 `step1_phase2` 结果和 `step3_phase2` 元数据被加载，用来：
1. 计算 `step1_yes_ratio`（从Step 1的概率推断）
2. 重建 `step_base_metadata`（所有 persona-proposal-action 组合）
3. 运行 Step 4a 和 Step 4b

## 使用方法

### 方法 1: 查看可用的断点

```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200 \
  --list-checkpoints
```

输出示例:
```
=== Available Checkpoints ===
  [✓] step1_phase1
  [✓] step1_phase2
  [✓] step2
  [✓] step3_phase1
  [✓] step3_phase2
  [✗] step4a_phase1
  [✗] step4a_phase2
  [✗] step4b_phase1
  [✗] step4b_phase2
```

### 方法 2: 从Step 4a继续运行

```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200 \
  --resume-step step4a
```

参数说明:
- `--resume-from`: 前面运行的 experiment_prefix （必需）
- `--resume-step`: 从哪一步继续 (默认: step4a)
  - 可选值: `step4a`, `step4b`

### 方法 3: 在Python代码中使用

```python
from src.experiment.logprob_experiment_runner import LogprobExperimentRunner

# 创建运行器，指定要恢复的checkpoint
runner = LogprobExperimentRunner(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    results_dir="./results",
    resume_from_checkpoint="meta-llama_Llama-3.1-8B-Instruct_20260303_095200"
)

# 从Step 4a继续运行
runner.run_experiments_from_step4a()

# 清理资源
runner.cleanup()
```

## 断点恢复的前置条件

要从 Step 4a/4b 成功恢复，必须存在以下checkpoint文件：

```
✓ step1_phase2.json     - Step 1的主要结果（包含概率）
✓ step1_metadata        - Step 1的元数据（personas和proposals）
✓ step2.json            - Step 2的结果（population预测）
✓ step3_phase2.json     - Step 3的主要结果（包含概率）
✓ step3_metadata        - Step 3的元数据（personas-proposals-actions）
```

系统会自动验证这些文件是否存在，如果缺少任何文件，会给出明确的错误信息。

## 中间结果文件内容

### step1_phase2.json 结构
```json
{
  "experiment_id": "20260303_095200",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "step": "step1_phase2",
  "metadata": [
    {
      "persona": "Donald Trump",
      "category": "climate",
      "proposal": "..."
    },
    ...
  ],
  "results": [
    {
      "probabilities": {
        "Yes": 0.75,
        "No": 0.25
      },
      "logprobs_raw": {
        "Yes": -0.288,
        "No": -1.386
      }
    },
    ...
  ]
}
```

### step3_phase2.json 结构
```json
{
  "experiment_id": "20260303_095200",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "step": "step3_phase2",
  "metadata": [
    {
      "persona": "Donald Trump",
      "category": "climate",
      "proposal": "...",
      "action_type": "protest",
      "action": "..."
    },
    ...
  ],
  "results": [
    {
      "probabilities": {
        "Yes": 0.65,
        "No": 0.35
      },
      "logprobs_raw": {...}
    },
    ...
  ]
}
```

## 工作流示例

### 场景：两次运行

**第一次运行（Step 1-3）：**
```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --max-experiments 1000
# 中途因服务器限制停止...
# 生成的 experiment_prefix: meta-llama_Llama-3.1-8B-Instruct_20260303_095200
```

**第二次运行（Step 4a/4b）：**
```bash
python -m src.experiment.logprob_experiment_runner \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --results-dir ./results \
  --resume-from meta-llama_Llama-3.1-8B-Instruct_20260303_095200
# 自动从 Step 4a 开始，加载之前的结果
```

## 关键特性

### 1. 自动验证
系统会自动检查所有必需的checkpoint文件是否存在。

### 2. 结果复用
- Step 1-3 的已保存结果被直接使用
- Step 1 的条件分布用于推断 Step 4 的分布参数
- 无需重新计算，节省时间

### 3. 结果一致性
- 所有结果保持相同的 `experiment_id`
- 最终编译的结果包含 Step 1-4b 的全部内容
- 保证experiment_id的一致性，便于数据追踪

### 4. 错误恢复
如果checkpoint文件缺失或损坏，会提供详细的错误信息。

## 常见问题

### Q1: 如何找到我之前运行的 experiment_prefix？
A: 查看 `results/intermediate/` 目录下的文件名。prefix 是文件名中的 model_name 和 timestamp 部分。

例如，`meta-llama_Llama-3.1-8B-Instruct_20260303_095200_step1_phase2.json` 的 prefix 是：
```
meta-llama_Llama-3.1-8B-Instruct_20260303_095200
```

### Q2: 能否从 Step 4b 恢复？
A: 可以，使用 `--resume-step step4b` 参数。系统仍会运行 Step 4a 和 Step 4b，但如果 Step 4a 结果已存在，可以进一步优化。

### Q3: 如果我想修改 Step 4 的参数（如分布百分比）？
A: 当前版本不支持此操作。如需修改，必须从 Step 1 重新开始。未来版本可能会添加此功能。

### Q4: 最终结果会保存在哪里？
A: 最终的编译结果保存在 `results/` 目录下，文件名格式为：
```
results_meta-llama_Llama-3.1-8B-Instruct_20260303_095200.json
```

## 故障排查

### 错误：Found: Checkpoint file not found
**原因**: 指定的 experiment_prefix 不存在或拼写错误

**解决方案**:
1. 确认 experiment_prefix 的拼写
2. 检查文件是否存在：`ls results/intermediate/`
3. 再次运行 `--list-checkpoints` 验证

### 错误：Cannot resume: Missing required checkpoints
**原因**: Step 1-3 的某些checkpoint文件缺失

**解决方案**:
1. 检查 `results/intermediate/` 中的文件
2. 确保至少有 step1_phase2 和 step3_phase2 两个文件
3. 如果文件损坏，必须重新运行 Step 1-3

## 技术实现细节

### 加载机制
```python
def _load_step_results(self, step_name: str) -> Tuple[List[Dict], List[Dict]]:
    """加载特定步骤的结果和元数据"""
    file_path = self.intermediate_dir / f"{self.experiment_prefix}_{step_name}.json"
    # 解析JSON并返回 (results, metadata)
```

### 验证机制
```python
def _check_resume_feasibility(self, resume_step: str) -> bool:
    """检查恢复的可行性"""
    required_steps = {
        "step4a": ["step1_phase2", "step3_phase2"],
        "step4b": ["step1_phase2", "step3_phase2"],
    }
    # 验证所有必需的文件
```

### 恢复流程
```python
def run_experiments_from_step4a(self):
    """从Step 4a恢复"""
    # 1. 验证checkpoint
    # 2. 加载Step 1-3的结果
    # 3. 计算step1_yes_ratio
    # 4. 重建step_base_metadata
    # 5. 运行Step 4a和Step 4b
    # 6. 编译最终结果
```

## 最佳实践

1. **记录 experiment_prefix**: 保存你的 experiment_prefix 以便后续使用
2. **定期备份**: 定期备份 `results/intermediate/` 目录
3. **验证checkpoint**: 在继续前，使用 `--list-checkpoints` 验证所有必需的文件
4. **监控磁盘空间**: 中间结果可能占用大量磁盘空间，管理好磁盘空间

## 版本信息

- **实现版本**: 1.0
- **支持的恢复点**: Step 4a, Step 4b
- **checkpoint格式版本**: 1.0（JSON）

---

如有问题，请查看代码注释或阅读 `src/experiment/logprob_experiment_runner.py` 的实现细节。
