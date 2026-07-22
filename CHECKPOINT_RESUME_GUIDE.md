# 断点继续运行指南（Checkpoint Resume Guide）

## 1. 适用范围

本文说明 `verbalize` 与 `logprob` 两个实验 runner 的版本化 checkpoint 和恢复机制。未传 `--results-dir` 时，默认结果目录是**当前工作目录**下的 `results/`；从源码仓库根目录运行时，它自然就是仓库根下的 `results/`。本文的真实运行与恢复示例仍显式传入 `--results-dir`，避免依赖启动时所在目录。

`data/proposal2action.py` 使用的是另一套相邻 `*.partial.json` 机制，不使用本文所述的 stage checkpoint。其 schema version 2 会校验模型、输入、prompt、选择计划、固定生成参数、规范化 OpenRouter endpoint，以及 generator 与统一 model interface 的源码 fingerprint；旧 schema 不兼容。该脚本的恢复方法见 README 的 “Optional action-data regeneration” 部分。

## 2. 核心语义

恢复不是“重新解释一批旧 JSON”，而是继续同一个不可变实验计划：

- 每次新运行先生成抗碰撞的 `run_id` 和 `manifest.json`；
- manifest 固定 pipeline、模型与后端配置、数据与 prompt 哈希、代码版本、选择计划、treatment 设计及每个 stage 的预期 `sample_id`；
- 每个逻辑 stage 按 chunk 写入不可变 JSON 分片；
- 恢复时严格验证 manifest、当前源码/数据/prompt 与 stage 依赖；
- 已存在的 `sample_id` 被跳过，缺失的 ID 才会执行；
- 结果按 `sample_id` 连接，不按列表位置连接。

checkpoint 的目标是安全地补齐**缺失记录**。它不是修改已记录观测、覆盖错误结果或改变实验设计的工具。

## 3. 文件布局

一次默认运行会生成：

```text
results/
├── checkpoints/
│   └── <run_id>/
│       ├── .checkpoint.lock
│       ├── manifest.json
│       ├── step1/
│       │   ├── .sample-index.sqlite3
│       │   ├── chunk_00000000.json
│       │   └── chunk_00000001.json
│       ├── step2/
│       │   ├── .sample-index.sqlite3
│       │   └── chunk_00000000.json
│       ├── step3/
│       │   └── ...
│       ├── step4a/
│       │   └── ...
│       └── step4b/
│           └── ...
└── results_<run_id>.json
```

`run_id` 的格式类似：

```text
verbalize-20260720T123456.123456Z-a1b2c3d4
logprob-20260720T123456.123456Z-a1b2c3d4
```

时间使用 UTC，末尾 UUID 随机片段使同一时刻启动的运行极难碰撞，但这不是数学上的绝对无碰撞保证；若目标 run 目录已经存在，严格的既有运行检查才是最终防线。`sample_id` 同样是基于规范化身份内容的 SHA-256 抗碰撞标识，不应描述为绝对不可能碰撞。

## 4. Manifest 固定了什么

`manifest.json` 与 stage chunks 的 checkpoint schema version 当前为 `2`。Version 1 与 version 2 明确不兼容：旧运行应保留作审计，不能由当前 loader 直接恢复或静默升级。存储 schema 与 measurement protocol 是两层独立契约；两个 run 即使同为 v2，也可能因为 scoring dialogue/protocol 不同而不能恢复或比较。主要字段包括：

| 字段 | 含义 |
| --- | --- |
| `run_id`, `pipeline`, `created_at` | 运行身份 |
| `config` | 模型、backend、temperature、max tokens、seed、replicates、treatment 开关、固定百分比、API/vLLM 参数、model/tokenizer/remote-code revisions、源码树哈希，以及 logprob 的 `binary_estimator`、`scoring_temperature`、`thinking_mode`、`bounded_scoring_protocol` 等 |
| `config_fingerprint` | `config` 的内容哈希 |
| `data_hashes`, `data_fingerprint` | `entities.json` 与 `proposal_actions.json` 的内容哈希 |
| `prompt_hashes`, `prompt_fingerprint` | 该 pipeline 全部必需 prompt 的内容哈希 |
| `sampling_plan` | 已选择的 proposal/action units 与 treatment 设计 |
| `expected_sample_ids` | Step 1、2、3、4a、4b 的完整预期 ID 列表 |
| `code_version` | Git commit；工作树非干净时带 `-dirty` |
| `run_fingerprint` | 上述运行身份的总哈希 |

创建运行前，`--max-base-units` 已经应用于 action-level 候选；Step 1/2 proposal units 随后才从所选 action units 去重得到。因此，小预算不会先触发全量 Step 1/2。

Step 4 的固定条件、simulated consensus 占位、retest、placebo、replicate、seed 和顺序都在第一次模型调用前形成稳定 ID。实际 simulated-consensus 百分比和 `survey_surprise` 在 Step 1/2 完成后解析，但不会改变既有 ID。所有 treatment-planning 百分比使用统一的六位小数契约：fixed values 在去重前量化，condition、simulated consensus 与 `survey_surprise` 以同一精度保存，prompt 使用去掉尾随零的定点文本而非科学计数法。

共识先在每个其他 persona 内平均有效 replicates，再对有贡献的 personas 等权；metadata 中 `consensus_persona_n` 与兼容字段 `consensus_n` 都表示贡献 persona 数，而不是有效 response 条数。logprob 记录只有在 `format_valid=true` 且 `sampled_choice` 为 `Yes`/`No` 时才贡献二元决策；即使条件 Yes/No 概率恰好相等，也按实际 sampled choice 计数，不用 argmax 制造伪标签。

## 5. Stage DAG

恢复依赖按最终编译所需的完整 DAG 定义：

```text
step1
  └── step2
        └── step3
              └── step4a
                    └── step4b
```

精确依赖为：

| 从该 stage 恢复 | 必须已完整存在的前置 stage |
| --- | --- |
| `step1` | 无 |
| `step2` | `step1` |
| `step3` | `step1`, `step2` |
| `step4a` | `step1`, `step2`, `step3` |
| `step4b` | `step1`, `step2`, `step3`, `step4a` |

选择 `--resume-step step4a` 时，系统加载并严格验证 Step 1–3，然后补齐 Step 4a，再补齐 Step 4b。选择 `step4b` 时不会重新运行 Step 4a，因此 Step 4a 必须完整。

## 6. Chunk 格式与不可变性

每个 `chunk_XXXXXXXX.json` 包含：

```json
{
  "schema_version": 2,
  "run_id": "...",
  "run_fingerprint": "...",
  "stage": "step3",
  "chunk_index": 0,
  "written_at": "...+00:00",
  "records_sha256": "<canonical-record-list SHA-256>",
  "records": []
}
```

每个 record 是自包含观测：

```text
sample_id, stage, metadata, status, value,
error_code, error_message, raw_response
```

写入规则：

- 临时文件在同一目录创建，`fsync` 后通过原子 rename 发布；
- 同一 run 的 manifest/chunk 写入由线程锁和 Unix 文件锁串行化；
- 已存在的同编号 chunk 先必须通过完整校验；其规范化 record 列表与待写 record 列表一致时才幂等接受，record 内容不同则失败；
- 同一个 `sample_id` 不得出现在多个 chunk；
- manifest 一旦已有任何 chunk，就不能替换为不同 manifest；
- loader 校验 `records_sha256`，并对 record 的 stage、pipeline、metadata、status 与 value 进行语义验证；digest 一致本身不能替代这些检查；
- loader 拒绝损坏 JSON、重复 key、`NaN`/`Infinity`、错误 stage、错误 fingerprint、未知/重复 ID 和错误 chunk 文件名；run 目录、manifest、lock、stage 目录、chunk 文件与 SQLite sample index 都不允许是 symbolic link。

每个 stage 还维护一个派生的 `.sample-index.sqlite3` 来辅助 duplicate-ID 检查。它不是 checkpoint 事实来源：manifest 与不可变 JSON chunks 才是权威数据。冷启动，或当前 `CheckpointStore` 之外改变了 shard/index 签名时，实现会重读并严格验证全部历史 JSON shards，再把权威 chunk/sample 映射与 SQLite 内容比较；索引缺失、损坏或任何不一致都会从 JSON 自动重建。同一 `CheckpointStore` 实例中，未变化的 warm authority cache 会先校验文件签名和索引摘要元数据，正常追加无需反复读取历史 shard 内容；任何不匹配都退回 JSON 权威全量扫描。在没有 writer 运行时可以安全删除索引。

不要手工编辑、移动或合并 manifest 与 JSON chunks。手工改动通常会使严格验证失败，而不是被静默接受。SQLite 索引被明确设计为可从 JSON shards 重建的派生缓存；只有在确认没有 writer 运行时才能删除它。`.checkpoint.lock` 也不是记录事实来源，但它是并发协调文件，不应在 writer 运行期间操作。

## 7. 开始运行

建议先 dry-run；它不加载模型，也不创建 manifest：

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model openai/gpt-4.1-mini \
  --use-api \
  --max-base-units 12 \
  --dry-run
```

dry-run 的 logical sample counts 是计划中的精确数量；backend sequence counts 是保守上界。若前序输出无效，simulated-consensus cell 或 logprob Phase-2 bounded scoring 可能不调用 backend，因此实际请求数可以更少。

两个 CLI 都把 `--dry-run` 与 `--resume-from` 定义为显式互斥；不能用 dry-run 静默忽略一个恢复请求。

真正运行后，终端会打印最终文件路径，其中含 `run_id`：

```bash
python -m src.experiment.verbalize_experiment_runner \
  --model openai/gpt-4.1-mini \
  --use-api \
  --max-base-units 12 \
  --results-dir results
```

logprob 示例：

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

logprob runner 显式关闭 chat-template thinking。每个 binary observation 先生成并关闭一个可见 assistant analysis turn，再追加新的 user 指令，最后只评分新 assistant 回答的首 token；它不使用 `reasoning_content`，也不继续一个尚未关闭的 assistant message。manifest 通过 `bounded_scoring_protocol=completed_analysis_new_user_fresh_assistant_v1` 固定这套协议，旧 dialogue/scoring 协议的 checkpoint 不能混用或恢复。

runner 在第一个可执行 chat（包括 Step 2）之前初始化 backend、只运行一次 `preflight_bounded_scoring()`。bounded-scoring/tokenizer/API/disabled-thinking contract 不兼容是 run-level fatal。普通 chat 或 bounded scoring 的整批本地 vLLM 异常（包括 OOM）同样是 fatal：当前 chunk 不发布，ID 保持缺失。修正原因后可以安全恢复；较小的 `--chunk-size` 可能解决 batch-size/OOM，但若必须改变 manifest-bound 配置则应新建 run。终端错误会打印 run ID。backend 已返回对齐 batch 后，单条 response schema 错误仍记录为不可变 `ERROR`。没有可执行 assignment 时不会初始化。phase 1 的 `finish_reason=length` 记为 `INVALID/phase1_truncated`，Step 2 对应记为 `INVALID/step2_truncated`；phase 1 保守识别到显式最终 Yes/No 时记为 `INVALID/phase1_contains_final_answer`，这些样本都不会进入后续 bounded scoring/解析。

本地 vLLM verbalize 同样固定 `enable_thinking=False`，避免 reasoning 前后缀破坏 standalone JSON；API verbalize 的 reasoning 行为由 provider 管理。两个本地 runner 都把 `max_model_len`、`max_num_seqs`、`language_model_only` 和 thinking contract 固定进 manifest，恢复时不能通过命令行改写。

这些 revision flags 只适用于 local vLLM，与 `--use-api` 同用会被拒绝；hosted API 版本必须通过 provider model identifier 选择。如果 local model、tokenizer 与 remote code 使用同一个不可变 revision，只传 `--model-revision` 即可；`--tokenizer-revision` 和 `--code-revision` 默认继承它。三者不同时应分别指定。启用 `--trust-remote-code` 时必须至少通过 `--code-revision` 或 `--model-revision` 固定所执行的代码，否则 runner 会拒绝启动。

`--chunk-size` 默认 `128`，表示同一 generation seed 分组内，一个 stage 分片最多处理的 assignment 数。较小 chunk 可减少进程在分片发布前中断时需要重算的工作，但会产生更多文件。

## 8. 查看 checkpoint

以下命令不需要 `--model`，也不会加载模型：

```bash
python -m src.experiment.verbalize_experiment_runner \
  --results-dir results \
  --list-checkpoints
```

```bash
python -m src.experiment.logprob_experiment_runner \
  --results-dir results \
  --list-checkpoints
```

两个命令只显示各自 pipeline 的运行。正常条目包含：

```json
{
  "run_id": "verbalize-...",
  "pipeline": "verbalize",
  "model": "openai/gpt-4.1-mini",
  "created_at": "...",
  "missing_by_stage": {
    "step1": 0,
    "step2": 0,
    "step3": 0,
    "step4a": 12,
    "step4b": 36
  }
}
```

如果 manifest 或 chunk 已损坏，列表会给该 run 返回 `error`，而不是假装它可恢复。

`missing_by_stage` 只计算缺失 ID。已经保存的 `INVALID` 或 `ERROR` record 都不算缺失。

## 9. 恢复命令

从 Step 4a 继续 verbalize API 运行：

```bash
python -m src.experiment.verbalize_experiment_runner \
  --results-dir results \
  --resume-from VERBALIZE_RUN_ID \
  --resume-step step4a
```

从 Step 3 继续 logprob 运行：

```bash
python -m src.experiment.logprob_experiment_runner \
  --results-dir results \
  --resume-from LOGPROB_RUN_ID \
  --resume-step step3
```

两个 CLI 在提供 `--resume-from` 时都可以省略 `--model`；只有新 run 和 dry-run 要求显式模型。恢复必须从尚未初始化或注入 backend 的 fresh runner 开始；测试或嵌入程序如需自定义 backend，应传入 lazy factory，让 manifest 配置先恢复再实例化。恢复过程中，manifest 会恢复并覆盖模型、backend、revision、treatment、seed、replicate 等实验配置，再重建并核对 fingerprint。重新传入这些实验参数不会把旧 run 改成新设计。

只有少数操作性参数不属于实验身份，例如当前 `--results-dir`、`--chunk-size` 和进度显示。`--results-dir` 必须指向包含目标 `checkpoints/<run_id>` 的目录。

## 10. 如何选择 `--resume-step`

- 如果 Step 1 分片不完整，使用 `step1`；
- 如果 Step 1 完整而 Step 2 不完整，使用 `step2`；
- 如果 Step 1/2 完整而 Step 3 不完整，使用 `step3`；
- 如果 Step 1–3 完整而 Step 4a 不完整，使用 `step4a`；
- 只有 Step 4b 不完整时，使用 `step4b`。

从较早 stage 恢复是允许的。该 stage 中已经存在的 ID 会被跳过，缺失 ID 被补齐，随后各 stage 依次完成并重新编译最终结果。

从较晚 stage 恢复不会放宽依赖。任何前置 stage 缺失、重复、损坏或身份不一致都会产生非零退出码。

## 11. 恢复时的严格一致性检查

恢复会检查：

1. `run_id` 格式和路径不能逃逸 checkpoint 根目录；
2. manifest/chunk schema 必须为 v2（v1 明确不兼容）、嵌套 fingerprint 与总 fingerprint；
3. pipeline 必须与所用 runner 一致；
4. 当前 Git/source version 必须与 manifest 一致；
5. 当前数据文件的精确已验证 byte snapshots、对应哈希与 prompt 内容哈希必须一致；
6. scoring dialogue/protocol、选择计划、treatment 设计和所有 expected sample IDs 必须可精确重建；
7. 所有前置 stage 必须完整；
8. 所有 chunk 必须属于同一 run、fingerprint 和 stage，`records_sha256` 必须匹配；
9. 已保存 record 的 metadata 必须与当前重建 assignment 精确一致，status/value 还要通过 pipeline/stage 语义验证；
10. SQLite index 必须与权威 JSON shards 一致，否则自动重建；
11. 不允许缺失、未知、重复或 wrong-stage records，也不跟随 run/manifest/lock/stage/chunk/index symbolic links。

DataLoader 从同一份 bytes 完成解析、严格验证与 SHA-256，只有全部成功后才发布缓存；对外返回 defensive copies。manifest 创建/恢复会再次检查当前文件是否相对已缓存 snapshot 漂移。若发生漂移，应创建 fresh runner，不得让旧缓存与新磁盘哈希混合。

因此，下列修改后通常不能恢复旧 run：

- 修改 Python 源码；
- 修改 prompt；
- 修改 `entities.json` 或 `proposal_actions.json`；
- 改模型、backend、model/tokenizer/remote-code revision、temperature、seed、replicates、固定百分比或 treatment 开关；
- 改 binary scoring dialogue、thinking mode 或 `bounded_scoring_protocol`；
- 手工修改 manifest、sample IDs 或 chunk。

这些情况应创建新 run。旧 assistant-content continuation 或 `assistant_reasoning_content` checkpoint 不能由当前 fresh-assistant 协议恢复。不要通过编辑 manifest 补字段或绕过检查；那会破坏 fingerprint，并错误描述旧 run 实际使用的条件上下文。

## 12. `VALID`、`INVALID` 与 `ERROR` 对恢复的影响

- `VALID`：测量契约通过；
- `INVALID`：样本存在，但回答或 treatment 条件不满足测量契约；不会填成 Yes/No/50%；
- `ERROR`：已取得可归属到单个 sample 的 response container 或执行路径失败。

三种状态都是已保存 record，因此都属于“该 `sample_id` 已完成”。恢复只补缺失 ID，不会覆盖 `INVALID` 或 `ERROR`。

这条规则尤其重要：**如果已发布 chunk 中已有 `ERROR`，必须创建新 run 才能重试该观测。** 直接再次 `--resume-from` 会跳过它。两个 CLI 在最终结果包含 `ERROR` 时返回非零状态，并明确提示新建 run。logprob 的整批 vLLM 异常不会发布这种 `ERROR` chunk，因此是“缺失 ID”场景，可以恢复。

`INVALID` 可以是有效的实验观测，例如模型输出不符合回答格式；它应进入缺失/拒答分析，而不是自动重试到得到 Yes/No 为止。

## 13. 中断场景

### 在 manifest 写入后、首个 chunk 前中断

使用同一 `run_id` 从 `step1` 恢复。由于没有已完成 ID，Step 1 从头执行。

### 在一个 stage 的若干 chunk 完成后中断

选择该 stage 作为 `--resume-step`。已发布 chunk 中的 ID 被跳过，缺失 ID 继续执行。

### 在模型处理 chunk 期间强制退出

chunk 只有完整产生 records 后才原子发布。尚未发布的 chunk 工作可能需要重算。缩小 `--chunk-size` 可减少这一窗口。

### 所有 checkpoint 完成，但最终结果文件缺失

从 `step4b` 恢复。所有 ID 会被发现为已存在，然后系统重新执行严格编译并原子写入 `results_<run_id>.json`。

### 结果含 backend `ERROR`

保留旧 run 作为审计记录，并以相同设计启动一个**新 run**。不要删除某个 record 或手改 chunk 来迫使旧 run 重试。

## 14. 最终编译

当五个 stage 的预期 ID 均存在后，runner 通过稳定 ID 做严格 join，并写入：

```text
results/results_<run_id>.json
```

文件包含：

- schema version、run ID、pipeline、model、run fingerprint；
- 完整 manifest；
- 每个 stage 的 expected/observed/valid/invalid/error/missing 数量和比率；
- action-level 编译结果。

Step 4a 是 proposal-level 推理，编译到 action-level 输出时会在匹配 action rows 中引用同一组 Step 4a records；这不是重复模型调用。编译器拒绝不完整 stage、未知 unit 或 positional mismatch。

## 15. 常见错误

### `manifest not found`

- 检查 `--results-dir` 是否正确；
- 检查 `run_id` 是否完整；
- 使用对应 pipeline 的 `--list-checkpoints`。

### `checkpoint pipeline is ... expected ...`

使用了错误 runner，例如用 logprob runner 恢复 verbalize run。改用创建该 run 的 pipeline。

### `current Git code version does not match`

源码/工作树状态与创建 run 时不同。恢复原代码状态，或开始新 run。

### `current data, prompts, or reconstructed plan do not match`

数据、prompt、配置、计划或预期 ID 已变化。不要修改 manifest；创建新 run。

### `checkpoint has an incompatible bounded-scoring protocol`

该 logprob run 缺少当前协议字段，或由旧 scoring dialogue 创建。保留旧 run 作审计，并使用当前代码新建 run；不要手工给旧 manifest 补上协议字段。

### `checkpoint is missing sample ID`

选择的 `--resume-step` 太晚，前置 stage 还不完整。根据 `--list-checkpoints` 从最早不完整 stage 恢复。

### `duplicate sample ID` / `unknown sample ID` / `fingerprint mismatch`

checkpoint 可能被手工修改、复制或混合。保留现场做审计，不要继续写入该 run；从可信输入启动新 run。

### 进程成功编译但退出码为 `1`

检查最终文件的 `status_summary`。如果存在 `ERROR` record，结果仍会为审计目的写出，但该 run 不被视为完全成功，且旧 shard 不允许覆盖重试。

## 16. 最佳实践

1. 全量运行前总是执行 `--dry-run`。
2. pilot 使用很小的 `--max-base-units`。
3. 生产运行固定 model、tokenizer 与 remote-code revisions；三者相同时可只用 `--model-revision`，并记录 provider/model 版本。
4. 记录 `run_id`，不要只记录最终文件名。
5. 不编辑 manifest 或 JSON chunk；备份整个 `checkpoints/<run_id>/`。`.sample-index.sqlite3` 只是可重建缓存，不必作为权威数据保存。
6. 把 `INVALID` 和 `ERROR` 作为不同状态分析。
7. 真实 API/GPU 全量实验前先用 fake backend/小模型验证流程。
8. 若要改变 treatment、prompt、数据、模型或 scoring protocol，创建新 run，而不是复用旧 ID。
9. 比较 logprob runs 前核对完全相同的 `bounded_scoring_protocol`、prompt/data hashes 与 model/tokenizer revisions；相同 `sample_id` 不能覆盖协议差异。
