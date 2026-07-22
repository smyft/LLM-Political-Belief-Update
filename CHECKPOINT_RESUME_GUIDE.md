# Checkpoint and Resume Guide

## Scope

This guide describes the versioned checkpoint system shared by the `verbalize` and `logprob` experiment runners. Installation, experiment design, and complete run examples are documented in [README.md](README.md).

The proposal-to-action generator uses a separate partial-file recovery mechanism. See "Optional action-data regeneration" in the README for that workflow.

## Recovery model

A checkpoint records an immutable run plan. It is not a way to change an experiment after it has started.

For each real run, the runner:

1. creates a new collision-resistant `run_id` and writes `manifest.json` before model inference;
2. writes immutable JSON shards as records complete across five logical stages;
3. identifies every observation with a stable `sample_id`;
4. resumes by validating the saved plan and executing only missing IDs;
5. compiles the final result by `sample_id`, not response order.

Persisted `VALID`, `INVALID`, and `ERROR` records are all considered complete. Resume never overwrites them.

## Output layout

If `--results-dir` is omitted, the default results directory is `./results` relative to the current working directory.

```text
RESULTS_DIR/
├── checkpoints/
│   └── RUN_ID/
│       ├── .checkpoint.lock
│       ├── manifest.json
│       ├── step1/
│       │   ├── .sample-index.sqlite3
│       │   └── chunk_00000000.json
│       ├── step2/
│       │   └── ...
│       ├── step3/
│       │   └── ...
│       ├── step4a/
│       │   └── ...
│       └── step4b/
│           └── ...
└── results_RUN_ID.json
```

The manifest and JSON chunk shards are the authoritative checkpoint data. Each shard contains its run identity, stage, chunk index, record digest, and records.

`.sample-index.sqlite3` is a derived duplicate-ID index. It can be rebuilt from the JSON shards and is not a substitute for them.

Keep each run's internal layout intact. Do not edit, merge, rename, or selectively delete manifest or chunk files, and do not move individual files between runs or stages. Do not alter checkpoint files while a writer is active.

## Manifest and compatibility

The current manifest and chunk schema version is `2`. Checkpoints using an unsupported schema should be preserved for audit but cannot be resumed by the current loader.

The manifest binds the following information to the run:

| Area | Bound information |
| --- | --- |
| Experiment | Pipeline, model, backend, revisions, generation settings, seed, replicates, and treatments |
| Inputs | Validated data hashes, prompt hashes, and source-tree fingerprint |
| Plan | Selected units, treatment design, and every expected `sample_id` |
| Provenance | Git code version when available, configuration fingerprints, and the overall run fingerprint |

For logprob runs, the measurement protocol is also manifest-bound. The current protocol is `completed_analysis_new_user_fresh_assistant_v1`; a checkpoint created under another scoring protocol cannot be resumed as this protocol.

Resume deterministically reconstructs the stage assignments from the current source, inputs, and saved manifest. Any incompatible difference causes validation to fail rather than silently changing the run.

## Stage dependencies

`--resume-step` is the earliest stage that may execute. Every earlier dependency must already be complete.

| Resume step | Required completed stages |
| --- | --- |
| `step1` | None |
| `step2` | `step1` |
| `step3` | `step1`, `step2` |
| `step4a` | `step1`, `step2`, `step3` |
| `step4b` | `step1`, `step2`, `step3`, `step4a` |

The selected stage and every later stage are processed in order. Existing IDs are skipped, and only missing IDs are executed.

Choose the earliest incomplete stage. Selecting a later stage does not relax its dependencies.

## List checkpoint runs

List verbalize checkpoints:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --results-dir RESULTS_DIR \
  --list-checkpoints
```

List logprob checkpoints:

```bash
python -m src.experiment.logprob_experiment_runner \
  --results-dir RESULTS_DIR \
  --list-checkpoints
```

These commands do not require `--model` and do not initialize a backend. They report each run's identity and the number of missing IDs in every stage.

`missing_by_stage` counts only absent IDs. Persisted `INVALID` and `ERROR` records are not missing. A discovered manifest or shard that fails validation is reported with an `error` field.

## Resume a run

Use the same pipeline runner and the results directory containing the checkpoint.

Resume a verbalize run:

```bash
python -m src.experiment.verbalize_experiment_runner \
  --results-dir RESULTS_DIR \
  --resume-from RUN_ID \
  --resume-step step3
```

Resume a logprob run:

```bash
python -m src.experiment.logprob_experiment_runner \
  --results-dir RESULTS_DIR \
  --resume-from RUN_ID \
  --resume-step step3
```

`--model` may be omitted when listing or resuming checkpoints. It remains required for a new run or dry run.

`--dry-run` and `--resume-from` are mutually exclusive.

On resume, the manifest restores the model, backend, revisions, generation settings, seed, replicate count, treatment design, and API or vLLM settings. Command-line arguments cannot turn the saved run into a different experiment.

Operational options may change without redefining the run. `--results-dir` locates the checkpoint, `--chunk-size` controls backend batch size and the maximum size of newly written shards, and `--no-progress` changes terminal output only.

`--resume-step` defaults to `step1`. Selection options such as `--persona`, `--category`, and `--max-base-units` do not narrow a resumed run; the saved selection is restored. `--data-dir` and `--prompts-dir` may point to relocated inputs only when their validated contents match the manifest hashes.

Credentials are never stored in the checkpoint. Resuming an API run still requires the corresponding credential in the environment.

## Immutable records and retry behavior

| Status | Meaning for resume |
| --- | --- |
| `VALID` | Completed measurement; skipped on resume |
| `INVALID` | Completed but did not satisfy the measurement contract; skipped on resume |
| `ERROR` | Completed record of a sample-level execution failure; skipped on resume |

To retry a persisted `ERROR`, create a new run. Do not delete or alter saved records to force a retry; doing so destroys the run's audit integrity.

Chunks are atomically published only after all records in that chunk are produced and validated. If a process stops before publication, those IDs remain missing and are recomputed on resume.

A fatal batch failure likewise leaves the current chunk unpublished. After correcting the cause, the same run can resume its missing IDs only if the manifest-bound configuration and inputs remain unchanged; otherwise, start a new run. Reducing `--chunk-size` may help when the failure is caused by batch size or memory pressure.

A compiled result containing `ERROR` records is retained for audit, but the command exits with a non-zero status.

## Resume validation

During resume, and before saved records are reused, the runner validates:

- checkpoint schema, pipeline, run identity, and fingerprints;
- the current Git code version when available, the Python source-tree hash, and data and prompt hashes;
- the reconstructed selection, treatment plan, stage assignments, and expected IDs;
- required stage completeness;
- chunk ownership, record digests, IDs, metadata, statuses, and stage-specific values.

Compilation also rejects missing, duplicate, unknown, or wrong-stage records.

Git compatibility is strict when a code version was recorded: resume requires the recorded commit and the same clean/dirty marker. A later commit, including a documentation-only commit, is rejected even when the Python source hash is unchanged.

Start a new run after changing any manifest-bound input, including:

- Python source, prompts, or experiment data;
- model, backend, model/tokenizer/code revision, or generation settings;
- seed, replicate count, selection, treatments, or fixed percentages;
- thinking behavior, binary estimator, or scoring protocol.

Do not edit a manifest to bypass compatibility checks. The old checkpoint should remain an audit record of the experiment that actually produced it.

## Common recovery cases

| Situation | Action |
| --- | --- |
| Manifest exists but no stage shard was published | Resume from `step1` |
| A stage has some completed shards | Resume from that stage if all earlier dependencies are complete |
| The process stopped while producing a shard | Resume from that stage if all earlier dependencies are complete; unpublished IDs are recomputed |
| All stage IDs exist but the compiled result is missing | Resume from `step4b` to validate and compile again |
| A persisted record has status `ERROR` | Preserve the run and start a new run to retry it |
| A required earlier stage is incomplete | Resume from the earliest incomplete stage |

## Final output

After all expected IDs exist, the runner writes:

```text
RESULTS_DIR/results_<RUN_ID>.json
```

The file is written atomically after strict compilation. It contains the manifest, per-stage status summaries, and compiled action-level rows. Proposal-level observations may be referenced from multiple action rows; identical `sample_id` values identify the same underlying observation rather than repeated model calls.

## Troubleshooting

| Error | Resolution |
| --- | --- |
| `manifest not found` | Check `--results-dir`, the complete run ID, and the pipeline-specific checkpoint list |
| `checkpoint pipeline is ... expected ...` | Use the runner that created the checkpoint |
| Code, data, prompt, plan, or fingerprint mismatch | Restore the recorded Git revision, clean/dirty state, and compatible inputs, or start a new run |
| Incompatible bounded-scoring protocol | Preserve the checkpoint for audit and create a new logprob run |
| Missing sample ID or incomplete dependency | Resume from an earlier incomplete stage |
| Duplicate, unknown, wrong-stage ID, or digest mismatch | Do not write further to the run; preserve it for audit and start from trusted inputs |

## Reproducibility practices

1. Use `--dry-run` before creating a new experiment run.
2. Use an explicit `--results-dir` and record the `run_id`.
3. Pin model, tokenizer, and remote-code revisions when using local models.
4. Preserve `manifest.json`, every JSON shard, and the final compiled result together.
5. Treat `.sample-index.sqlite3` as a rebuildable cache, not primary evidence.
6. Never combine files from different runs or edit a checkpoint in place.
7. Start a new run when the experiment design changes or a persisted `ERROR` must be retried.
8. Before comparing runs, verify the source, data, and prompt fingerprints; model and tokenizer revisions; and measurement protocol.
