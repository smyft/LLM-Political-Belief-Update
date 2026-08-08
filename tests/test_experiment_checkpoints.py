import json
import math
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.experiment.checkpoints import (
    CheckpointStore,
    CheckpointValidationError,
    RunManifest,
    atomic_write_json,
    dependencies_for,
    validate_record_ids,
)
from src.experiment.core import (
    ExperimentRecord,
    ResultStatus,
    canonical_json,
    sha256_text,
)
from src.models.binary_logprob import LOGPROB_NUMERIC_TOLERANCE


_DEFAULT_VALUE = object()


def record(sample_id, stage, value=_DEFAULT_VALUE):
    if value is _DEFAULT_VALUE:
        value = 50.0 if stage == "step2" else "Yes"
    return ExperimentRecord(
        sample_id=sample_id,
        stage=stage,
        metadata={"sample": sample_id},
        status=ResultStatus.VALID,
        value=value,
    )


def manifest(expected=None, *, pipeline="verbalize"):
    expected_ids = {
        "step1": ("s1-a", "s1-b"),
        "step2": ("s2",),
        "step3": ("s3",),
        "step4a": ("s4a",),
        "step4b": ("s4b",),
    }
    if expected is not None:
        expected_ids.update(expected)
    return RunManifest.create(
        run_id="run-001",
        pipeline=pipeline,
        config={"model": "fake", "seed": 42},
        data_hashes={"data": sha256_text("data")},
        prompt_hashes={"step1": sha256_text("prompt")},
        sampling_plan={"base_units": [{"id": "base"}]},
        expected_sample_ids=expected_ids,
        code_version="deadbeef",
        created_at="2026-01-01T00:00:00+00:00",
    )


def logprob_value(*, sampled_choice="Yes", format_valid=True):
    candidate_mass = 0.8
    yes_probability = 0.75
    sampled_token_id = {"Yes": 11, "No": 22}.get(sampled_choice)
    return {
        "probabilities": {"Yes": yes_probability, "No": 1.0 - yes_probability},
        "label_logprobs": {
            "Yes": math.log(candidate_mass * yes_probability),
            "No": math.log(candidate_mass * (1.0 - yes_probability)),
        },
        "candidate_mass": candidate_mass,
        "residual_mass": 1.0 - candidate_mass,
        "candidates": [
            {
                "token_id": 11,
                "choice": "Yes",
                "decoded_token": "Yes",
                "logprob": math.log(candidate_mass * yes_probability),
                "probability": candidate_mass * yes_probability,
            },
            {
                "token_id": 22,
                "choice": "No",
                "decoded_token": "No",
                "logprob": math.log(candidate_mass * (1.0 - yes_probability)),
                "probability": candidate_mass * (1.0 - yes_probability),
            },
        ],
        "sampled_token_id": sampled_token_id,
        "sampled_choice": sampled_choice,
        "format_valid": format_valid,
        "analysis_text": "visible analysis",
        "analysis_text_kind": "model_generated_visible_text",
        "estimator": "bounded_single_token_candidate_set",
        "conditional_on_candidate_set": True,
        "scoring_temperature": 0.0,
        "finish_reason": "length",
    }


class ManifestTests(unittest.TestCase):
    def test_manifest_round_trip_and_fingerprint_validation(self):
        original = manifest()
        restored = RunManifest.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)

        tampered = original.to_dict()
        tampered["config"]["seed"] = 99
        with self.assertRaisesRegex(CheckpointValidationError, "config fingerprint"):
            RunManifest.from_dict(tampered)

    def test_manifest_rejects_path_like_run_id_and_duplicate_expected_ids(self):
        with self.assertRaises(ValueError):
            RunManifest.create(
                run_id="../escape",
                pipeline="x",
                config={},
                data_hashes={},
                prompt_hashes={},
                sampling_plan={},
                expected_sample_ids={},
            )
        with self.assertRaises(ValueError):
            RunManifest.create(
                run_id="safe",
                pipeline="x",
                config={},
                data_hashes={},
                prompt_hashes={},
                sampling_plan={},
                expected_sample_ids={"step1": ["a", "a"]},
            )
        for run_id in (".", ".."):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    RunManifest.create(
                        run_id=run_id,
                        pipeline="x",
                        config={},
                        data_hashes={},
                        prompt_hashes={},
                        sampling_plan={},
                        expected_sample_ids={},
                    )

    def test_nested_manifest_mutation_is_detected_before_persistence(self):
        value = RunManifest.create(
            run_id="mutable",
            pipeline="x",
            config={"nested": {"seed": 1}},
            data_hashes={},
            prompt_hashes={},
            sampling_plan={},
            expected_sample_ids={},
            created_at="2026-01-01T00:00:00+00:00",
        )
        value.config["nested"]["seed"] = 2
        with self.assertRaisesRegex(CheckpointValidationError, "config fingerprint"):
            value.to_dict()

    def test_dependencies_form_full_compile_dag(self):
        self.assertEqual(dependencies_for("step1"), ())
        self.assertEqual(dependencies_for("step3"), ("step1", "step2"))
        self.assertEqual(
            dependencies_for("step4b"),
            ("step1", "step2", "step3", "step4a"),
        )


class AtomicWriteTests(unittest.TestCase):
    def test_failed_serialization_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "value.json"
            atomic_write_json(path, {"old": True})
            with self.assertRaises(TypeError):
                atomic_write_json(path, {"not-json": {1, 2}})
            self.assertEqual(json.loads(path.read_text()), {"old": True})
            self.assertEqual(list(path.parent.glob(".value.json.*.tmp")), [])


class CheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = CheckpointStore(self.tempdir.name)
        self.manifest = manifest()
        self.store.save_manifest(self.manifest)

    def test_manifest_and_chunks_round_trip_with_missing_detection(self):
        self.assertEqual(self.store.load_manifest("run-001"), self.manifest)
        self.store.write_chunk(
            self.manifest, "step1", 0, [record("s1-a", "step1", "Yes")]
        )
        self.assertEqual(
            self.store.missing_sample_ids(self.manifest, "step1"), ("s1-b",)
        )
        with self.assertRaisesRegex(CheckpointValidationError, "missing sample ID"):
            self.store.load_stage(self.manifest, "step1")

        self.store.write_chunk(
            self.manifest, "step1", 1, [record("s1-b", "step1", "No")]
        )
        loaded = self.store.load_stage(self.manifest, "step1")
        self.assertEqual(list(loaded), ["s1-a", "s1-b"])
        self.assertEqual(loaded["s1-a"].value, "Yes")

        payload = json.loads(
            self.store._chunk_path(self.manifest, "step1", 0).read_text()
        )
        self.assertEqual(
            payload["records_sha256"],
            sha256_text(canonical_json(payload["records"])),
        )

    def test_write_rejects_pipeline_and_status_semantic_mismatches(self):
        with self.assertRaisesRegex(CheckpointValidationError, "must be Yes or No"):
            self.store.write_chunk(
                self.manifest,
                "step1",
                0,
                [record("s1-a", "step1", {"probabilities": {"Yes": 2, "No": 1}})],
            )
        with self.assertRaisesRegex(CheckpointValidationError, "not boolean"):
            self.store.write_chunk(
                self.manifest, "step2", 0, [record("s2", "step2", True)]
            )
        with self.assertRaisesRegex(CheckpointValidationError, "requires error_code"):
            self.store.write_chunk(
                self.manifest,
                "step1",
                0,
                [
                    ExperimentRecord(
                        "s1-a",
                        "step1",
                        {"sample": "s1-a"},
                        ResultStatus.INVALID,
                    )
                ],
            )

    def test_logprob_semantics_accept_random_sample_but_reject_malformed_score(self):
        logprob_manifest = manifest(pipeline="logprob")
        logprob_store = CheckpointStore(Path(self.tempdir.name) / "logprob")
        logprob_store.save_manifest(logprob_manifest)
        sampled_no = ExperimentRecord(
            "s1-a",
            "step1",
            {"sample": "s1-a"},
            ResultStatus.VALID,
            logprob_value(sampled_choice="No"),
        )
        logprob_store.write_chunk(logprob_manifest, "step1", 0, [sampled_no])
        self.assertEqual(
            logprob_store.load_stage(logprob_manifest, "step1", require_complete=False)[
                "s1-a"
            ].value["sampled_choice"],
            "No",
        )

        malformed = logprob_value()
        malformed["probabilities"] = {"Yes": 2.0, "No": 1.0}
        with self.assertRaisesRegex(
            CheckpointValidationError, "probability must be at most"
        ):
            logprob_store.write_chunk(
                logprob_manifest,
                "step1",
                1,
                [
                    ExperimentRecord(
                        "s1-b",
                        "step1",
                        {"sample": "s1-b"},
                        ResultStatus.VALID,
                        malformed,
                    )
                ],
            )

    def test_logprob_numeric_tolerance_accepts_clamped_boundary_mass(self):
        logprob_manifest = manifest(pipeline="logprob")
        logprob_store = CheckpointStore(Path(self.tempdir.name) / "boundary-logprob")
        logprob_store.save_manifest(logprob_manifest)
        overshoot = LOGPROB_NUMERIC_TOLERANCE / 2
        raw_yes = 0.75
        raw_no = 0.25 + overshoot
        raw_mass = raw_yes + raw_no
        value = logprob_value()
        value["probabilities"] = {
            "Yes": raw_yes / raw_mass,
            "No": raw_no / raw_mass,
        }
        value["label_logprobs"] = {
            "Yes": math.log(raw_yes),
            "No": math.log(raw_no),
        }
        value["candidate_mass"] = 1.0
        value["residual_mass"] = 0.0
        value["candidates"][0].update(
            logprob=math.log(raw_yes), probability=raw_yes
        )
        value["candidates"][1].update(logprob=math.log(raw_no), probability=raw_no)

        logprob_store.write_chunk(
            logprob_manifest,
            "step1",
            0,
            [
                ExperimentRecord(
                    "s1-a",
                    "step1",
                    {"sample": "s1-a"},
                    ResultStatus.VALID,
                    value,
                )
            ],
        )
        loaded = logprob_store.load_stage(
            logprob_manifest, "step1", require_complete=False
        )
        self.assertEqual(loaded["s1-a"].value["candidate_mass"], 1.0)

    def test_logprob_numeric_tolerance_rejects_larger_mass_mismatch(self):
        logprob_manifest = manifest(pipeline="logprob")
        logprob_store = CheckpointStore(Path(self.tempdir.name) / "large-overshoot")
        logprob_store.save_manifest(logprob_manifest)
        overshoot = LOGPROB_NUMERIC_TOLERANCE * 2
        raw_yes = 0.75
        raw_no = 0.25 + overshoot
        raw_mass = raw_yes + raw_no
        value = logprob_value()
        value["probabilities"] = {
            "Yes": raw_yes / raw_mass,
            "No": raw_no / raw_mass,
        }
        value["label_logprobs"] = {
            "Yes": math.log(raw_yes),
            "No": math.log(raw_no),
        }
        value["candidate_mass"] = 1.0
        value["residual_mass"] = 0.0
        value["candidates"][0].update(
            logprob=math.log(raw_yes), probability=raw_yes
        )
        value["candidates"][1].update(logprob=math.log(raw_no), probability=raw_no)

        with self.assertRaisesRegex(
            CheckpointValidationError, "candidate_mass does not match candidates"
        ):
            logprob_store.write_chunk(
                logprob_manifest,
                "step1",
                0,
                [
                    ExperimentRecord(
                        "s1-a",
                        "step1",
                        {"sample": "s1-a"},
                        ResultStatus.VALID,
                        value,
                    )
                ],
            )

    def test_sparse_invalid_logprob_diagnostics_round_trip(self):
        logprob_manifest = manifest(pipeline="logprob")
        logprob_store = CheckpointStore(Path(self.tempdir.name) / "sparse-logprob")
        logprob_store.save_manifest(logprob_manifest)
        sparse_diagnostics = {
            "probabilities": None,
            "label_logprobs": None,
            "candidate_mass": None,
            "residual_mass": None,
            "candidates": None,
            "sampled_token_id": None,
            "sampled_choice": None,
            "format_valid": None,
            "analysis_text": "visible analysis",
            "analysis_text_kind": "model_generated_visible_text",
            "estimator": "bounded_single_token_candidate_set",
            "conditional_on_candidate_set": False,
            "scoring_temperature": 0.0,
            "finish_reason": None,
        }
        invalid = ExperimentRecord(
            "s1-a",
            "step1",
            {"sample": "s1-a"},
            ResultStatus.INVALID,
            sparse_diagnostics,
            "first_token_logprobs_unavailable",
            "bounded Yes/No candidate score is invalid",
        )

        logprob_store.write_chunk(logprob_manifest, "step1", 0, [invalid])
        loaded = logprob_store.load_stage(
            logprob_manifest, "step1", require_complete=False
        )
        self.assertIsNone(loaded["s1-a"].value["format_valid"])
        self.assertIsNone(loaded["s1-a"].value["sampled_choice"])

    def test_invalid_logprob_step2_requires_null_value(self):
        logprob_manifest = manifest(pipeline="logprob")
        logprob_store = CheckpointStore(Path(self.tempdir.name) / "step2-logprob")
        logprob_store.save_manifest(logprob_manifest)
        invalid = ExperimentRecord(
            "s2",
            "step2",
            {"sample": "s2"},
            ResultStatus.INVALID,
            50.0,
            "invalid_schema",
            "invalid percentage response",
        )

        with self.assertRaisesRegex(
            CheckpointValidationError, "INVALID Step 2.*null value"
        ):
            logprob_store.write_chunk(logprob_manifest, "step2", 0, [invalid])

    def test_load_rejects_semantic_tamper_even_with_recomputed_records_digest(self):
        path = self.store.write_chunk(
            self.manifest, "step1", 0, [record("s1-a", "step1", "Yes")]
        )
        payload = json.loads(path.read_text())
        payload["records"][0]["value"] = {"probabilities": {"Yes": 2, "No": 1}}
        payload["records_sha256"] = sha256_text(canonical_json(payload["records"]))
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CheckpointValidationError, "must be Yes or No"):
            self.store.load_stage(self.manifest, "step1", require_complete=False)

    def test_load_rejects_record_content_tamper_and_exponent_overflow(self):
        path = self.store.write_chunk(
            self.manifest, "step1", 0, [record("s1-a", "step1", "Yes")]
        )
        original = path.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload["records"][0]["value"] = "No"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            CheckpointValidationError, "records digest mismatch"
        ):
            self.store.load_stage(self.manifest, "step1", require_complete=False)

        path.write_text(
            original.replace('"value": "Yes"', '"value": 1e309'),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CheckpointValidationError, "non-finite JSON number"
        ):
            self.store.load_stage(self.manifest, "step1", require_complete=False)

    def test_stage_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            stage_path = self.store.run_directory(self.manifest.run_id) / "step1"
            try:
                stage_path.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")
            with self.assertRaisesRegex(
                CheckpointValidationError, "must not be a symbolic link"
            ):
                self.store.stage_directory(self.manifest, "step1")

    def test_chunk_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            outside_chunk = Path(outside) / "chunk.json"
            outside_chunk.write_text("{}", encoding="utf-8")
            stage_path = self.store.stage_directory(self.manifest, "step1")
            stage_path.mkdir(parents=True)
            chunk_path = stage_path / "chunk_00000000.json"
            try:
                chunk_path.symlink_to(outside_chunk)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")
            with self.assertRaisesRegex(
                CheckpointValidationError, "must be a regular file within its stage"
            ):
                self.store.load_stage(self.manifest, "step1", require_complete=False)

    def test_run_manifest_and_lock_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as other_root:
            other_store = CheckpointStore(Path(other_root) / "checkpoints")
            run_link = other_store.root / self.manifest.run_id
            run_link.parent.mkdir(parents=True)
            target = Path(other_root) / "target-run"
            target.mkdir()
            try:
                run_link.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")
            with self.assertRaisesRegex(
                CheckpointValidationError, "run directory must not be a symbolic link"
            ):
                other_store.run_directory(self.manifest.run_id)

        manifest_path = self.store.manifest_path(self.manifest.run_id)
        original_manifest = manifest_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as outside:
            outside_manifest = Path(outside) / "manifest.json"
            outside_manifest.write_text(original_manifest, encoding="utf-8")
            manifest_path.unlink()
            manifest_path.symlink_to(outside_manifest)
            with self.assertRaisesRegex(
                CheckpointValidationError, "JSON must not be a symbolic link"
            ):
                self.store.load_manifest(self.manifest.run_id)

        manifest_path.unlink()
        manifest_path.write_text(original_manifest, encoding="utf-8")
        lock_path = self.store._run_lock_path(self.manifest.run_id)
        lock_path.unlink()
        with tempfile.TemporaryDirectory() as outside:
            outside_lock = Path(outside) / "lock"
            outside_lock.write_text("sentinel", encoding="utf-8")
            lock_path.symlink_to(outside_lock)
            with self.assertRaisesRegex(
                CheckpointValidationError, "lock must not be a symbolic link"
            ):
                self.store.save_manifest(self.manifest)
            self.assertEqual(outside_lock.read_text(encoding="utf-8"), "sentinel")

    def test_same_chunk_is_idempotent_but_conflict_or_cross_chunk_duplicate_fails(self):
        first = record("s1-a", "step1", "Yes")
        path = self.store.write_chunk(self.manifest, "step1", 0, [first])
        self.assertEqual(
            self.store.write_chunk(self.manifest, "step1", 0, [first]), path
        )
        with self.assertRaises(FileExistsError):
            self.store.write_chunk(
                self.manifest,
                "step1",
                0,
                [record("s1-a", "step1", "No")],
            )
        with self.assertRaisesRegex(
            CheckpointValidationError, "another checkpoint chunk"
        ):
            self.store.write_chunk(self.manifest, "step1", 1, [first])

    def test_incremental_index_avoids_full_stage_reload_and_is_rebuildable(self):
        self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1", "Yes")],
        )
        index_path = (
            self.store.stage_directory(self.manifest, "step1") / ".sample-index.sqlite3"
        )
        self.assertTrue(index_path.is_file())

        with sqlite3.connect(index_path) as connection:
            connection.execute("DELETE FROM samples")

        original_load_stage = self.store.load_stage

        def forbidden_full_reload(*args, **kwargs):
            raise AssertionError("write_chunk must not reload every historical record")

        self.store.load_stage = forbidden_full_reload
        try:
            self.store.write_chunk(
                self.manifest,
                "step1",
                1,
                [record("s1-b", "step1", "No")],
            )
        finally:
            self.store.load_stage = original_load_stage

        index_path.unlink()
        with self.assertRaisesRegex(
            CheckpointValidationError, "another checkpoint chunk"
        ):
            self.store.write_chunk(
                self.manifest,
                "step1",
                2,
                [record("s1-a", "step1", "Yes")],
            )

    def test_warm_store_does_not_reload_historical_shards_on_each_append(self):
        expected_ids = tuple(f"sample-{index:02d}" for index in range(10))
        incremental_manifest = manifest({"step1": expected_ids})
        root = Path(self.tempdir.name) / "incremental"
        store = CheckpointStore(root)
        store.save_manifest(incremental_manifest)

        for index, sample_id in enumerate(expected_ids[:4]):
            store.write_chunk(
                incremental_manifest,
                "step1",
                index,
                [record(sample_id, "step1")],
            )

        original_load_chunk = store._load_chunk
        warm_loads = []

        def count_warm_loads(manifest_value, stage, path):
            warm_loads.append(path.name)
            return original_load_chunk(manifest_value, stage, path)

        store._load_chunk = count_warm_loads
        try:
            for index, sample_id in enumerate(expected_ids[4:8], start=4):
                store.write_chunk(
                    incremental_manifest,
                    "step1",
                    index,
                    [record(sample_id, "step1")],
                )
        finally:
            store._load_chunk = original_load_chunk
        self.assertEqual(warm_loads, [])

        # A new store has no store-local authority cache. It must validate all
        # existing JSON shards once, then its next append is warm and incremental.
        restarted = CheckpointStore(root)
        restarted_load_chunk = restarted._load_chunk
        cold_loads = []

        def count_cold_loads(manifest_value, stage, path):
            cold_loads.append(path.name)
            return restarted_load_chunk(manifest_value, stage, path)

        restarted._load_chunk = count_cold_loads
        try:
            restarted.write_chunk(
                incremental_manifest,
                "step1",
                8,
                [record(expected_ids[8], "step1")],
            )
            self.assertEqual(
                cold_loads,
                [f"chunk_{index:08d}.json" for index in range(8)],
            )
            restarted.write_chunk(
                incremental_manifest,
                "step1",
                9,
                [record(expected_ids[9], "step1")],
            )
        finally:
            restarted._load_chunk = restarted_load_chunk
        self.assertEqual(len(cold_loads), 8)

    def test_external_shard_change_invalidates_warm_authority_cache(self):
        chunk_path = self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1", "Yes")],
        )
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        payload["records"][0]["value"] = "No"
        chunk_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(
            CheckpointValidationError, "records digest mismatch"
        ):
            self.store.write_chunk(
                self.manifest,
                "step1",
                1,
                [record("s1-b", "step1", "No")],
            )
        self.assertFalse(self.store._chunk_path(self.manifest, "step1", 1).exists())

    def test_other_store_append_forces_fresh_json_authority_scan(self):
        expected_ids = ("sample-a", "sample-b", "sample-c")
        shared_manifest = manifest({"step1": expected_ids})
        root = Path(self.tempdir.name) / "shared"
        first_store = CheckpointStore(root)
        second_store = CheckpointStore(root)
        first_store.save_manifest(shared_manifest)
        first_store.write_chunk(
            shared_manifest, "step1", 0, [record("sample-a", "step1")]
        )
        second_store.write_chunk(
            shared_manifest, "step1", 1, [record("sample-b", "step1")]
        )

        original_load_chunk = first_store._load_chunk
        reloads = []

        def count_reloads(manifest_value, stage, path):
            reloads.append(path.name)
            return original_load_chunk(manifest_value, stage, path)

        first_store._load_chunk = count_reloads
        try:
            first_store.write_chunk(
                shared_manifest, "step1", 2, [record("sample-c", "step1")]
            )
        finally:
            first_store._load_chunk = original_load_chunk
        self.assertEqual(
            reloads,
            ["chunk_00000000.json", "chunk_00000001.json"],
        )
        self.assertEqual(
            list(first_store.load_stage(shared_manifest, "step1")),
            list(expected_ids),
        )

    def test_large_chunks_do_not_hit_sqlite_parameter_limit(self):
        expected_ids = tuple(f"sample-{index:04d}" for index in range(1_200))
        large_manifest = manifest({"step1": expected_ids})
        large_store = CheckpointStore(Path(self.tempdir.name) / "large")
        large_store.save_manifest(large_manifest)
        large_store.write_chunk(
            large_manifest,
            "step1",
            0,
            [record(sample_id, "step1") for sample_id in expected_ids[:600]],
        )
        large_store.write_chunk(
            large_manifest,
            "step1",
            1,
            [record(sample_id, "step1") for sample_id in expected_ids[600:]],
        )
        self.assertEqual(
            len(large_store.load_stage(large_manifest, "step1")),
            len(expected_ids),
        )

    def test_logically_corrupt_index_metadata_is_rebuilt_from_json_shards(self):
        self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1", "Yes")],
        )
        index_path = (
            self.store.stage_directory(self.manifest, "step1") / ".sample-index.sqlite3"
        )
        with sqlite3.connect(index_path) as connection:
            connection.execute("UPDATE chunks SET sample_count = 'not-an-integer'")

        self.store.write_chunk(
            self.manifest,
            "step1",
            1,
            [record("s1-b", "step1", "No")],
        )

        loaded = self.store.load_stage(self.manifest, "step1")
        self.assertEqual(list(loaded), ["s1-a", "s1-b"])

    def test_logically_corrupt_sample_index_cannot_authorize_duplicate_shard(self):
        self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1", "Yes")],
        )
        index_path = (
            self.store.stage_directory(self.manifest, "step1") / ".sample-index.sqlite3"
        )
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                "UPDATE samples SET sample_id = ? WHERE sample_id = ?",
                ("ghost", "s1-a"),
            )

        with self.assertRaisesRegex(
            CheckpointValidationError, "another checkpoint chunk"
        ):
            self.store.write_chunk(
                self.manifest,
                "step1",
                1,
                [record("s1-a", "step1", "No")],
            )
        self.assertFalse(self.store._chunk_path(self.manifest, "step1", 1).exists())
        loaded = self.store.load_stage(self.manifest, "step1", require_complete=False)
        self.assertEqual(loaded["s1-a"].value, "Yes")
        with sqlite3.connect(index_path) as connection:
            indexed_ids = {
                row[0] for row in connection.execute("SELECT sample_id FROM samples")
            }
        self.assertEqual(indexed_ids, {"s1-a"})

    def test_unknown_duplicate_and_wrong_stage_records_fail(self):
        with self.assertRaisesRegex(CheckpointValidationError, "unknown sample ID"):
            self.store.write_chunk(
                self.manifest, "step1", 0, [record("unknown", "step1")]
            )
        with self.assertRaisesRegex(CheckpointValidationError, "duplicate"):
            self.store.write_chunk(
                self.manifest,
                "step1",
                0,
                [record("s1-a", "step1"), record("s1-a", "step1")],
            )
        with self.assertRaisesRegex(CheckpointValidationError, "has stage"):
            self.store.write_chunk(self.manifest, "step1", 0, [record("s1-a", "step2")])

    def test_corrupt_or_fingerprint_mismatched_shard_fails(self):
        path = self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1")],
        )
        payload = json.loads(path.read_text())
        payload["run_fingerprint"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CheckpointValidationError, "fingerprint mismatch"):
            self.store.load_stage(self.manifest, "step1", require_complete=False)

        path.write_text("{truncated", encoding="utf-8")
        with self.assertRaisesRegex(CheckpointValidationError, "cannot load"):
            self.store.load_stage(self.manifest, "step1", require_complete=False)

    def test_resume_requires_every_prior_stage_needed_for_final_compile(self):
        self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1"), record("s1-b", "step1")],
        )
        self.store.write_chunk(self.manifest, "step2", 0, [record("s2", "step2")])
        self.store.write_chunk(self.manifest, "step3", 0, [record("s3", "step3")])
        with self.assertRaisesRegex(CheckpointValidationError, "missing sample ID"):
            self.store.validate_resume_dependencies(self.manifest, "step4b")

        self.store.write_chunk(self.manifest, "step4a", 0, [record("s4a", "step4a")])
        loaded = self.store.validate_resume_dependencies(self.manifest, "step4b")
        self.assertEqual(set(loaded), {"step1", "step2", "step3", "step4a"})

    def test_chunk_write_checks_disk_manifest_and_manifest_cannot_change_after_shards(
        self,
    ):
        self.store.write_chunk(
            self.manifest,
            "step1",
            0,
            [record("s1-a", "step1")],
        )
        other = RunManifest.create(
            run_id="run-001",
            pipeline="verbalize",
            config={"model": "different"},
            data_hashes={"data": sha256_text("data")},
            prompt_hashes={"step1": sha256_text("prompt")},
            sampling_plan={"base_units": [{"id": "base"}]},
            expected_sample_ids=self.manifest.expected_sample_ids,
            code_version="deadbeef",
            created_at="2026-01-01T00:00:00+00:00",
        )
        with self.assertRaisesRegex(CheckpointValidationError, "stored on disk"):
            self.store.write_chunk(other, "step1", 1, [record("s1-b", "step1")])
        with self.assertRaisesRegex(CheckpointValidationError, "cannot overwrite"):
            self.store.save_manifest(other, overwrite=True)

    def test_concurrent_writers_cannot_overwrite_same_chunk(self):
        def write(value):
            return self.store.write_chunk(
                self.manifest,
                "step1",
                0,
                [record(value, "step1")],
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, value) for value in ("s1-a", "s1-b")]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError:
                outcomes.append("conflict")
        self.assertEqual(outcomes.count("conflict"), 1)
        loaded = self.store.load_stage(self.manifest, "step1", require_complete=False)
        self.assertEqual(len(loaded), 1)

    def test_checkpoint_loader_rejects_duplicate_keys_and_nonfinite_numbers(self):
        manifest_path = self.store.manifest_path("run-001")
        manifest_path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        with self.assertRaisesRegex(CheckpointValidationError, "duplicate_json_key"):
            self.store.load_manifest("run-001")
        manifest_path.write_text('{"value": NaN}', encoding="utf-8")
        with self.assertRaisesRegex(CheckpointValidationError, "invalid_json"):
            self.store.load_manifest("run-001")

    def test_checkpoint_loader_rejects_fenced_json(self):
        manifest_path = self.store.manifest_path("run-001")
        manifest_path.write_text(
            '```json\n{"run_id": "run-001"}\n```',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CheckpointValidationError, "invalid_json"):
            self.store.load_manifest("run-001")

    def test_standalone_id_validation_detects_every_mismatch_class(self):
        self.assertEqual(
            validate_record_ids(["a", "b"], ["a"], require_complete=False), ("b",)
        )
        with self.assertRaises(CheckpointValidationError):
            validate_record_ids(["a"], ["a", "a"])
        with self.assertRaises(CheckpointValidationError):
            validate_record_ids(["a"], ["other"])
        with self.assertRaises(CheckpointValidationError):
            validate_record_ids(["a", "b"], ["a"])


if __name__ == "__main__":
    unittest.main()
