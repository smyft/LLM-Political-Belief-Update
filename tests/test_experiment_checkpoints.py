import json
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
from src.experiment.core import ExperimentRecord, ResultStatus, sha256_text


def record(sample_id, stage, value=None):
    return ExperimentRecord(
        sample_id=sample_id,
        stage=stage,
        metadata={"sample": sample_id},
        status=ResultStatus.VALID,
        value=value,
    )


def manifest(expected=None):
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
        pipeline="verbalize",
        config={"model": "fake", "seed": 42},
        data_hashes={"data": sha256_text("data")},
        prompt_hashes={"step1": sha256_text("prompt")},
        sampling_plan={"base_units": [{"id": "base"}]},
        expected_sample_ids=expected_ids,
        code_version="deadbeef",
        created_at="2026-01-01T00:00:00+00:00",
    )


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

    def test_large_chunk_duplicate_query_is_batched_below_sqlite_limit(self):
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
