import unittest

from src.experiment.checkpoints import CheckpointValidationError
from src.experiment.compiler import compile_grouped_results, join_records_by_id
from src.experiment.core import ExperimentRecord, ResultStatus, stable_sample_id
from src.experiment.planning import ActionUnit


def make_record(stage, metadata, value):
    return ExperimentRecord(
        sample_id=stable_sample_id(stage, {**metadata, "value-key": str(value)}),
        stage=stage,
        metadata=metadata,
        status=ResultStatus.VALID,
        value=value,
    )


def fixture():
    bases = [
        ActionUnit("p", "cat", "proposal", "a1", "action one"),
        ActionUnit("p", "cat", "proposal", "a2", "action two"),
    ]
    proposal_meta = {"persona": "p", "category": "cat", "proposal": "proposal"}
    step1 = [make_record("step1", {**proposal_meta, "replicate_id": 0}, "Yes")]
    step2 = [make_record("step2", {**proposal_meta, "replicate_id": 0}, 35.0)]
    step3 = [
        make_record(
            "step3", {**base.as_metadata(), "replicate_id": 0}, f"s3-{base.action_type}"
        )
        for base in bases
    ]
    step4a = [
        make_record(
            "step4a",
            {
                **proposal_meta,
                "replicate_id": 0,
                "order_index": order,
                "treatment_id": treatment,
            },
            treatment,
        )
        for order, treatment in ((1, "high"), (0, "low"))
    ]
    step4b = [
        make_record(
            "step4b",
            {
                **base.as_metadata(),
                "replicate_id": 0,
                "order_index": 0,
                "treatment_id": "low",
            },
            f"s4b-{base.action_type}",
        )
        for base in bases
    ]
    return bases, step1, step2, step3, step4a, step4b


def expected_ids(step1, step2, step3, step4a, step4b):
    return {
        stage: [record.sample_id for record in records]
        for stage, records in {
            "step1": step1,
            "step2": step2,
            "step3": step3,
            "step4a": step4a,
            "step4b": step4b,
        }.items()
    }


class CountingIterable:
    def __init__(self, values):
        self.values = list(values)
        self.yield_count = 0

    def __iter__(self):
        for value in self.values:
            self.yield_count += 1
            yield value


class IdJoinTests(unittest.TestCase):
    def test_join_is_order_independent_and_expected_order_is_restored(self):
        _, step1, _, _, step4a, _ = fixture()
        records = step4a + step1
        expected = [record.sample_id for record in reversed(records)]
        joined = join_records_by_id(reversed(records), expected_ids=expected)
        self.assertEqual(list(joined), expected)

    def test_duplicate_unknown_missing_and_wrong_stage_fail(self):
        _, step1, _, _, _, _ = fixture()
        item = step1[0]
        with self.assertRaisesRegex(CheckpointValidationError, "duplicate"):
            join_records_by_id([item, item])
        with self.assertRaisesRegex(CheckpointValidationError, "unknown"):
            join_records_by_id([item], expected_ids=["different"])
        with self.assertRaisesRegex(CheckpointValidationError, "missing"):
            join_records_by_id([item], expected_ids=[item.sample_id, "missing"])
        with self.assertRaisesRegex(CheckpointValidationError, "expected 'step2'"):
            join_records_by_id([item], expected_stage="step2")


class LinearCompilerTests(unittest.TestCase):
    def test_complete_mode_requires_authoritative_expected_ids(self):
        bases, step1, step2, step3, step4a, step4b = fixture()
        with self.assertRaisesRegex(ValueError, "expected_ids_by_stage"):
            compile_grouped_results(
                bases,
                step1_records=step1,
                step2_records=step2,
                step3_records=step3,
                step4a_records=step4a,
                step4b_records=step4b,
            )

    def test_expected_id_map_must_cover_exactly_the_logical_stages(self):
        bases, step1, step2, step3, step4a, step4b = fixture()
        expected = expected_ids(step1, step2, step3, step4a, step4b)

        missing = dict(expected)
        missing.pop("step4b")
        with self.assertRaisesRegex(ValueError, "missing stage: step4b"):
            compile_grouped_results(
                bases,
                step1_records=step1,
                step2_records=step2,
                step3_records=step3,
                step4a_records=step4a,
                step4b_records=step4b,
                expected_ids_by_stage=missing,
            )

        unknown = {**expected, "step5": []}
        with self.assertRaisesRegex(ValueError, "unknown stage: step5"):
            compile_grouped_results(
                bases,
                step1_records=step1,
                step2_records=step2,
                step3_records=step3,
                step4a_records=step4a,
                step4b_records=step4b,
                expected_ids_by_stage=unknown,
            )

    def test_step4a_is_grouped_once_and_shared_by_id_across_action_rows(self):
        bases, step1, step2, step3, step4a, step4b = fixture()
        compiled = compile_grouped_results(
            reversed(bases),
            step1_records=reversed(step1),
            step2_records=reversed(step2),
            step3_records=reversed(step3),
            step4a_records=reversed(step4a),
            step4b_records=reversed(step4b),
            expected_ids_by_stage=expected_ids(step1, step2, step3, step4a, step4b),
        )
        self.assertEqual(len(compiled), 2)
        shared_ids = [
            record["sample_id"]
            for record in compiled[0]["step4a_first_order_with_treatment"]
        ]
        self.assertEqual(
            shared_ids,
            [
                record["sample_id"]
                for record in compiled[1]["step4a_first_order_with_treatment"]
            ],
        )
        self.assertEqual(
            [
                record["value"]
                for record in compiled[0]["step4a_first_order_with_treatment"]
            ],
            ["low", "high"],
        )
        step3_values = {
            row["action_type"]: row["step3_action_support_no_distribution"][0]["value"]
            for row in compiled
        }
        self.assertEqual(step3_values, {"a1": "s3-a1", "a2": "s3-a2"})

    def test_each_stage_iterable_is_consumed_exactly_once(self):
        bases, step1, step2, step3, step4a, step4b = fixture()
        counters = [
            CountingIterable(values) for values in (step1, step2, step3, step4a, step4b)
        ]
        compile_grouped_results(
            bases,
            step1_records=counters[0],
            step2_records=counters[1],
            step3_records=counters[2],
            step4a_records=counters[3],
            step4b_records=counters[4],
            expected_ids_by_stage=expected_ids(step1, step2, step3, step4a, step4b),
        )
        self.assertEqual(
            [counter.yield_count for counter in counters],
            [len(step1), len(step2), len(step3), len(step4a), len(step4b)],
        )

    def test_missing_or_unselected_unit_data_fails_instead_of_defaulting(self):
        bases, step1, step2, step3, step4a, step4b = fixture()
        with self.assertRaisesRegex(CheckpointValidationError, "missing sample ID"):
            compile_grouped_results(
                bases,
                step1_records=step1,
                step2_records=step2,
                step3_records=step3,
                step4a_records=step4a,
                step4b_records=step4b[:1],
                expected_ids_by_stage=expected_ids(step1, step2, step3, step4a, step4b),
            )

        unknown_meta = {
            "persona": "unknown",
            "category": "cat",
            "proposal": "proposal",
            "replicate_id": 0,
        }
        with self.assertRaisesRegex(CheckpointValidationError, "unknown sample ID"):
            compile_grouped_results(
                bases,
                step1_records=step1 + [make_record("step1", unknown_meta, "Yes")],
                step2_records=step2,
                step3_records=step3,
                step4a_records=step4a,
                step4b_records=step4b,
                expected_ids_by_stage=expected_ids(
                    step1,
                    step2,
                    step3,
                    step4a,
                    step4b,
                ),
            )

    def test_expected_id_join_detects_incomplete_stage_before_grouping(self):
        bases, step1, step2, step3, step4a, step4b = fixture()
        expected = {
            "step1": [step1[0].sample_id, "missing-id"],
            "step2": [step2[0].sample_id],
            "step3": [record.sample_id for record in step3],
            "step4a": [record.sample_id for record in step4a],
            "step4b": [record.sample_id for record in step4b],
        }
        with self.assertRaisesRegex(CheckpointValidationError, "missing sample ID"):
            compile_grouped_results(
                bases,
                step1_records=step1,
                step2_records=step2,
                step3_records=step3,
                step4a_records=step4a,
                step4b_records=step4b,
                expected_ids_by_stage=expected,
            )


if __name__ == "__main__":
    unittest.main()
