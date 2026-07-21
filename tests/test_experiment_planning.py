import unittest

from src.experiment.planning import (
    ActionUnit,
    ProposalUnit,
    TreatmentCondition,
    build_baseline_assignments,
    build_selection_plan,
    build_treatment_plan,
    compute_survey_surprise,
    format_presented_percentage,
    leave_one_persona_out_consensus,
)


def make_units():
    units = []
    for persona in ("persona-b", "persona-a"):
        for action_type, action in (
            ("public", "speak publicly"),
            ("personal", "change behavior"),
            ("strategic", "donate"),
        ):
            units.append(
                ActionUnit(
                    persona=persona,
                    category="economy",
                    proposal="A policy",
                    action_type=action_type,
                    action=action,
                )
            )
    return units


class SelectionPlanningTests(unittest.TestCase):
    def test_budget_is_applied_before_proposal_units_are_derived(self):
        plan = build_selection_plan(reversed(make_units()), max_base_units=1)
        self.assertEqual(len(plan.base_units), 1)
        self.assertEqual(len(plan.proposal_units), 1)
        self.assertEqual(
            plan.expected_baseline_counts(), {"step1": 1, "step2": 1, "step3": 1}
        )

    def test_selection_is_stable_and_invalid_budgets_or_duplicates_fail(self):
        units = make_units()
        first = build_selection_plan(units, max_base_units=4)
        second = build_selection_plan(reversed(units), max_base_units=4)
        self.assertEqual(first.to_dict(), second.to_dict())
        other_seed = build_selection_plan(units, max_base_units=4, selection_seed=99)
        self.assertNotEqual(
            [unit.unit_id for unit in first.base_units],
            [unit.unit_id for unit in other_seed.base_units],
        )

        for value in (0, -1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_selection_plan(units, max_base_units=value)
        with self.assertRaises(ValueError):
            build_selection_plan(units + [units[0]])

    def test_baseline_replicates_have_unique_stable_ids_and_seeds(self):
        selection = build_selection_plan(make_units())
        first = build_baseline_assignments(selection, replicates=2, master_seed=7)
        second = build_baseline_assignments(selection, replicates=2, master_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["step1"]), 4)
        self.assertEqual(len(first["step3"]), 12)
        all_assignments = first["step1"] + first["step2"] + first["step3"]
        self.assertEqual(
            len({item.sample_id for item in all_assignments}), len(all_assignments)
        )
        self.assertEqual(len({item.seed for item in all_assignments}), 6)
        for stage in ("step1", "step2", "step3"):
            for replicate_id in (0, 1):
                seeds = {
                    item.seed
                    for item in first[stage]
                    if item.replicate_id == replicate_id
                }
                self.assertEqual(len(seeds), 1)


class TreatmentPlanningTests(unittest.TestCase):
    def setUp(self):
        self.selection = build_selection_plan(make_units())
        self.beliefs = {
            ("persona-a", "economy", "A policy"): 1,
            ("persona-b", "economy", "A policy"): 0,
        }
        self.predictions = {
            ("persona-a", "economy", "A policy"): 25.0,
            ("persona-b", "economy", "A policy"): 75.0,
        }

    def test_leave_one_out_excludes_target_persona(self):
        pct_a, n_a = leave_one_persona_out_consensus(
            self.beliefs,
            ProposalUnit("persona-a", "economy", "A policy"),
        )
        pct_b, n_b = leave_one_persona_out_consensus(
            self.beliefs,
            ProposalUnit("persona-b", "economy", "A policy"),
        )
        self.assertEqual((pct_a, n_a), (0.0, 1))
        self.assertEqual((pct_b, n_b), (100.0, 1))

    def test_leave_one_out_weights_personas_not_valid_replicate_count(self):
        beliefs = {
            ("target", "economy", "A policy"): [1, 0],
            ("many-valid", "economy", "A policy"): [1, 1, None],
            ("one-valid", "economy", "A policy"): [0, None, None],
        }
        percentage, contributing_personas = leave_one_persona_out_consensus(
            beliefs,
            ProposalUnit("target", "economy", "A policy"),
        )
        self.assertEqual(percentage, 50.0)
        self.assertEqual(contributing_personas, 2)

    def test_leave_one_out_consensus_uses_presented_percentage_precision(self):
        beliefs = {
            ("target", "economy", "A policy"): 1,
            ("peer-a", "economy", "A policy"): 1,
            ("peer-b", "economy", "A policy"): 0,
            ("peer-c", "economy", "A policy"): 0,
        }

        percentage, contributing_personas = leave_one_persona_out_consensus(
            beliefs,
            ProposalUnit("target", "economy", "A policy"),
        )

        self.assertEqual(percentage, 33.333333)
        self.assertEqual(contributing_personas, 3)

    def test_step4a_is_proposal_level_and_step4b_remains_action_level(self):
        plan = build_treatment_plan(
            self.selection,
            valid_binary_beliefs=self.beliefs,
            step2_predictions=self.predictions,
            fixed_percentages=(10, 90),
            replicates=2,
            master_seed=11,
        )
        # 2 fixed + LOO consensus + retest + placebo = 5 conditions.
        self.assertEqual(len(plan.step4a), 2 * 5 * 2)
        self.assertEqual(len(plan.step4b), 6 * 5 * 2)
        self.assertTrue(all("action" not in item.unit_metadata for item in plan.step4a))
        self.assertTrue(all("action" in item.unit_metadata for item in plan.step4b))
        self.assertEqual(
            len({item.sample_id for item in plan.step4a}), len(plan.step4a)
        )

        persona_a = [
            item
            for item in plan.step4a
            if item.unit_metadata["persona"] == "persona-a" and item.replicate_id == 0
        ]
        simulated = next(
            item
            for item in persona_a
            if item.condition.kind == "simulated_persona_consensus"
        )
        fixed_ten = next(
            item
            for item in persona_a
            if (
                item.condition.kind == "fixed_hypothetical_survey"
                and item.condition.percentage == 10.0
            )
        )
        self.assertEqual(simulated.condition.percentage, 0.0)
        self.assertEqual(simulated.condition.excluded_persona, "persona-a")
        self.assertEqual(simulated.survey_surprise, -25.0)
        self.assertEqual(fixed_ten.survey_surprise, -15.0)

    def test_treatment_order_is_randomized_but_reproducible(self):
        kwargs = dict(
            valid_binary_beliefs=self.beliefs,
            step2_predictions=self.predictions,
            fixed_percentages=(10, 30, 50, 70, 90),
            replicates=2,
        )
        first = build_treatment_plan(self.selection, master_seed=123, **kwargs)
        second = build_treatment_plan(self.selection, master_seed=123, **kwargs)
        other_seed = build_treatment_plan(self.selection, master_seed=124, **kwargs)

        self.assertEqual(first, second)
        self.assertNotEqual(
            [
                item.condition.kind + str(item.condition.percentage)
                for item in first.step4a
            ],
            [
                item.condition.kind + str(item.condition.percentage)
                for item in other_seed.step4a
            ],
        )
        for persona in ("persona-a", "persona-b"):
            for replicate in (0, 1):
                assignments = [
                    item
                    for item in first.step4a
                    if (
                        item.unit_metadata["persona"] == persona
                        and item.replicate_id == replicate
                    )
                ]
                self.assertEqual(
                    sorted(item.order_index for item in assignments),
                    list(range(len(assignments))),
                )

    def test_equal_fixed_and_consensus_percentages_remain_distinct_conditions(self):
        plan = build_treatment_plan(
            self.selection,
            valid_binary_beliefs=self.beliefs,
            step2_predictions=self.predictions,
            fixed_percentages=(100,),
            include_retest=False,
            include_placebo=False,
        )
        persona_b = [
            item for item in plan.step4a if item.unit_metadata["persona"] == "persona-b"
        ]
        self.assertEqual(len(persona_b), 2)
        self.assertEqual({item.condition.percentage for item in persona_b}, {100.0})
        self.assertEqual(
            {item.condition.kind for item in persona_b},
            {"fixed_hypothetical_survey", "simulated_persona_consensus"},
        )
        self.assertEqual(len({item.sample_id for item in persona_b}), 2)

    def test_missing_peers_is_explicit_and_fixed_conditions_still_exist(self):
        one = build_selection_plan(make_units(), max_base_units=1)
        plan = build_treatment_plan(
            one,
            valid_binary_beliefs={(one.proposal_units[0].key): 1},
            step2_predictions={},
            fixed_percentages=(50,),
        )
        self.assertEqual(
            {issue.code for issue in plan.issues},
            {"simulated_consensus_unavailable", "step2_prediction_unavailable"},
        )
        self.assertEqual(
            {item.condition.kind for item in plan.step4a},
            {
                "fixed_hypothetical_survey",
                "simulated_persona_consensus",
                "no_information_retest",
                "placebo_text",
            },
        )
        self.assertTrue(all(item.survey_surprise is None for item in plan.step4a))

    def test_unresolved_and_resolved_plans_keep_the_same_sample_ids(self):
        skeleton = build_treatment_plan(
            self.selection,
            valid_binary_beliefs=None,
            step2_predictions={},
            fixed_percentages=(10, 90),
            replicates=2,
            master_seed=77,
        )
        resolved = build_treatment_plan(
            self.selection,
            valid_binary_beliefs=self.beliefs,
            step2_predictions=self.predictions,
            fixed_percentages=(10, 90),
            replicates=2,
            master_seed=77,
        )
        self.assertEqual(
            [assignment.sample_id for assignment in skeleton.step4a],
            [assignment.sample_id for assignment in resolved.step4a],
        )
        self.assertEqual(
            [assignment.sample_id for assignment in skeleton.step4b],
            [assignment.sample_id for assignment in resolved.step4b],
        )
        unresolved_simulated = next(
            item
            for item in skeleton.step4a
            if item.condition.kind == "simulated_persona_consensus"
        )
        resolved_by_id = {item.sample_id: item for item in resolved.step4a}
        self.assertIsNone(unresolved_simulated.condition.percentage)
        self.assertIsNotNone(
            resolved_by_id[unresolved_simulated.sample_id].condition.percentage
        )

    def test_surprise_validation_is_strict(self):
        self.assertEqual(compute_survey_surprise(70, 25), 45.0)
        self.assertIsNone(compute_survey_surprise(None, 25))
        with self.assertRaises(ValueError):
            compute_survey_surprise(101, 25)
        bad = dict(self.beliefs)
        bad[("persona-b", "economy", "A policy")] = 0.5
        with self.assertRaises(ValueError):
            leave_one_persona_out_consensus(
                bad,
                ProposalUnit("persona-a", "economy", "A policy"),
            )

    def test_presented_percentages_share_storage_format_and_surprise_precision(self):
        condition = TreatmentCondition(
            kind="fixed_hypothetical_survey",
            source="hypothetical_survey",
            percentage=12.3456789,
        )

        self.assertEqual(condition.percentage, 12.345679)
        self.assertEqual(format_presented_percentage(condition.percentage), "12.345679")
        self.assertEqual(format_presented_percentage(12.3400004), "12.34")
        self.assertEqual(format_presented_percentage(0.000001), "0.000001")
        self.assertNotIn("e", format_presented_percentage(0.000001).casefold())
        self.assertEqual(
            compute_survey_surprise(70.1234567, 25.0000002),
            45.123457,
        )

    def test_fixed_percentages_must_be_unique_after_quantization(self):
        with self.assertRaisesRegex(ValueError, "unique after quantization"):
            build_treatment_plan(
                self.selection,
                valid_binary_beliefs=self.beliefs,
                step2_predictions=self.predictions,
                fixed_percentages=(10.0000001, 10.0000002),
            )


if __name__ == "__main__":
    unittest.main()
