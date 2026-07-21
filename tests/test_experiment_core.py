import json
import tempfile
import unittest
from pathlib import Path

from src.experiment.core import (
    ExperimentRecord,
    ResultStatus,
    ValidationResult,
    hash_files,
    hash_templates,
    make_record,
    parse_json_object,
    parse_percentage_response,
    parse_yes_no_response,
    sha256_text,
    stable_sample_id,
    summarize_binary_results,
)


class StrictParsingTests(unittest.TestCase):
    def test_direct_and_fenced_json_objects_are_accepted(self):
        direct = parse_json_object('{"thinking": "ok", "answer": "Yes"}')
        fenced = parse_json_object('```json\n{"answer": 62.5}\n```')

        self.assertTrue(direct.is_valid)
        self.assertEqual(direct.value["answer"], "Yes")
        self.assertTrue(fenced.is_valid)
        self.assertEqual(fenced.value["answer"], 62.5)

    def test_non_object_ambiguous_or_nonstandard_json_is_rejected(self):
        cases = (
            "[1]",
            '"Yes"',
            'prefix {"answer": "Yes"}',
            '{"answer": "Yes", "answer": "No"}',
            '{"answer": NaN}',
            "",
        )
        for response in cases:
            with self.subTest(response=response):
                self.assertEqual(
                    parse_json_object(response).status, ResultStatus.INVALID
                )

    def test_yes_no_parser_accepts_only_enum_answer(self):
        self.assertEqual(parse_yes_no_response('{"answer":" yes "}').value, "Yes")
        self.assertEqual(parse_yes_no_response('{"answer":"NO"}').value, "No")

        for response in (
            '{"answer":"Maybe"}',
            '{"answer":true}',
            '{"answer":1}',
            '{"thinking":"Yes"}',
            "Yes",
        ):
            with self.subTest(response=response):
                self.assertEqual(
                    parse_yes_no_response(response).status, ResultStatus.INVALID
                )

    def test_invalid_and_error_results_do_not_bias_binary_ratio(self):
        summary = summarize_binary_results(
            [
                parse_yes_no_response('{"answer":"Yes"}'),
                parse_yes_no_response('{"answer":"No"}'),
                parse_yes_no_response('{"answer":"Maybe"}'),
                ValidationResult(status=ResultStatus.ERROR, error_code="transport"),
            ]
        )
        self.assertEqual(summary.yes, 1)
        self.assertEqual(summary.no, 1)
        self.assertEqual(summary.invalid, 1)
        self.assertEqual(summary.error, 1)
        self.assertEqual(summary.yes_ratio, 0.5)
        only_failures = summarize_binary_results(
            [ValidationResult(status=ResultStatus.INVALID)]
        )
        self.assertIsNone(only_failures.yes_ratio)

    def test_percentage_parser_preserves_decimal_and_never_guesses_or_clamps(self):
        accepted = {
            '{"answer":62.5}': 62.5,
            '{"answer":"62.5%"}': 62.5,
            '{"answer":"47 percent"}': 47.0,
            '{"thinking":"cost 1200", "answer":"47%"}': 47.0,
            '```json\n{"answer": 100}\n```': 100.0,
        }
        for response, expected in accepted.items():
            with self.subTest(response=response):
                result = parse_percentage_response(response)
                self.assertTrue(result.is_valid)
                self.assertEqual(result.value, expected)

        rejected = (
            "The policy costs 1200 dollars; estimated support is 47%",
            '{"thinking":"47%"}',
            '{"answer":"about 47%"}',
            '{"answer":-1}',
            '{"answer":101}',
            '{"answer":true}',
            '{"answer":"NaN"}',
        )
        for response in rejected:
            with self.subTest(response=response):
                self.assertEqual(
                    parse_percentage_response(response).status,
                    ResultStatus.INVALID,
                )
        huge_integer = '{"answer":' + ("9" * 400) + "}"
        self.assertEqual(
            parse_percentage_response(huge_integer).status,
            ResultStatus.INVALID,
        )


class IdentityAndFingerprintTests(unittest.TestCase):
    def test_sample_id_is_order_independent_full_sha256(self):
        first = stable_sample_id("step1", {"persona": "a", "proposal": "p"})
        second = stable_sample_id("step1", {"proposal": "p", "persona": "a"})
        other_stage = stable_sample_id("step2", {"proposal": "p", "persona": "a"})

        self.assertEqual(first, second)
        self.assertNotEqual(first, other_stage)
        self.assertEqual(len(first.split(":", 1)[1]), 64)

    def test_template_and_file_hashes_are_content_based(self):
        templates = hash_templates({"b": "second", "a": "first"})
        self.assertEqual(list(templates), ["a", "b"])
        self.assertEqual(templates["a"], sha256_text("first"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            hashes = hash_files({"logical-data": path})
            self.assertEqual(hashes["logical-data"], sha256_text(path.read_text()))

    def test_record_round_trip_retains_explicit_invalid_status(self):
        validation = ValidationResult(
            status=ResultStatus.INVALID,
            error_code="bad_answer",
            message="not Yes or No",
            raw_response="Maybe",
        )
        record = make_record("step1", {"persona": "a"}, validation)
        restored = ExperimentRecord.from_dict(json.loads(json.dumps(record.to_dict())))

        self.assertEqual(restored, record)
        self.assertEqual(restored.status, ResultStatus.INVALID)
        self.assertIsNone(restored.value)


if __name__ == "__main__":
    unittest.main()
