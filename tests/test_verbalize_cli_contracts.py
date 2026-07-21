import pytest

from src.experiment.planning import TreatmentAssignment, TreatmentCondition
from src.experiment.verbalize_experiment_runner import (
    VerbalizeExperimentRunner,
    main,
)


def test_treatment_prompt_uses_fixed_decimal_percentage_contract():
    runner = VerbalizeExperimentRunner(
        "fake/model",
        llm_interface=object(),
        show_progress=False,
    )
    runner.load_prompt_templates()
    assignment = TreatmentAssignment(
        stage="step4a",
        unit_metadata={
            "persona": "none",
            "category": "economy",
            "proposal": "A policy",
        },
        condition=TreatmentCondition(
            kind="fixed_hypothetical_survey",
            source="hypothetical_survey",
            percentage=0.000001,
        ),
        replicate_id=0,
        seed=123,
        order_index=0,
        step2_predicted_percentage=50.0,
        survey_surprise=-49.999999,
        sample_id="step4a:fixed-tiny",
    )

    prompt = runner._build_prompt("step4a", assignment)

    assert prompt is not None
    assert "0.000001%" in prompt
    assert "e-" not in prompt.casefold()


def test_dry_run_and_resume_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--dry-run", "--resume-from", "existing-run"])

    assert exc_info.value.code == 2
    assert (
        "--dry-run and --resume-from are mutually exclusive" in capsys.readouterr().err
    )


def test_default_results_directory_is_relative_to_runtime_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    runner = VerbalizeExperimentRunner(
        "fake/model",
        llm_interface=object(),
        show_progress=False,
    )

    assert runner.results_dir == tmp_path / "results"
