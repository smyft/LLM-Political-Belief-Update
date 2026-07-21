import json
from pathlib import Path

import pytest

from src.data.data_loader import DataLoader, instantiate_prompt, load_prompt_template


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def valid_data_dir(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "entities.json",
        {"politicians": ["Person A"], "platforms": ["Platform B"]},
    )
    _write_json(
        tmp_path / "proposal_actions.json",
        {
            "economy": [
                {
                    "political_proposal": "Proposal 1",
                    "actions": [
                        {
                            "action_type": "Personal Commitment",
                            "action_description": "Action 1",
                        },
                        {
                            "action_type": "Public Advocacy",
                            "action_description": "Action 2",
                        },
                        {
                            "action_type": "Strategic Support",
                            "action_description": "Action 3",
                        },
                    ],
                }
            ]
        },
    )
    return tmp_path


def test_loader_validates_and_indexes_data(valid_data_dir: Path) -> None:
    loader = DataLoader(str(valid_data_dir))

    assert loader.get_personas() == ["none", "Person A", "Platform B"]
    assert loader.get_unique_proposals() == [("economy", "Proposal 1")]
    assert loader.get_proposal_action_pairs() == [
        ("economy", "Proposal 1", "Personal Commitment", "Action 1"),
        ("economy", "Proposal 1", "Public Advocacy", "Action 2"),
        ("economy", "Proposal 1", "Strategic Support", "Action 3"),
    ]

    first = loader.get_actions_for_proposal("economy", "Proposal 1")
    first.append(("mutation", "must not leak into the cache"))
    assert loader.get_actions_for_proposal("economy", "Proposal 1") == [
        ("Personal Commitment", "Action 1"),
        ("Public Advocacy", "Action 2"),
        ("Strategic Support", "Action 3"),
    ]


@pytest.mark.parametrize(
    "entities",
    [
        {"politicians": [], "platforms": [], "unexpected": []},
        {"politicians": "not-a-list", "platforms": []},
        {"politicians": ["same"], "platforms": ["same"]},
        {"politicians": [""], "platforms": []},
    ],
)
def test_invalid_entities_fail_fast(tmp_path: Path, entities: object) -> None:
    _write_json(tmp_path / "entities.json", entities)

    with pytest.raises(ValueError):
        DataLoader(str(tmp_path)).load_entities()


@pytest.mark.parametrize(
    "proposals",
    [
        {},
        {"economy": []},
        {"economy": [{"political_proposal": "P"}]},
        {"economy": [{"political_proposal": "P", "actions": []}]},
        {
            "economy": [
                {
                    "political_proposal": "P",
                    "actions": [
                        {
                            "action_type": "Personal Commitment",
                            "action_description": "",
                        },
                        {
                            "action_type": "Public Advocacy",
                            "action_description": "B",
                        },
                        {
                            "action_type": "Strategic Support",
                            "action_description": "C",
                        },
                    ],
                }
            ]
        },
        {
            "economy": [
                {
                    "political_proposal": "P",
                    "actions": [
                        {"action_type": "one", "action_description": "A"},
                        {"action_type": "two", "action_description": "B"},
                        {"action_type": "three", "action_description": "C"},
                    ],
                }
            ]
        },
        {
            "economy": [
                {
                    "political_proposal": "P",
                    "actions": [
                        {
                            "action_type": "Personal Commitment",
                            "action_description": "A",
                        },
                        {
                            "action_type": "Personal Commitment",
                            "action_description": "A",
                        },
                        {
                            "action_type": "Strategic Support",
                            "action_description": "B",
                        },
                    ],
                }
            ]
        },
    ],
)
def test_invalid_proposals_fail_fast(tmp_path: Path, proposals: object) -> None:
    _write_json(tmp_path / "proposal_actions.json", proposals)

    with pytest.raises(ValueError):
        DataLoader(str(tmp_path)).load_proposals()


def test_distribution_is_explicit_and_validated(valid_data_dir: Path) -> None:
    loader = DataLoader(str(valid_data_dir))

    assert loader.generate_distribution(0.7) == (
        "In this hypothetical survey vignette, 70% of respondents are described "
        "as believing this policy will be beneficial to the United States."
    )
    assert loader.generate_distribution(0.7, "negative") == (
        "In this hypothetical survey vignette, 30% of respondents are described "
        "as believing this policy will be harmful to the United States."
    )
    with pytest.raises(ValueError):
        loader.generate_distribution(1.1)
    with pytest.raises(ValueError):
        loader.generate_distribution(0.5, "ambiguous")
    with pytest.raises(TypeError):
        loader.generate_distribution(True)


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"politicians": [], "politicians": [], "platforms": []}',
        '{"politicians": [], "platforms": [NaN]}',
    ],
)
def test_data_json_rejects_duplicate_keys_and_nonstandard_constants(
    tmp_path: Path, raw_json: str
) -> None:
    (tmp_path / "entities.json").write_text(raw_json, encoding="utf-8")

    with pytest.raises(ValueError):
        DataLoader(str(tmp_path)).load_entities()


def test_prompt_loading_and_strict_instantiation(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Persona={PERSONA}; proposal={PROPOSAL}", encoding="utf-8")

    template = load_prompt_template(str(prompt_path))
    assert instantiate_prompt(template, PERSONA="A", PROPOSAL="B") == (
        "Persona=A; proposal=B"
    )

    with pytest.raises(ValueError, match="Missing prompt values"):
        instantiate_prompt(template, PERSONA="A")

    with pytest.raises(ValueError, match="Unresolved prompt placeholders"):
        instantiate_prompt("Unexpected={lowercase}")

    assert (
        instantiate_prompt(
            "Persona={PERSONA}; proposal={PROPOSAL}",
            PERSONA="literal {PROPOSAL}",
            PROPOSAL="actual policy",
        )
        == "Persona=literal {PROPOSAL}; proposal=actual policy"
    )

    prompt_path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_prompt_template(str(prompt_path))
