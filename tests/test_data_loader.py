import hashlib
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


def test_validated_snapshots_are_defensive_copies(valid_data_dir: Path) -> None:
    loader = DataLoader(str(valid_data_dir))

    entities = loader.load_entities()
    entities["politicians"].append("injected")
    assert loader.load_entities()["politicians"] == ["Person A"]
    public_entities = loader.entities
    assert public_entities is not None
    public_entities["platforms"].clear()
    assert loader.get_personas() == ["none", "Person A", "Platform B"]

    proposals = loader.load_proposals()
    proposals["economy"][0]["political_proposal"] = "injected"
    assert loader.get_unique_proposals() == [("economy", "Proposal 1")]
    public_proposals = loader.proposals
    assert public_proposals is not None
    public_proposals.clear()
    assert loader.get_categories() == ["economy"]


def test_failed_validation_does_not_poison_entity_cache(valid_data_dir: Path) -> None:
    entities_path = valid_data_dir / "entities.json"
    valid_entities = json.loads(entities_path.read_text(encoding="utf-8"))
    _write_json(
        entities_path,
        {"politicians": "not-a-list", "platforms": []},
    )
    loader = DataLoader(str(valid_data_dir))

    with pytest.raises(ValueError, match="politicians must be a list"):
        loader.load_entities()

    _write_json(entities_path, valid_entities)
    assert loader.get_personas() == ["none", "Person A", "Platform B"]


def test_failed_validation_does_not_poison_proposal_cache(valid_data_dir: Path) -> None:
    proposals_path = valid_data_dir / "proposal_actions.json"
    valid_proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    _write_json(proposals_path, {"economy": []})
    loader = DataLoader(str(valid_data_dir))

    with pytest.raises(ValueError, match="must be non-empty"):
        loader.load_proposals()

    _write_json(proposals_path, valid_proposals)
    assert loader.get_unique_proposals() == [("economy", "Proposal 1")]


def test_snapshot_hashes_describe_loaded_bytes_and_detect_drift(
    valid_data_dir: Path,
) -> None:
    loader = DataLoader(str(valid_data_dir))
    loader.get_proposal_action_pairs()
    hashes = loader.snapshot_hashes()
    proposal_path = valid_data_dir / "proposal_actions.json"

    assert (
        hashes["proposal_actions.json"]
        == hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    )

    proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposals["economy"][0]["actions"][0]["action_description"] = "Changed action"
    _write_json(proposal_path, proposals)
    with pytest.raises(ValueError, match="changed after its validated data snapshot"):
        loader.snapshot_hashes()


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


def test_entities_reject_reserved_none_persona_at_load_boundary(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "entities.json",
        {"politicians": ["none"], "platforms": []},
    )

    with pytest.raises(ValueError, match="persona label 'none' is reserved"):
        DataLoader(str(tmp_path)).load_entities()


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"politicians":["\\ud800"],"platforms":[]}',
        '{"politicians":[],"platforms":[],"\\udfff":[]}',
    ],
)
def test_data_json_rejects_unpaired_surrogates(tmp_path: Path, raw_json: str) -> None:
    (tmp_path / "entities.json").write_text(raw_json, encoding="utf-8")

    with pytest.raises(ValueError, match="unpaired UTF-16 surrogate"):
        DataLoader(str(tmp_path)).load_entities()


def test_data_json_accepts_valid_supplementary_unicode(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "entities.json",
        {"politicians": ["Person 😀"], "platforms": []},
    )

    assert DataLoader(str(tmp_path)).load_entities()["politicians"] == ["Person 😀"]


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
