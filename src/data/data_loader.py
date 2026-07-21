"""Strict loading and validation for experiment entities and proposal actions."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PLACEHOLDER_PATTERN = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
NAMED_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
REQUIRED_ACTION_FIELDS = {"action_type", "action_description"}
EXPECTED_ACTION_TYPES = {
    "Personal Commitment",
    "Public Advocacy",
    "Strategic Support",
}
INSTALLED_DATA_SUBDIRECTORY = Path("share") / "llm-political-belief-update" / "data"


def default_data_directory() -> Path:
    """Resolve source-tree data first, then wheel-installed shared data."""

    source_data = Path(__file__).resolve().parents[2] / "data"
    installed_data = Path(sys.prefix) / INSTALLED_DATA_SUBDIRECTORY
    required = ("entities.json", "proposal_actions.json")
    for candidate in (source_data, installed_data):
        if all((candidate / filename).is_file() for filename in required):
            return candidate
    # Preserve the conventional source path so downstream errors name the
    # expected files clearly when neither installation is complete.
    return source_data


class _DuplicateJsonKey(ValueError):
    pass


def _strict_object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _load_strict_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )


class DataLoader:
    """Data loader for political entities and policy proposals."""

    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory. Defaults to ./data relative to project root.
        """
        if data_dir is None:
            data_dir = default_data_directory()

        self.data_dir = Path(data_dir)
        self.entities = None
        self.proposals = None
        self._action_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    def load_entities(self) -> Dict:
        """
        Load political entities from entities.json.

        Returns:
            Dictionary containing 'politicians' and 'platforms' lists
        """
        if self.entities is None:
            entities_file = self.data_dir / "entities.json"
            self.entities = _load_strict_json(entities_file)
            self._validate_entities(self.entities, entities_file)

        return self.entities

    def get_personas(self, include_none: bool = True) -> List[str]:
        """
        Get list of all available personas for experiments.

        Args:
            include_none: Whether to include "none" (no persona) option

        Returns:
            List of persona strings
        """
        entities = self.load_entities()
        personas = []

        if include_none:
            personas.append("none")

        # Add all politicians
        personas.extend(entities.get("politicians", []))

        # Add all platforms
        personas.extend(entities.get("platforms", []))

        if len(personas) != len(set(personas)):
            raise ValueError("entities.json contains duplicate persona labels")
        return personas

    def load_proposals(self) -> Dict:
        """
        Load policy proposals and corresponding actions from proposal_actions.json.

        Returns:
            Dictionary with categories as keys and lists of proposal-action pairs as values
        """
        if self.proposals is None:
            proposals_file = self.data_dir / "proposal_actions.json"
            self.proposals = _load_strict_json(proposals_file)
            self._validate_proposals(self.proposals, proposals_file)
            self._build_action_index()

        return self.proposals

    def get_unique_proposals(self) -> List[Tuple[str, str]]:
        """
        Get all unique proposals (without actions) from the dataset.

        This is useful for Step 1 and Step 2 which only depend on PERSONA and PROPOSAL,
        not on the specific ACTION.

        Returns:
            List of tuples: (category, proposal)
        """
        proposals = self.load_proposals()
        unique_proposals = []

        for category, proposal_list in proposals.items():
            for item in proposal_list:
                proposal = item.get("political_proposal", "")
                unique_proposals.append((category, proposal))

        return unique_proposals

    def get_actions_for_proposal(
        self, category: str, proposal: str
    ) -> List[Tuple[str, str]]:
        """
        Get all actions for a specific proposal.

        Args:
            category: The category of the proposal
            proposal: The proposal text

        Returns:
            List of tuples: (action_type, action_description)
        """
        self.load_proposals()
        return list(self._action_index.get((category, proposal), []))

    @staticmethod
    def _validate_entities(data: Any, source: Path) -> None:
        if not isinstance(data, dict) or set(data) != {"politicians", "platforms"}:
            raise ValueError(
                f"{source} must contain exactly 'politicians' and 'platforms' lists"
            )
        labels: List[str] = []
        for field in ("politicians", "platforms"):
            values = data[field]
            if not isinstance(values, list):
                raise ValueError(f"{source}: {field} must be a list")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{source}: every {field} entry must be non-empty")
                labels.append(value)
        if len(labels) != len(set(labels)):
            raise ValueError(f"{source} contains duplicate entity labels")

    @staticmethod
    def _validate_proposals(data: Any, source: Path) -> None:
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{source} must be a non-empty object")
        seen_proposals: set[Tuple[str, str]] = set()
        for category, entries in data.items():
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"{source} contains an invalid category")
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{source}: category {category!r} must be non-empty")
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {
                    "political_proposal",
                    "actions",
                }:
                    raise ValueError(
                        f"{source}: each proposal must contain exactly political_proposal and actions"
                    )
                proposal = entry["political_proposal"]
                if not isinstance(proposal, str) or not proposal.strip():
                    raise ValueError(f"{source}: proposal text must be non-empty")
                key = (category, proposal)
                if key in seen_proposals:
                    raise ValueError(f"{source}: duplicate proposal in {category!r}")
                seen_proposals.add(key)

                actions = entry["actions"]
                if not isinstance(actions, list) or len(actions) != len(
                    EXPECTED_ACTION_TYPES
                ):
                    raise ValueError(
                        f"{source}: proposal {proposal!r} must have exactly "
                        f"{len(EXPECTED_ACTION_TYPES)} actions"
                    )
                action_keys: set[Tuple[str, str]] = set()
                action_types: set[str] = set()
                for action in actions:
                    if (
                        not isinstance(action, dict)
                        or set(action) != REQUIRED_ACTION_FIELDS
                    ):
                        raise ValueError(
                            f"{source}: each action must contain exactly action_type and action_description"
                        )
                    action_type = action["action_type"]
                    description = action["action_description"]
                    if not isinstance(action_type, str) or not action_type.strip():
                        raise ValueError(f"{source}: action_type must be non-empty")
                    if not isinstance(description, str) or not description.strip():
                        raise ValueError(
                            f"{source}: action_description must be non-empty"
                        )
                    action_key = (action_type, description)
                    if action_key in action_keys:
                        raise ValueError(f"{source}: duplicate action for {proposal!r}")
                    action_keys.add(action_key)
                    action_types.add(action_type)
                if action_types != EXPECTED_ACTION_TYPES:
                    raise ValueError(
                        f"{source}: proposal {proposal!r} must contain exactly the "
                        "three supported action types"
                    )

    def _build_action_index(self) -> None:
        self._action_index = {}
        for category, entries in self.proposals.items():
            for entry in entries:
                self._action_index[(category, entry["political_proposal"])] = [
                    (action["action_type"], action["action_description"])
                    for action in entry["actions"]
                ]

    def get_proposal_action_pairs(self) -> List[Tuple[str, str, str, str]]:
        """
        Get all proposal-action pairs from the dataset.

        Returns:
            List of tuples: (category, proposal, action_type, action_description)
        """
        proposals = self.load_proposals()
        pairs = []

        for category, proposal_list in proposals.items():
            for item in proposal_list:
                proposal = item.get("political_proposal", "")
                actions = item.get("actions", [])

                for action_item in actions:
                    action_type = action_item.get("action_type", "")
                    action_description = action_item.get("action_description", "")
                    pairs.append((category, proposal, action_type, action_description))

        return pairs

    def generate_distribution(
        self, first_order_belief: float, polarity: str = "positive"
    ) -> str:
        """
        Generate a distribution string for Step 3 experiments.

        Args:
            first_order_belief: Explicit probability in the inclusive range 0-1.
            polarity: Either "positive" or "negative" for the framing

        Returns:
            An explicitly hypothetical survey-vignette statement.
        """
        if not isinstance(first_order_belief, (int, float)) or isinstance(
            first_order_belief, bool
        ):
            raise TypeError("first_order_belief must be a number")
        if not 0 <= first_order_belief <= 1:
            raise ValueError("first_order_belief must be between 0 and 1")
        if polarity not in {"positive", "negative"}:
            raise ValueError("polarity must be 'positive' or 'negative'")

        percentage = round(first_order_belief * 100)

        if polarity == "positive":
            return (
                "In this hypothetical survey vignette, "
                f"{percentage}% of respondents are described as believing this "
                "policy will be beneficial to the United States."
            )
        else:
            negative_pct = 100 - percentage
            return (
                "In this hypothetical survey vignette, "
                f"{negative_pct}% of respondents are described as believing this "
                "policy will be harmful to the United States."
            )

    def get_categories(self) -> List[str]:
        """
        Get all policy categories.

        Returns:
            List of category names
        """
        proposals = self.load_proposals()
        return list(proposals.keys())


def load_prompt_template(template_path: str) -> str:
    """
    Load a prompt template from a text file.

    Args:
        template_path: Path to the template file

    Returns:
        The template string
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    if not template.strip():
        raise ValueError(f"Prompt template is empty: {template_path}")
    return template


def instantiate_prompt(template: str, **kwargs) -> str:
    """
    Instantiate a prompt template with given values.

    Args:
        template: The prompt template string
        **kwargs: Key-value pairs for template placeholders

    Returns:
        The instantiated prompt
    """
    required = set(PLACEHOLDER_PATTERN.findall(template))
    unsupported = sorted(
        set(NAMED_PLACEHOLDER_PATTERN.findall(template)).difference(required)
    )
    if unsupported:
        raise ValueError(f"Unresolved prompt placeholders: {', '.join(unsupported)}")
    missing = sorted(required - set(kwargs))
    if missing:
        raise ValueError(f"Missing prompt values: {', '.join(missing)}")

    for key in required:
        value = kwargs[key]
        if value is None:
            raise ValueError(f"Prompt value {key} must not be None")

    # Substitute against the original template in one regex pass. Repeated
    # ``str.replace`` calls can accidentally reinterpret placeholder-looking
    # text inside an earlier injected persona, proposal, action, or analysis.
    return PLACEHOLDER_PATTERN.sub(
        lambda match: str(kwargs[match.group(1)]),
        template,
    )


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()

    # Test loading entities
    print("Testing data loader...")
    entities = loader.load_entities()
    print(f"Loaded {len(entities.get('politicians', []))} politicians")
    print(f"Loaded {len(entities.get('platforms', []))} platforms")

    # Test getting personas
    personas = loader.get_personas(include_none=True)
    print(f"Total personas: {len(personas)}")

    # Test loading proposals
    proposals = loader.load_proposals()
    print(f"Loaded {len(proposals)} categories")

    # Test getting unique proposals
    unique_proposals = loader.get_unique_proposals()
    print(f"Total unique proposals: {len(unique_proposals)}")

    # Test getting proposal-action pairs
    pairs = loader.get_proposal_action_pairs()
    print(f"Total proposal-action pairs: {len(pairs)}")

    # Test getting actions for a specific proposal
    if unique_proposals:
        cat, prop = unique_proposals[0]
        actions = loader.get_actions_for_proposal(cat, prop)
        print(f"Actions for first proposal: {len(actions)}")

    # Test distribution generation
    dist = loader.generate_distribution(0.75, "positive")
    print(f"Generated distribution: {dist}")
