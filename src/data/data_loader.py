"""
Data loader module for loading entities and policy proposals.

This module provides functions to:
- Load political entities (politicians and platforms) from entities.json
- Load policy proposals and corresponding actions from proposal_actions.json
- Generate distribution information for Step 3 experiments

Author: [Author Name]
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """Data loader for political entities and policy proposals."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory. Defaults to ./data relative to project root.
        """
        if data_dir is None:
            # Default to ./data relative to project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
        
        self.data_dir = Path(data_dir)
        self.entities = None
        self.proposals = None
    
    def load_entities(self) -> Dict:
        """
        Load political entities from entities.json.
        
        Returns:
            Dictionary containing 'politicians' and 'platforms' lists
        """
        if self.entities is None:
            entities_file = self.data_dir / "entities.json"
            with open(entities_file, 'r', encoding='utf-8') as f:
                self.entities = json.load(f)
        
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
        
        return personas
    
    def load_proposals(self) -> Dict:
        """
        Load policy proposals and corresponding actions from proposal_actions.json.
        
        Returns:
            Dictionary with categories as keys and lists of proposal-action pairs as values
        """
        if self.proposals is None:
            proposals_file = self.data_dir / "proposal_actions.json"
            with open(proposals_file, 'r', encoding='utf-8') as f:
                self.proposals = json.load(f)
        
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
    
    def get_actions_for_proposal(self, category: str, proposal: str) -> List[Tuple[str, str]]:
        """
        Get all actions for a specific proposal.
        
        Args:
            category: The category of the proposal
            proposal: The proposal text
            
        Returns:
            List of tuples: (action_type, action_description)
        """
        proposals = self.load_proposals()
        actions = []
        
        if category in proposals:
            for item in proposals[category]:
                if item.get("political_proposal", "") == proposal:
                    for action_item in item.get("actions", []):
                        action_type = action_item.get("action_type", "")
                        action_description = action_item.get("action_description", "")
                        actions.append((action_type, action_description))
                    break
        
        return actions
    
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
    
    def generate_distribution(self, first_order_belief: float = None, 
                           polarity: str = "positive") -> str:
        """
        Generate a distribution string for Step 3 experiments.
        
        Args:
            first_order_belief: The first-order belief as a probability (0-1).
                                If None, generates a random distribution.
            polarity: Either "positive" or "negative" for the framing
            
        Returns:
            Distribution string like "85% of the U.S. population believe this policy will be beneficial to the United States"
        """
        if first_order_belief is None:
            # Generate a random distribution for simulation
            first_order_belief = random.uniform(0.2, 0.8)
        
        percentage = int(first_order_belief * 100)
        
        if polarity == "positive":
            return f"{percentage}% of the U.S. population believe this policy will be beneficial to the United States"
        else:
            negative_pct = 100 - percentage
            return f"{negative_pct}% of the U.S. population believe this policy will be harmful to the United States"
    
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
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def instantiate_prompt(template: str, **kwargs) -> str:
    """
    Instantiate a prompt template with given values.
    
    Args:
        template: The prompt template string
        **kwargs: Key-value pairs for template placeholders
        
    Returns:
        The instantiated prompt
    """
    result = template
    
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if value is None:
            # Replace with empty string if value is None
            value = ""
        result = result.replace(placeholder, str(value))
    
    return result


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
