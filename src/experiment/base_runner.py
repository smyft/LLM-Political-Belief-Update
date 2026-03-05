"""
Base Experiment Runner Module for Political Belief Experiments.

This module implements the shared functionality for experiment runners.
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, load_prompt_template, instantiate_prompt
from models.unified_llm_interface import UnifiedLLMInterface


class BaseExperimentRunner:
    """
    Base class for experiment runners.
    
    This class manages:
    - Initialization of LLM and data loader
    - Common utility methods
    - Results saving
    """
    
    def __init__(self,
                 model_name: str,
                 data_dir: str = None,
                 prompts_dir: str = None,
                 results_dir: str = None,
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 logprobs: int = 20,
                 seed: int = 42,
                 debug: bool = False,
                 use_api: bool = False,
                 prompt_type: str = "verbalize"):
        """
        Initialize the experiment runner.
        
        Args:
            model_name: Name of the LLM model to use
            data_dir: Path to the data directory
            prompts_dir: Path to the prompts directory (if None, uses default src/prompts)
            results_dir: Path to save results
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            logprobs: Number of top log probabilities
            seed: Random seed
            debug: Debug mode flag
            use_api: Whether to use API instead of vLLM
            prompt_type: Type of prompts to use ('logprob' or 'verbalize')
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.seed = seed
        self.debug = debug
        self.use_api = use_api
        self.prompt_type = prompt_type
        
        # Set up directories
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else project_root / "data"
        
        # Determine prompts directory
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            # Use default prompts directory with prompt_type subdirectory
            self.prompts_dir = project_root / "src" / "prompts" / prompt_type
        
        self.results_dir = Path(results_dir) if results_dir else project_root / "results"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_dir=str(self.data_dir))
        
        # Initialize LLM interface
        self.llm_interface = None
        
        # Load prompt templates
        self.prompt_templates = {}
        
        # Results storage
        self.results = []
    
    def load_prompt_templates(self):
        """Load all prompt templates from files. To be implemented by subclasses."""
        raise NotImplementedError
    
    def initialize_llm(self):
        """Initialize the LLM interface."""
        if self.llm_interface is None:
            self.llm_interface = UnifiedLLMInterface(
                model_name=self.model_name,
                use_api=self.use_api,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                trust_remote_code=True,
            )
    
    def get_persona_prompt(self, persona: str) -> str:
        """
        Generate persona injection prompt.
        
        For political figures, LLM directly plays as that person.
        For political platforms, LLM plays as a typical user of that platform.
        
        Args:
            persona: The persona name ('none' for no injection)
            
        Returns:
            The persona prompt or empty string
        """
        if persona == "none":
            return "Act as an objective policy analyst."
        
        # Check if persona is a politician or platform
        entities = self.data_loader.load_entities()
        politicians = entities.get("politicians", [])
        platforms = entities.get("platforms", [])
        
        if persona in platforms:
            # For platforms, LLM plays as a typical user of that platform
            return f"Act as a typical voter who supports {persona}. Please answer questions based on the political views and values of a typical {persona} supporter."
        else:
            # For politicians, LLM directly plays as that person
            return f"Act as {persona}. Please answer questions based on the political views and values of {persona}."
    
    def save_results(self, filename: str = None):
        """
        Save results to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            model_name_safe = self.model_name.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{model_name_safe}_{timestamp}.json"
        
        output_path = self.results_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.llm_interface:
            self.llm_interface.free_memory()
            self.llm_interface = None
