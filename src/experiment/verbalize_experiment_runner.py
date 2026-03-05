"""
Verbalize Belief Experiment Runner Module.

This module implements the experiment workflow using verbalized belief extraction (multi-stage).
This runner uses verbalize prompts by default, which ask the LLM to verbalize its thinking process.

Key Design:
- Step 1 and Step 2 run once per (persona, proposal) pair (no redundancy)
- Step 3 runs for each (persona, proposal, action) combination WITHOUT distribution information
- Step 4a and Step 4b run for each (persona, proposal, action) combination WITH various distributions
- Results only contain core data: metadata and answers (no thinking process)
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_prompt_template, instantiate_prompt
from experiment.base_runner import BaseExperimentRunner


# Fixed distribution percentages for Step 4
FIXED_DISTRIBUTION_PERCENTAGES = [10, 30, 50, 70, 90]


class VerbalizeExperimentRunner(BaseExperimentRunner):
    """
    Experiment runner for verbalized belief extraction.
    
    This class uses verbalize prompts by default, which instruct the LLM to:
    - Think step by step
    - Provide reasoning in a "thinking" field
    - Provide final answer in an "answer" field
    - Output in JSON format
    
    Key optimization: Step 1 and Step 2 run only once per (persona, proposal) pair,
    avoiding redundant runs for proposals with multiple actions.
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
        Initialize the verbalize experiment runner.
        
        Args:
            model_name: Name of the LLM model to use
            data_dir: Path to the data directory
            prompts_dir: Path to the prompts directory
            results_dir: Path to save results
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            logprobs: Number of top log probabilities
            seed: Random seed
            debug: Debug mode flag
            use_api: Whether to use API instead of vLLM
            prompt_type: Type of prompts to use ('logprob' or 'verbalize')
        """
        super().__init__(
            model_name=model_name,
            data_dir=data_dir,
            prompts_dir=prompts_dir,
            results_dir=results_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            seed=seed,
            debug=debug,
            use_api=use_api,
            prompt_type=prompt_type
        )
        
        # Create intermediate results directory
        self.intermediate_dir = self.results_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment ID for this run
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.model_name.replace("/", "_")
        self.experiment_prefix = f"{model_name_safe}_{self.experiment_id}"
    
    def load_prompt_templates(self):
        """Load prompt templates for verbalize belief experiments."""
        template_files = {
            "step1": "step1.txt",
            "step2": "step2.txt",
            "step3": "step3.txt",
            "step4a": "step4a.txt",
            "step4b": "step4b.txt",
        }
        
        for key, filename in template_files.items():
            template_path = self.prompts_dir / filename
            if template_path.exists():
                self.prompt_templates[key] = load_prompt_template(str(template_path))
                print(f"Loaded template: {filename}")
            else:
                print(f"Warning: Template not found: {template_path}")
    
    def _save_step_results(self, step_name: str, results: List[Dict], metadata: List[Dict] = None):
        """
        Save intermediate results for a specific step.
        
        Args:
            step_name: Name of the step (e.g., 'step1', 'step2', 'step3', 'step4a', 'step4b')
            results: List of LLM response results
            metadata: Optional metadata list to include
        """
        output_path = self.intermediate_dir / f"{self.experiment_prefix}_{step_name}.json"
        
        save_data = {
            "experiment_id": self.experiment_id,
            "model": self.model_name,
            "step": step_name,
            # "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        if metadata:
            save_data["metadata"] = metadata
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {step_name} results to: {output_path}")
    
    def _generate_distribution_text(self, percentage: int) -> str:
        """
        Generate distribution text from percentage.
        
        Args:
            percentage: Percentage value (0-100)
            
        Returns:
            Distribution text string
        """
        return f"{percentage}% of the U.S. population believe this policy will be beneficial to the United States"
    
    def _compute_step1_yes_ratio(self, step1_results: List[Dict], step1_metadata: List[Dict]) -> Dict[str, float]:
        """
        Compute the ratio of "Yes" answers for each proposal from Step 1 results.
        
        Args:
            step1_results: List of Step 1 response results
            step1_metadata: Metadata list for Step 1 (unique persona-proposal pairs)
            
        Returns:
            Dictionary mapping proposal to Yes ratio
        """
        proposal_yes_counts = defaultdict(lambda: {"yes": 0, "total": 0})
        
        for i, result in enumerate(step1_results):
            step1_text = result.get("generated_text", "")
            step1_data = self._parse_json_response(step1_text)
            belief = step1_data.get("answer") if step1_data else None
            
            proposal = step1_metadata[i]["proposal"]
            proposal_yes_counts[proposal]["total"] += 1
            if belief and belief.lower() == "yes":
                proposal_yes_counts[proposal]["yes"] += 1
        
        # Compute ratio
        proposal_ratio = {}
        for proposal, counts in proposal_yes_counts.items():
            if counts["total"] > 0:
                proposal_ratio[proposal] = counts["yes"] / counts["total"]
            else:
                proposal_ratio[proposal] = 0.5  # default
        
        return proposal_ratio
    
    def _get_all_distribution_percentages(self, inferred_percentage: int = None) -> List[int]:
        """
        Get all distribution percentages to test including fixed and inferred.
        
        Args:
            inferred_percentage: The inferred percentage from Step 1 (if available)
            
        Returns:
            List of unique percentage values to test
        """
        percentages = set(FIXED_DISTRIBUTION_PERCENTAGES)
        if inferred_percentage is not None:
            # Round to nearest integer and clamp to 0-100
            inferred_pct = max(0, min(100, int(round(inferred_percentage))))
            percentages.add(inferred_pct)
        return sorted(list(percentages))
    
    def run_step1(self, personas: List[str], unique_proposals: List[Tuple[str, str]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 1: First-order Belief (persona's own opinion on policy).
        
        This runs once per unique (persona, proposal) pair, not per action.
        
        Args:
            personas: List of personas to test
            unique_proposals: List of (category, proposal) tuples
            
        Returns:
            Tuple of (step1_results, step1_metadata)
        """
        print("\n=== Step 1: First-order Belief ===")
        
        dialogues = []
        step1_metadata = []
        
        for persona in personas:
            for category, proposal in unique_proposals:
                persona_prompt = self.get_persona_prompt(persona)
                
                user_prompt = instantiate_prompt(
                    self.prompt_templates.get("step1", ""),
                    POLICY_PROPOSAL=proposal,
                    PERSONA_INJECTION=persona_prompt
                )
                dialogue = [{"role": "user", "content": user_prompt}]
                dialogues.append(dialogue)
                
                step1_metadata.append({
                    "persona": persona,
                    "category": category,
                    "proposal": proposal,
                })
        
        print(f"Processing {len(dialogues)} unique (persona, proposal) pairs...")
        
        step1_results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc="Step 1 (First-order Belief)"
        )
        
        # Save intermediate results
        self._save_step_results("step1", step1_results, step1_metadata)
        
        return step1_results, step1_metadata
    
    def run_step2(self, step1_metadata: List[Dict]) -> List[Dict]:
        """
        Run Step 2: Second-order Belief (prediction of population opinion).
        
        This runs once per unique (persona, proposal) pair, matching Step 1.
        
        Args:
            step1_metadata: Metadata from Step 1 (unique persona-proposal pairs)
            
        Returns:
            List of Step 2 results
        """
        print("\n=== Step 2: Second-order Belief (Population Prediction) ===")
        
        dialogues = []
        
        for meta in step1_metadata:
            persona_prompt = self.get_persona_prompt(meta['persona'])
            
            user_prompt = instantiate_prompt(
                self.prompt_templates.get("step2", ""),
                POLICY_PROPOSAL=meta['proposal'],
                PERSONA_INJECTION=persona_prompt
            )
            dialogue = [{"role": "user", "content": user_prompt}]
            dialogues.append(dialogue)
        
        print(f"Processing {len(dialogues)} unique (persona, proposal) pairs...")
        
        step2_results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc="Step 2 (Second-order Belief)"
        )
        
        # Save intermediate results
        self._save_step_results("step2", step2_results, step1_metadata)
        
        return step2_results
    
    def run_step3(self, step3_base_metadata: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 3: Action Support WITHOUT Distribution Information.
        
        This step asks whether the LLM would support a specific action without
        knowing any population distribution information.
        
        Args:
            step3_base_metadata: Metadata for Step 3 experiments (persona, proposal, action combinations)
            
        Returns:
            Tuple of (step3_results, step3_metadata)
        """
        print("\n=== Step 3: Action Support (No Distribution) ===")
        
        dialogues = []
        step3_metadata = []
        
        for meta in step3_base_metadata:
            persona_prompt = self.get_persona_prompt(meta['persona'])
            
            user_prompt = instantiate_prompt(
                self.prompt_templates.get("step3", ""),
                POLICY_PROPOSAL=meta['proposal'],
                PERSONA_INJECTION=persona_prompt,
                CORRESPONDING_ACTION=meta['action']
            )
            dialogue = [{"role": "user", "content": user_prompt}]
            dialogues.append(dialogue)
            
            step3_metadata.append({
                "persona": meta['persona'],
                "category": meta['category'],
                "proposal": meta['proposal'],
                "action_type": meta['action_type'],
                "action": meta['action'],
            })
        
        print(f"Processing {len(dialogues)} dialogues...")
        
        step3_results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc="Step 3 (Action Support - No Distribution)"
        )
        
        # Save intermediate results
        self._save_step_results("step3", step3_results, step3_metadata)
        
        return step3_results, step3_metadata
    
    def run_step4a(self, step4_base_metadata: List[Dict], step1_yes_ratio: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 4a: First-order Belief with Distribution Information.
        
        For each (persona, proposal, action) combination, iterate through all possible
        distributions (fixed percentages + inferred from Step 1).
        
        Args:
            step4_base_metadata: Metadata for Step 4 experiments (persona, proposal, action combinations)
            step1_yes_ratio: Dictionary mapping proposal to Yes ratio from Step 1
            
        Returns:
            Tuple of (step4a_results, step4a_metadata)
        """
        print("\n=== Step 4a: First-order Belief with Distribution ===")
        
        dialogues = []
        step4a_metadata = []
        
        for meta in step4_base_metadata:
            proposal = meta['proposal']
            
            # Get inferred percentage from Step 1 yes ratio
            inferred_ratio = step1_yes_ratio.get(proposal, 0.5)
            inferred_percentage = int(round(inferred_ratio * 100))
            
            # Get all distribution percentages to test
            percentages = self._get_all_distribution_percentages(inferred_percentage)
            
            for percentage in percentages:
                distribution_text = self._generate_distribution_text(percentage)
                persona_prompt = self.get_persona_prompt(meta['persona'])
                
                user_prompt = instantiate_prompt(
                    self.prompt_templates.get("step4a", ""),
                    POLICY_PROPOSAL=proposal,
                    PERSONA_INJECTION=persona_prompt,
                    DISTRIBUTION=distribution_text
                )
                dialogue = [{"role": "user", "content": user_prompt}]
                dialogues.append(dialogue)
                
                step4a_metadata.append({
                    **meta,
                    "distribution_percentage": percentage,
                    "distribution_text": distribution_text,
                    "is_inferred": percentage == inferred_percentage,
                })
        
        print(f"Processing {len(dialogues)} dialogues...")
        
        step4a_results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc="Step 4a (Belief with Distribution)"
        )
        
        # Save intermediate results
        self._save_step_results("step4a", step4a_results, step4a_metadata)
        
        return step4a_results, step4a_metadata
    
    def run_step4b(self, step4_base_metadata: List[Dict], step1_yes_ratio: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 4b: Action Support with Distribution Information.
        
        For each (persona, proposal, action) combination, iterate through all possible
        distributions (fixed percentages + inferred from Step 1).
        
        Args:
            step4_base_metadata: Metadata for Step 4 experiments (persona, proposal, action combinations)
            step1_yes_ratio: Dictionary mapping proposal to Yes ratio from Step 1
            
        Returns:
            Tuple of (step4b_results, step4b_metadata)
        """
        print("\n=== Step 4b: Action Support with Distribution ===")
        
        dialogues = []
        step4b_metadata = []
        
        for meta in step4_base_metadata:
            proposal = meta['proposal']
            action = meta['action']
            
            # Get inferred percentage from Step 1 yes ratio
            inferred_ratio = step1_yes_ratio.get(proposal, 0.5)
            inferred_percentage = int(round(inferred_ratio * 100))
            
            # Get all distribution percentages to test
            percentages = self._get_all_distribution_percentages(inferred_percentage)
            
            for percentage in percentages:
                distribution_text = self._generate_distribution_text(percentage)
                persona_prompt = self.get_persona_prompt(meta['persona'])
                
                user_prompt = instantiate_prompt(
                    self.prompt_templates.get("step4b", ""),
                    POLICY_PROPOSAL=proposal,
                    PERSONA_INJECTION=persona_prompt,
                    DISTRIBUTION=distribution_text,
                    CORRESPONDING_ACTION=action
                )
                dialogue = [{"role": "user", "content": user_prompt}]
                dialogues.append(dialogue)
                
                step4b_metadata.append({
                    **meta,
                    "distribution_percentage": percentage,
                    "distribution_text": distribution_text,
                    "is_inferred": percentage == inferred_percentage,
                })
        
        print(f"Processing {len(dialogues)} dialogues...")
        
        step4b_results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc="Step 4b (Action Support)"
        )
        
        # Save intermediate results
        self._save_step_results("step4b", step4b_results, step4b_metadata)
        
        return step4b_results, step4b_metadata
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse JSON response from LLM.
        
        Expected format (verbalize prompts):
        {
            "thinking": "...",
            "answer": "Yes" or "No"
        }
        
        Or for Step 2:
        {
            "thinking": "...",
            "answer": 75  # or "75%" or "75 percent"
        }
        
        Args:
            response_text: The raw response text from LLM
            
        Returns:
            Parsed JSON as dict, or None if parsing fails
        """
        if not response_text:
            return None
        
        # Try to find JSON in the response
        try:
            # First try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON within markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*"thinking"[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_percentage_from_response(self, response_text: str) -> Optional[int]:
        """
        Extract percentage from JSON response.
        
        Expected format:
        {
            "thinking": "...",
            "answer": 75  # or "75%" or "75 percent"
        }
        
        Args:
            response_text: The raw response text from LLM
            
        Returns:
            Extracted percentage (0-100), or None if extraction fails
        """
        if not response_text:
            return None
        
        # First try to parse as JSON
        json_data = self._parse_json_response(response_text)
        if json_data:
            answer = json_data.get("answer")
            if answer is not None:
                # If answer is already a number
                if isinstance(answer, (int, float)):
                    return max(0, min(100, int(answer)))
                # If answer is a string
                if isinstance(answer, str):
                    # Try to extract number
                    numbers = re.findall(r'\d+', answer)
                    if numbers:
                        return max(0, min(100, int(numbers[0])))
        
        # Fallback: try regex on raw text
        try:
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                predicted_pct = int(numbers[0])
                return max(0, min(100, predicted_pct))
        except:
            pass
        
        return None
    
    def _compile_results(self, 
                        step1_results: List[Dict], step1_metadata: List[Dict],
                        step2_results: List[Dict],
                        step3_results: List[Dict], step3_metadata: List[Dict],
                        step4a_results: List[Dict], step4a_metadata: List[Dict],
                        step4b_results: List[Dict], step4b_metadata: List[Dict],
                        step_base_metadata: List[Dict]) -> List[Dict]:
        """
        Compile all step results into final experiment results.
        
        Only keeps core data:
        - Metadata: PERSONA_INJECTION, POLICY_PROPOSAL, DISTRIBUTION, CORRESPONDING_ACTION
        - Each step's answer (no thinking process)
        
        Args:
            step1_results: Results from Step 1
            step1_metadata: Metadata for Step 1 (unique persona-proposal pairs)
            step2_results: Results from Step 2
            step3_results: Results from Step 3 (action support without distribution)
            step3_metadata: Metadata for Step 3
            step4a_results: Results from Step 4a
            step4a_metadata: Metadata for Step 4a
            step4b_results: Results from Step 4b
            step4b_metadata: Metadata for Step 4b
            step_base_metadata: Base metadata for Step 3/4 (persona, proposal, action combinations)
            
        Returns:
            List of compiled experiment results
        """
        compiled_results = []
        
        # Create lookup for Step 1 and Step 2 results by (persona, proposal)
        step1_lookup = {}
        step2_lookup = {}
        for i, meta in enumerate(step1_metadata):
            key = (meta['persona'], meta['proposal'])
            
            # Parse Step 1 result
            step1_text = step1_results[i].get("generated_text", "")
            step1_data = self._parse_json_response(step1_text)
            step1_lookup[key] = {
                "answer": step1_data.get("answer") if step1_data else None,
            }
            
            # Parse Step 2 result
            step2_response = step2_results[i].get("generated_text", "").strip()
            predicted_pct = self._extract_percentage_from_response(step2_response)
            step2_lookup[key] = {
                "predicted_percentage": predicted_pct,
            }
        
        # Create lookup for Step 3 results by (persona, proposal, action)
        step3_lookup = {}
        for i, meta in enumerate(step3_metadata):
            key = (meta['persona'], meta['proposal'], meta['action'])
            result_text = step3_results[i].get("generated_text", "")
            result_data = self._parse_json_response(result_text)
            step3_lookup[key] = {
                "answer": result_data.get("answer") if result_data else None,
            }
        
        # Create lookup for Step 4a and Step 4b results by (persona, proposal, action, distribution_percentage)
        step4a_lookup = {}
        for i, meta in enumerate(step4a_metadata):
            key = (meta['persona'], meta['proposal'], meta['action'], meta['distribution_percentage'])
            result_text = step4a_results[i].get("generated_text", "")
            result_data = self._parse_json_response(result_text)
            step4a_lookup[key] = {
                "answer": result_data.get("answer") if result_data else None,
                "distribution_text": meta['distribution_text'],
                "is_inferred": meta['is_inferred'],
            }
        
        step4b_lookup = {}
        for i, meta in enumerate(step4b_metadata):
            key = (meta['persona'], meta['proposal'], meta['action'], meta['distribution_percentage'])
            result_text = step4b_results[i].get("generated_text", "")
            result_data = self._parse_json_response(result_text)
            step4b_lookup[key] = {
                "answer": result_data.get("answer") if result_data else None,
                "distribution_text": meta['distribution_text'],
                "is_inferred": meta['is_inferred'],
            }
        
        # Process each (persona, proposal, action) combination
        for meta in step_base_metadata:
            persona = meta['persona']
            proposal = meta['proposal']
            action = meta['action']
            
            # Get Step 1 and Step 2 results
            step12_key = (persona, proposal)
            step1_result = step1_lookup.get(step12_key, {"answer": None})
            step2_result = step2_lookup.get(step12_key, {"predicted_percentage": None})
            
            # Get Step 3 result (action support without distribution)
            step3_key = (persona, proposal, action)
            step3_result = step3_lookup.get(step3_key, {"answer": None})
            
            # Collect Step 4a results for all distributions
            step4a_by_distribution = {}
            for key, data in step4a_lookup.items():
                if key[0] == persona and key[1] == proposal and key[2] == action:
                    percentage = key[3]
                    step4a_by_distribution[percentage] = {
                        "answer": data['answer'],
                        "distribution_text": data['distribution_text'],
                        "is_inferred": data['is_inferred']
                    }
            
            # Collect Step 4b results for all distributions
            step4b_by_distribution = {}
            for key, data in step4b_lookup.items():
                if key[0] == persona and key[1] == proposal and key[2] == action:
                    percentage = key[3]
                    step4b_by_distribution[percentage] = {
                        "answer": data['answer'],
                        "distribution_text": data['distribution_text'],
                        "is_inferred": data['is_inferred']
                    }
            
            final_result = {
                "experiment_id": self.experiment_id,
                "model": self.model_name,
                # "timestamp": datetime.now().isoformat(),
                
                # Core metadata
                "persona": persona,
                "category": meta['category'],
                "policy_proposal": proposal,
                "action_type": meta['action_type'],
                "corresponding_action": action,
                
                # Step results (only answers, no thinking)
                "step1_first_order_belief": step1_result,
                "step2_second_order_belief": step2_result,
                "step3_action_support_no_distribution": step3_result,
                "step4a_first_order_with_distribution": step4a_by_distribution,
                "step4b_action_support_with_distribution": step4b_by_distribution
            }
            compiled_results.append(final_result)
        
        return compiled_results

    def run_experiments(self, personas: List[str] = None,
                       unique_proposals: List[Tuple[str, str]] = None,
                       max_experiments: int = None):
        """
        Run verbalize belief experiments.
        
        This method orchestrates all steps:
        - Step 1: First-order belief (persona's own opinion on policy) - once per (persona, proposal)
        - Step 2: Second-order belief (prediction of population opinion) - once per (persona, proposal)
        - Step 3: Action support WITHOUT distribution information - per (persona, proposal, action)
        - Step 4a: First-order belief with distribution information - per (persona, proposal, action)
        - Step 4b: Action support WITH distribution information - per (persona, proposal, action)
        
        Args:
            personas: List of personas to test (None = all)
            unique_proposals: List of (category, proposal) tuples (None = all)
            max_experiments: Maximum number of base experiments (None = all)
        """
        # Handle debug mode - limit to 2 proposals with ALL actions
        if self.debug:
            print("Debug mode enabled: Limiting to 2 proposals with ALL actions.")
            all_unique_proposals = self.data_loader.get_unique_proposals()
            unique_proposals = all_unique_proposals[:2]
            print(f"Debug: Using {len(unique_proposals)} unique proposals")
            max_experiments = None  # No limit, run all actions for these 2 proposals

        # Load prompt templates
        print("Loading prompt templates...")
        self.load_prompt_templates()
        
        # Initialize LLM
        print(f"Initializing LLM: {self.model_name}...")
        self.initialize_llm()
        
        # Get personas
        if personas is None:
            personas = self.data_loader.get_personas(include_none=True)

        if self.debug:
            personas = [personas[0], personas[1], personas[33]] # hard code None, one politician and one platform
        
        print(f"Testing {len(personas)} personas...")
        
        # Get unique proposals for Step 1 and Step 2
        if unique_proposals is None:
            unique_proposals = self.data_loader.get_unique_proposals()
        
        print(f"Testing {len(unique_proposals)} unique proposals...")
        
        # Calculate Step 1/2 experiments (no redundancy)
        step12_count = len(personas) * len(unique_proposals)
        print(f"Total Step 1/2 experiments (unique persona-proposal pairs): {step12_count}")
        
        # Run Step 1: First-order Belief (once per unique persona-proposal pair)
        step1_results, step1_metadata = self.run_step1(personas, unique_proposals)
        
        # Run Step 2: Second-order Belief (once per unique persona-proposal pair)
        step2_results = self.run_step2(step1_metadata)
        
        # Compute Step 1 Yes ratio per proposal for distribution inference
        step1_yes_ratio = self._compute_step1_yes_ratio(step1_results, step1_metadata)
        print(f"Computed Step 1 Yes ratios for {len(step1_yes_ratio)} proposals")
        
        # Build Step 3/4 metadata: (persona, proposal, action) combinations
        step_base_metadata = []
        for persona in personas:
            for category, proposal in unique_proposals:
                actions = self.data_loader.get_actions_for_proposal(category, proposal)
                for action_type, action_description in actions:
                    step_base_metadata.append({
                        "persona": persona,
                        "category": category,
                        "proposal": proposal,
                        "action_type": action_type,
                        "action": action_description,
                    })
        
        # Limit to max_experiments if specified (non-debug mode)
        if max_experiments and not self.debug:
            step_base_metadata = step_base_metadata[:max_experiments]
        
        print(f"Total Step 3/4 base experiments (persona-proposal-action combinations): {len(step_base_metadata)}")
        
        # Run Step 3: Action Support WITHOUT Distribution
        step3_results, step3_metadata = self.run_step3(step_base_metadata)
        
        # Run Step 4a: First-order Belief with Distribution
        step4a_results, step4a_metadata = self.run_step4a(step_base_metadata, step1_yes_ratio)
        
        # Run Step 4b: Action Support with Distribution
        step4b_results, step4b_metadata = self.run_step4b(step_base_metadata, step1_yes_ratio)
        
        # Compile all results
        print("\n=== Compiling Results ===")
        self.results = self._compile_results(
            step1_results, step1_metadata,
            step2_results,
            step3_results, step3_metadata,
            step4a_results, step4a_metadata,
            step4b_results, step4b_metadata,
            step_base_metadata
        )
        
        # Save final results
        self.save_results()
        print(f"\nCompleted {len(self.results)} experiments.")


def main():
    """Main entry point for the verbalize experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run political belief experiments with LLM (Verbalize)"
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                        help="Model name")
    
    # Experiment configuration
    parser.add_argument("--personas", type=str, nargs="*", default=None,
                        help="Specific personas to test (default: all)")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Specific categories to test (default: all)")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Maximum number of experiments to run")
    
    # LLM configuration
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output configuration
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")
    parser.add_argument("--use-api", action="store_true",
                        help="Use API instead of vLLM")
    
    # Prompt configuration
    parser.add_argument("--prompt-type", type=str, default="verbalize",
                        choices=["logprob", "verbalize"],
                        help="Type of prompts to use")
    parser.add_argument("--prompts-dir", type=str, default=None,
                        help="Custom prompts directory (overrides prompt-type)")
    
    cmd = ["--use-api", "--debug"]
    args = parser.parse_args(cmd)
    
    # Create experiment runner
    runner = VerbalizeExperimentRunner(
        model_name=args.model,
        results_dir=args.results_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        debug=args.debug,
        use_api=args.use_api,
        prompts_dir=args.prompts_dir,
        prompt_type=args.prompt_type
    )
    
    # Get personas
    personas = args.personas if args.personas else None
    
    # Get unique proposals
    if args.categories:
        unique_proposals = []
        all_proposals = runner.data_loader.get_unique_proposals()
        for cat, prop in all_proposals:
            if cat in args.categories:
                unique_proposals.append((cat, prop))
    else:
        unique_proposals = None
    
    # Run experiments
    try:
        runner.run_experiments(
            personas=personas,
            unique_proposals=unique_proposals,
            max_experiments=args.max_experiments
        )
    finally:
        runner.cleanup()
    
    print("Experiment completed!")


if __name__ == "__main__":
    main()
