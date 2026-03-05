"""
Logprob Belief Experiment Runner Module.

This module implements the experiment workflow using logprob-based belief extraction (two-phase approach).
This runner uses vLLM to extract probability distributions over Yes/No answers.

Key Design:
- Each step is split into two phases:
  - Phase 1: Extract the LLM's thinking/reasoning process
  - Phase 2: Inject thinking process into assistant message, add suffix to guide Yes/No answer,
             then extract next token logprob distribution
- Step 1 and Step 2 run once per (persona, proposal) pair (no redundancy)
- Step 3 runs for each (persona, proposal, action) combination WITHOUT distribution information
- Step 4a and Step 4b run for each (persona, proposal, action) combination WITH various distributions
- Results contain probability distributions over Yes/No instead of single answers
"""

import argparse
import json
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

class LogprobExperimentRunner(BaseExperimentRunner):
    """
    Experiment runner for logprob-based belief extraction.
    
    This class uses a two-phase approach for each step:
    - Phase 1: Generate the LLM's thinking/reasoning process
    - Phase 2: Continue from the thinking process to extract Yes/No probability distribution
    
    Key difference from VerbalizeExperimentRunner:
    - Uses vLLM for local model inference with logprob extraction
    - Returns probability distributions over Yes/No instead of single answers
    - Cannot use API-based models (requires logprob access)
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
                 prompt_type: str = "logprob",
                 resume_from_checkpoint: str = None):
        """
        Initialize the logprob experiment runner.
        
        Args:
            model_name: Name of the LLM model to use (must be compatible with vLLM)
            data_dir: Path to the data directory
            prompts_dir: Path to the prompts directory
            results_dir: Path to save results
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate for thinking phase
            logprobs: Number of top log probabilities to extract
            seed: Random seed
            debug: Debug mode flag
            use_api: Must be False (logprob extraction requires vLLM)
            prompt_type: Type of prompts to use (default: 'logprob')
            resume_from_checkpoint: Experiment prefix to resume from (e.g., 'meta-llama_Llama-3.1-8B-Instruct_20260303_095200')
        """
        if use_api:
            raise ValueError("LogprobExperimentRunner requires vLLM backend. Set use_api=False.")
        
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
            use_api=use_api,  # Force vLLM
            prompt_type=prompt_type
        )
        
        # Create intermediate results directory
        self.intermediate_dir = self.results_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle experiment ID and prefix
        if resume_from_checkpoint:
            # Use existing checkpoint
            self.experiment_prefix = resume_from_checkpoint
            # Extract experiment_id from prefix (last 15 chars: YYYYMMDD_HHMMSS)
            self.experiment_id = resume_from_checkpoint[-15:]
            self.is_resuming = True
            print(f"Resuming from checkpoint: {self.experiment_prefix}")
        else:
            # Generate new experiment ID
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_safe = self.model_name.replace("/", "_")
            self.experiment_prefix = f"{model_name_safe}_{self.experiment_id}"
            self.is_resuming = False
    
    def load_prompt_templates(self):
        """Load prompt templates for logprob belief experiments (both phases)."""
        template_files = {
            # Phase 1 templates (thinking extraction)
            "step1_phase1": "step1_phase1.txt",
            "step3_phase1": "step3_phase1.txt",
            "step4a_phase1": "step4a_phase1.txt",
            "step4b_phase1": "step4b_phase1.txt",
            # Phase 2 templates (logprob extraction suffix)
            "step1_phase2": "step1_phase2.txt",
            "step3_phase2": "step3_phase2.txt",
            "step4a_phase2": "step4a_phase2.txt",
            "step4b_phase2": "step4b_phase2.txt",
            # Step 2: Single-stage template (mimicking verbalize approach)
            "step2": "step2.txt",
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
            step_name: Name of the step (e.g., 'step1_phase1', 'step1_phase2')
            results: List of LLM response results
            metadata: Optional metadata list to include
        """
        output_path = self.intermediate_dir / f"{self.experiment_prefix}_{step_name}.json"
        
        save_data = {
            "experiment_id": self.experiment_id,
            "model": self.model_name,
            "step": step_name,
            "results": results
        }
        
        if metadata:
            save_data["metadata"] = metadata
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {step_name} results to: {output_path}")
    
    def _load_step_results(self, step_name: str) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Load intermediate results and metadata for a specific step.
        
        Args:
            step_name: Name of the step (e.g., 'step1_phase2', 'step3_phase2')
            
        Returns:
            Tuple of (results, metadata) or (None, None) if file not found
        """
        file_path = self.intermediate_dir / f"{self.experiment_prefix}_{step_name}.json"
        
        if not file_path.exists():
            print(f"Warning: Checkpoint file not found: {file_path}")
            return None, None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get("results", [])
            metadata = data.get("metadata", [])
            
            print(f"Loaded {step_name} results from: {file_path}")
            print(f"  - {len(results)} results")
            print(f"  - {len(metadata)} metadata entries" if metadata else "  - No metadata")
            
            return results, metadata
        except Exception as e:
            print(f"Error loading checkpoint file {file_path}: {e}")
            return None, None
    
    def _list_available_checkpoints(self) -> Dict[str, bool]:
        """
        List all available checkpoint files for the current experiment prefix.
        
        Returns:
            Dictionary mapping step names to whether the checkpoint exists
        """
        step_names = [
            "step1_phase1", "step1_phase2",
            "step2",
            "step3_phase1", "step3_phase2",
            "step4a_phase1", "step4a_phase2",
            "step4b_phase1", "step4b_phase2",
        ]
        
        available = {}
        for step_name in step_names:
            file_path = self.intermediate_dir / f"{self.experiment_prefix}_{step_name}.json"
            available[step_name] = file_path.exists()
        
        return available
    
    def _get_resume_dependencies(self, resume_step: str) -> List[str]:
        """
        Get the list of required checkpoint steps for resuming from a given step.
        
        Defines the dependency relationship between steps:
        - step1: No dependencies
        - step2: Requires step1_phase2 (to get metadata)
        - step3: Requires step1_phase2 (to build step_base_metadata)
        - step4a: Requires step1_phase2 (for step1_yes_ratio) and step3_phase2 (metadata)
        - step4b: Requires step1_phase2 (for step1_yes_ratio) and step3_phase2 (metadata)
        
        Args:
            resume_step: The step to resume from (e.g., 'step1', 'step2', 'step3', 'step4a', 'step4b')
            
        Returns:
            List of required checkpoint step names
        """
        dependencies = {
            "step1": [],
            "step2": ["step1_phase2"],
            "step3": ["step1_phase2"],
            "step4a": ["step1_phase2", "step3_phase2"],
            "step4b": ["step1_phase2", "step3_phase2"],
        }
        
        return dependencies.get(resume_step, [])
    
    def _check_resume_feasibility(self, resume_step: str) -> bool:
        """
        Check if resuming from a given step is feasible.
        
        Verifies that all required checkpoint files for the given step exist.
        
        Args:
            resume_step: The step to resume from (e.g., 'step1', 'step2', 'step3', 'step4a', 'step4b')
            
        Returns:
            True if all required checkpoints exist, False otherwise
        """
        required_steps = self._get_resume_dependencies(resume_step)
        
        if not required_steps:
            # Steps with no dependencies are always feasible
            print(f"✓ Step {resume_step} has no dependencies, resumption is always feasible")
            return True
        
        available = self._list_available_checkpoints()
        missing = []
        
        for required_step in required_steps:
            if not available[required_step]:
                missing.append(required_step)
        
        if missing:
            print(f"Error: Cannot resume from {resume_step}. Missing required checkpoints:")
            for step in missing:
                print(f"  - {step}")
            return False
        
        print(f"✓ All required checkpoints available for resuming from {resume_step}")
        return True
    
    def _generate_distribution_text(self, percentage: int) -> str:
        """
        Generate distribution text from percentage.
        
        Args:
            percentage: Percentage value (0-100)
            
        Returns:
            Distribution text string
        """
        return f"{percentage}% of the U.S. population believe this policy will be beneficial to the United States"
    
    def _run_phase1(self, dialogues: List[List[Dict]], desc: str = "Phase 1") -> List[Dict]:
        """
        Run Phase 1: Generate thinking process.
        
        Args:
            dialogues: List of dialogue histories
            desc: Description for progress bar
            
        Returns:
            List of Phase 1 results containing generated thinking
        """
        results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc=f"{desc} (Thinking)"
        )
        
        # Extract thinking from each result
        for result in results:
            thinking = self.llm_interface.extract_thinking(result.get("generated_text", ""))
            result["thinking"] = thinking
        
        return results
    
    def _run_phase2(self, dialogues: List[List[Dict]], desc: str = "Phase 2") -> List[Dict]:
        """
        Run Phase 2: Extract Yes/No probability distribution.
        
        Args:
            dialogues: List of dialogue histories with assistant continuation
            desc: Description for progress bar
            
        Returns:
            List of Phase 2 results containing probabilities
        """
        results = self.llm_interface.chat_with_continuation(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=1,  # Only need one token for Yes/No
            seed=self.seed,
            logprobs=self.logprobs,
            desc=f"{desc} (Logprob)"
        )
        
        return results
    
    def _print_debug_dialogues(self, dialogues: List[List[Dict]], step_name: str, max_count: int = 5):
        """
        Print debug information about dialogues.
        
        Shows the final dialogue history that will be sent to the LLM, including all messages
        and the complete content being passed to the model.
        
        Args:
            dialogues: List of dialogue histories (final form sent to LLM)
            step_name: Name of the step for the debug output
            max_count: Maximum number of dialogues to print (default: 5)
        """
        if not self.debug:
            return
        
        print(f"\n[DEBUG {step_name}] Final Dialogue Histories (First {min(max_count, len(dialogues))} dialogues sent to LLM):")
        for i in range(min(max_count, len(dialogues))):
            print(f"\n--- Dialogue {i+1} ---")
            for turn_idx, msg in enumerate(dialogues[i]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                print(f"\n  Message {turn_idx+1} - Role: {role.upper()}")
                print(f"  Content length: {len(content)} characters")
                
                # Truncate long content for readability but show more than before
                if len(content) > 2048:
                    print(f"  Content (first 2048 chars):\n")
                    print(f"  {content[:2048]}...")
                    print(f"  ...[middle part truncated]...")
                    print(f"  ...{content[-512:]}")
                else:
                    print(f"  Content:\n{content}")
    
    def _print_debug_phase2_logprobs(self, results: List[Dict], step_name: str, max_count: int = 5):
        """
        Print debug information about Phase 2 logprob results.
        
        Shows the raw, unprocessed logprobs returned directly from vLLM's chat interface,
        including all tokens and their logprob values.
        
        Args:
            results: List of Phase 2 results containing logprobs and probabilities
            step_name: Name of the step for the debug output
            max_count: Maximum number of results to print (default: 5)
        """
        if not self.debug:
            return
        
        print(f"\n[DEBUG {step_name}] Phase 2 Logprob Information (first {min(max_count, len(results))} results):")
        for i in range(min(max_count, len(results))):
            result = results[i]
            print(f"\n--- Result {i+1} ---")
            
            print(f"  Generated text: {result.get('generated_text', '')}")
            print(f"  Finish reason: {result.get('finish_reason', 'N/A')}")
            
            # Print complete raw vLLM logprobs (unprocessed from vLLM)
            logprobs_raw_vllm = result.get("logprobs_raw_vllm", {})
            if logprobs_raw_vllm:
                print(f"\n  Raw Logprobs from vLLM (all tokens in top-K):")
                for token_id, logprob_obj in logprobs_raw_vllm.items():
                    token_str = getattr(logprob_obj, 'decoded_token', 'N/A')
                    logprob_val = getattr(logprob_obj, 'logprob', float('-inf'))
                    print(f"    Token ID {token_id}: '{token_str}' -> logprob={logprob_val:.6f}")
            else:
                print(f"  Raw vLLM logprobs: (empty or not available)")
            
            # Print processed Yes/No logprobs (for reference)
            logprobs_raw = result.get("logprobs_raw", {})
            if logprobs_raw:
                print(f"\n  Extracted Yes/No Logprobs:")
                for choice, logprob in logprobs_raw.items():
                    print(f"    {choice}: {logprob:.6f}")
            
            # Print computed probabilities
            probabilities = result.get("probabilities", {})
            if probabilities:
                print(f"\n  Computed Probabilities:")
                for choice, prob in probabilities.items():
                    print(f"    {choice}: {prob:.6f}")
    
    def _build_phase2_dialogues(self, 
                                 phase1_results: List[Dict], 
                                 phase1_dialogues: List[List[Dict]], 
                                 phase2_template: str) -> List[List[Dict]]:
        """
        Build Phase 2 dialogues by injecting thinking into assistant message.
        
        Args:
            phase1_results: Results from Phase 1 containing thinking
            phase1_dialogues: Original Phase 1 dialogues
            phase2_template: Template for Phase 2 suffix
            
        Returns:
            List of dialogues ready for Phase 2
        """
        phase2_dialogues = []
        
        for i, result in enumerate(phase1_results):
            thinking = result.get("thinking", "")
            
            # Create assistant message with thinking and suffix
            assistant_content = instantiate_prompt(
                phase2_template,
                THINKING_PROCESS=thinking
            )
            
            # Build Phase 2 dialogue
            dialogue = phase1_dialogues[i].copy()
            dialogue.append({"role": "assistant", "content": assistant_content})
            
            phase2_dialogues.append(dialogue)
        
        return phase2_dialogues
    
    def run_step1(self, personas: List[str], unique_proposals: List[Tuple[str, str]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 1: First-order Belief (persona's own opinion on policy).
        
        This runs once per unique (persona, proposal) pair using two phases:
        - Phase 1: Extract thinking process
        - Phase 2: Extract Yes/No probability distribution
        
        Args:
            personas: List of personas to test
            unique_proposals: List of (category, proposal) tuples
            
        Returns:
            Tuple of (results with probabilities, metadata)
        """
        print("\n=== Step 1: First-order Belief ===")
        
        # Build Phase 1 dialogues
        phase1_dialogues = []
        step1_metadata = []
        
        for persona in personas:
            for category, proposal in unique_proposals:
                persona_prompt = self.get_persona_prompt(persona)
                
                user_prompt = instantiate_prompt(
                    self.prompt_templates.get("step1_phase1", ""),
                    POLICY_PROPOSAL=proposal,
                    PERSONA_INJECTION=persona_prompt
                )
                dialogue = [{"role": "user", "content": user_prompt}]
                phase1_dialogues.append(dialogue)
                
                step1_metadata.append({
                    "persona": persona,
                    "category": category,
                    "proposal": proposal,
                })
        
        print(f"Processing {len(phase1_dialogues)} unique (persona, proposal) pairs...")
        
        # Phase 1: Generate thinking
        phase1_results = self._run_phase1(phase1_dialogues, "Step 1")
        self._save_step_results("step1_phase1", phase1_results, step1_metadata)
        
        # Phase 2: Extract probabilities
        phase2_dialogues = self._build_phase2_dialogues(
            phase1_results, 
            phase1_dialogues, 
            self.prompt_templates.get("step1_phase2", "")
        )
        self._print_debug_dialogues(phase2_dialogues, "Step 1 Phase 2")
        phase2_results = self._run_phase2(phase2_dialogues, "Step 1")
        
        # Combine results
        final_results = []
        for i in range(len(phase1_results)):
            final_results.append({
                # "thinking": phase1_results[i].get("thinking", ""),
                "probabilities": phase2_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": phase2_results[i].get("logprobs_raw", {}),
            })
        
        self._print_debug_phase2_logprobs(phase2_results, "Step 1 Phase 2")
        self._save_step_results("step1_phase2", final_results, step1_metadata)
        
        return final_results, step1_metadata
    
    def run_step2(self, step1_metadata: List[Dict]) -> List[Dict]:
        """
        Run Step 2: Second-order Belief (prediction of population opinion).
        
        This uses a single-stage approach (mimicking verbalize), where the LLM is asked
        to directly output a JSON response with thinking and the predicted percentage.
        
        Args:
            step1_metadata: Metadata from Step 1 (unique persona-proposal pairs)
            
        Returns:
            List of Step 2 results
        """
        print("\n=== Step 2: Second-order Belief (Population Prediction) ===")
        
        # Build single-stage dialogues
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
        
        # Print debug information about dialogues
        self._print_debug_dialogues(dialogues, "Step 2")
        
        # Single-stage: Generate response directly with JSON format
        step2_results = self.llm_interface.chat(
            dialogue_history=dialogues,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            desc="Step 2 (Second-order Belief)"
        )
        
        # Extract percentages from JSON responses and add to results
        for i in range(len(step2_results)):
            response_text = step2_results[i].get("generated_text", "")
            predicted_pct = self._extract_percentage_from_response(response_text)
            step2_results[i]["predicted_percentage"] = predicted_pct
        
        # Save intermediate results (now includes predicted_percentage)
        self._save_step_results("step2", step2_results, step1_metadata)
        
        # Extract final results with predicted percentages
        final_results = []
        for i in range(len(step2_results)):
            final_results.append({
                "predicted_percentage": step2_results[i].get("predicted_percentage"),
                "raw_response": step2_results[i].get("generated_text", ""),
            })
        
        return final_results
    
    def _extract_percentage(self, text: str) -> Optional[int]:
        """
        Extract percentage from response text.
        
        Args:
            text: Response text
            
        Returns:
            Extracted percentage (0-100) or None
        """
        if not text:
            return None
        
        # Try to find a number in the text
        numbers = re.findall(r'\d+', text)
        if numbers:
            pct = int(numbers[0])
            return max(0, min(100, pct))
        
        return None
    
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

    def _compute_step1_yes_ratio(self, step1_results: List[Dict], step1_metadata: List[Dict]) -> Dict[str, float]:
        """
        Compute the frequency of Yes responses for each proposal from Step 1 results.
        
        Instead of averaging raw probabilities, this method converts each probability 
        to a binary decision (Yes if prob >= 0.5) and calculates the ratio of Yes decisions.

        Args:
            step1_results: List of Step 1 response results with probabilities
            step1_metadata: Metadata list for Step 1 (unique persona-proposal pairs)
            
        Returns:
            Dictionary mapping proposal to the frequency of Yes decisions (0.0 to 1.0)
        """
        proposal_decisions = defaultdict(list)
        
        for i, result in enumerate(step1_results):
            probs = result.get("probabilities", {"Yes": 0.5, "No": 0.5})
            yes_prob = probs.get("Yes", 0.5)
            
            # Convert probability to a binary integer (1 for Yes, 0 for No)
            # We treat probability >= 0.5 as a "Yes" decision
            is_yes = 1 if yes_prob >= 0.5 else 0
            
            proposal = step1_metadata[i]["proposal"]
            proposal_decisions[proposal].append(is_yes)
        
        # Compute frequency (ratio of Yes decisions)
        proposal_ratio = {}
        for proposal, decisions in proposal_decisions.items():
            # Sum of 1s divided by total count gives the frequency
            proposal_ratio[proposal] = sum(decisions) / len(decisions) if decisions else 0.5
        
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
            inferred_pct = max(0, min(100, int(round(inferred_percentage))))
            percentages.add(inferred_pct)
        return sorted(list(percentages))
    
    def run_step3(self, step3_base_metadata: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 3: Action Support WITHOUT Distribution Information.
        
        Args:
            step3_base_metadata: Metadata for Step 3 experiments (persona, proposal, action combinations)
            
        Returns:
            Tuple of (step3_results, step3_metadata)
        """
        print("\n=== Step 3: Action Support (No Distribution) ===")
        
        # Build Phase 1 dialogues
        phase1_dialogues = []
        step3_metadata = []
        
        for meta in step3_base_metadata:
            persona_prompt = self.get_persona_prompt(meta['persona'])
            
            user_prompt = instantiate_prompt(
                self.prompt_templates.get("step3_phase1", ""),
                POLICY_PROPOSAL=meta['proposal'],
                PERSONA_INJECTION=persona_prompt,
                CORRESPONDING_ACTION=meta['action']
            )
            dialogue = [{"role": "user", "content": user_prompt}]
            phase1_dialogues.append(dialogue)
            
            step3_metadata.append({
                "persona": meta['persona'],
                "category": meta['category'],
                "proposal": meta['proposal'],
                "action_type": meta['action_type'],
                "action": meta['action'],
            })
        
        print(f"Processing {len(phase1_dialogues)} dialogues...")
        
        # Phase 1: Generate thinking
        phase1_results = self._run_phase1(phase1_dialogues, "Step 3")
        self._save_step_results("step3_phase1", phase1_results, step3_metadata)
        
        # Phase 2: Extract probabilities
        phase2_dialogues = self._build_phase2_dialogues(
            phase1_results,
            phase1_dialogues,
            self.prompt_templates.get("step3_phase2", "")
        )
        self._print_debug_dialogues(phase2_dialogues, "Step 3 Phase 2")
        phase2_results = self._run_phase2(phase2_dialogues, "Step 3")
        
        # Combine results
        final_results = []
        for i in range(len(phase1_results)):
            final_results.append({
                # "thinking": phase1_results[i].get("thinking", ""),
                "probabilities": phase2_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": phase2_results[i].get("logprobs_raw", {}),
            })
        
        self._print_debug_phase2_logprobs(phase2_results, "Step 3 Phase 2")
        self._save_step_results("step3_phase2", final_results, step3_metadata)
        
        return final_results, step3_metadata
    
    def run_step4a(self, step4_base_metadata: List[Dict], step1_yes_ratio: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 4a: First-order Belief with Distribution Information.
        
        Args:
            step4_base_metadata: Metadata for Step 4 experiments
            step1_yes_ratio: Dictionary mapping proposal to Yes ratio from Step 1
            
        Returns:
            Tuple of (step4a_results, step4a_metadata)
        """
        print("\n=== Step 4a: First-order Belief with Distribution ===")
        
        # Build Phase 1 dialogues
        phase1_dialogues = []
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
                    self.prompt_templates.get("step4a_phase1", ""),
                    POLICY_PROPOSAL=proposal,
                    PERSONA_INJECTION=persona_prompt,
                    DISTRIBUTION=distribution_text
                )
                dialogue = [{"role": "user", "content": user_prompt}]
                phase1_dialogues.append(dialogue)
                
                step4a_metadata.append({
                    **meta,
                    "distribution_percentage": percentage,
                    "distribution_text": distribution_text,
                    "is_inferred": percentage == inferred_percentage,
                })
        
        print(f"Processing {len(phase1_dialogues)} dialogues...")
        
        # Phase 1: Generate thinking
        phase1_results = self._run_phase1(phase1_dialogues, "Step 4a")
        self._save_step_results("step4a_phase1", phase1_results, step4a_metadata)
        
        # Phase 2: Extract probabilities
        phase2_dialogues = self._build_phase2_dialogues(
            phase1_results,
            phase1_dialogues,
            self.prompt_templates.get("step4a_phase2", "")
        )
        self._print_debug_dialogues(phase2_dialogues, "Step 4a Phase 2")
        phase2_results = self._run_phase2(phase2_dialogues, "Step 4a")
        
        # Combine results
        final_results = []
        for i in range(len(phase1_results)):
            final_results.append({
                # "thinking": phase1_results[i].get("thinking", ""),
                "probabilities": phase2_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": phase2_results[i].get("logprobs_raw", {}),
            })
        
        self._print_debug_phase2_logprobs(phase2_results, "Step 4a Phase 2")
        self._save_step_results("step4a_phase2", final_results, step4a_metadata)
        
        return final_results, step4a_metadata
    
    def run_step4b(self, step4_base_metadata: List[Dict], step1_yes_ratio: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run Step 4b: Action Support with Distribution Information.
        
        Args:
            step4_base_metadata: Metadata for Step 4 experiments
            step1_yes_ratio: Dictionary mapping proposal to Yes ratio from Step 1
            
        Returns:
            Tuple of (step4b_results, step4b_metadata)
        """
        print("\n=== Step 4b: Action Support with Distribution ===")
        
        # Build Phase 1 dialogues
        phase1_dialogues = []
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
                    self.prompt_templates.get("step4b_phase1", ""),
                    POLICY_PROPOSAL=proposal,
                    PERSONA_INJECTION=persona_prompt,
                    DISTRIBUTION=distribution_text,
                    CORRESPONDING_ACTION=action
                )
                dialogue = [{"role": "user", "content": user_prompt}]
                phase1_dialogues.append(dialogue)
                
                step4b_metadata.append({
                    **meta,
                    "distribution_percentage": percentage,
                    "distribution_text": distribution_text,
                    "is_inferred": percentage == inferred_percentage,
                })
        
        print(f"Processing {len(phase1_dialogues)} dialogues...")
        
        # Phase 1: Generate thinking
        phase1_results = self._run_phase1(phase1_dialogues, "Step 4b")
        self._save_step_results("step4b_phase1", phase1_results, step4b_metadata)
        
        # Phase 2: Extract probabilities
        phase2_dialogues = self._build_phase2_dialogues(
            phase1_results,
            phase1_dialogues,
            self.prompt_templates.get("step4b_phase2", "")
        )
        self._print_debug_dialogues(phase2_dialogues, "Step 4b Phase 2")
        phase2_results = self._run_phase2(phase2_dialogues, "Step 4b")
        
        # Combine results
        final_results = []
        for i in range(len(phase1_results)):
            final_results.append({
                # "thinking": phase1_results[i].get("thinking", ""),
                "probabilities": phase2_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": phase2_results[i].get("logprobs_raw", {}),
            })
        
        self._print_debug_phase2_logprobs(phase2_results, "Step 4b Phase 2")
        self._save_step_results("step4b_phase2", final_results, step4b_metadata)
        
        return final_results, step4b_metadata
    
    def _compile_results(self,
                        step1_results: List[Dict], step1_metadata: List[Dict],
                        step2_results: List[Dict],
                        step3_results: List[Dict], step3_metadata: List[Dict],
                        step4a_results: List[Dict], step4a_metadata: List[Dict],
                        step4b_results: List[Dict], step4b_metadata: List[Dict],
                        step_base_metadata: List[Dict]) -> List[Dict]:
        """
        Compile all step results into final experiment results.
        
        Results contain probability distributions instead of single answers.
        
        Args:
            step1_results: Results from Step 1
            step1_metadata: Metadata for Step 1
            step2_results: Results from Step 2
            step3_results: Results from Step 3
            step3_metadata: Metadata for Step 3
            step4a_results: Results from Step 4a
            step4a_metadata: Metadata for Step 4a
            step4b_results: Results from Step 4b
            step4b_metadata: Metadata for Step 4b
            step_base_metadata: Base metadata for Step 3/4
            
        Returns:
            List of compiled experiment results
        """
        compiled_results = []
        
        # Create lookup for Step 1 and Step 2 results by (persona, proposal)
        step1_lookup = {}
        step2_lookup = {}
        for i, meta in enumerate(step1_metadata):
            key = (meta['persona'], meta['proposal'])
            step1_lookup[key] = {
                "probabilities": step1_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": step1_results[i].get("logprobs_raw", {}),
            }
            step2_lookup[key] = {
                "predicted_percentage": step2_results[i].get("predicted_percentage"),
            }
        
        # Create lookup for Step 3 results by (persona, proposal, action)
        step3_lookup = {}
        for i, meta in enumerate(step3_metadata):
            key = (meta['persona'], meta['proposal'], meta['action'])
            step3_lookup[key] = {
                "probabilities": step3_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": step3_results[i].get("logprobs_raw", {}),
            }
        
        # Create lookup for Step 4a and Step 4b results
        step4a_lookup = {}
        for i, meta in enumerate(step4a_metadata):
            key = (meta['persona'], meta['proposal'], meta['action'], meta['distribution_percentage'])
            step4a_lookup[key] = {
                "probabilities": step4a_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": step4a_results[i].get("logprobs_raw", {}),
                "distribution_text": meta['distribution_text'],
                "is_inferred": meta['is_inferred'],
            }
        
        step4b_lookup = {}
        for i, meta in enumerate(step4b_metadata):
            key = (meta['persona'], meta['proposal'], meta['action'], meta['distribution_percentage'])
            step4b_lookup[key] = {
                "probabilities": step4b_results[i].get("probabilities", {"Yes": 0.5, "No": 0.5}),
                "logprobs_raw": step4b_results[i].get("logprobs_raw", {}),
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
            step1_result = step1_lookup.get(step12_key, {"probabilities": {"Yes": 0.5, "No": 0.5}})
            step2_result = step2_lookup.get(step12_key, {"predicted_percentage": None})
            
            # Get Step 3 result
            step3_key = (persona, proposal, action)
            step3_result = step3_lookup.get(step3_key, {"probabilities": {"Yes": 0.5, "No": 0.5}})
            
            # Collect Step 4a results for all distributions
            step4a_by_distribution = {}
            for key, data in step4a_lookup.items():
                if key[0] == persona and key[1] == proposal and key[2] == action:
                    percentage = key[3]
                    step4a_by_distribution[percentage] = {
                        "probabilities": data['probabilities'],
                        "distribution_text": data['distribution_text'],
                        "is_inferred": data['is_inferred']
                    }
            
            # Collect Step 4b results for all distributions
            step4b_by_distribution = {}
            for key, data in step4b_lookup.items():
                if key[0] == persona and key[1] == proposal and key[2] == action:
                    percentage = key[3]
                    step4b_by_distribution[percentage] = {
                        "probabilities": data['probabilities'],
                        "distribution_text": data['distribution_text'],
                        "is_inferred": data['is_inferred']
                    }
            
            final_result = {
                "experiment_id": self.experiment_id,
                "model": self.model_name,
                
                # Core metadata
                "persona": persona,
                "category": meta['category'],
                "policy_proposal": proposal,
                "action_type": meta['action_type'],
                "corresponding_action": action,
                
                # Step results (probability distributions)
                "step1_first_order_belief": step1_result,
                "step2_second_order_belief": step2_result,
                "step3_action_support_no_distribution": step3_result,
                "step4a_first_order_with_distribution": step4a_by_distribution,
                "step4b_action_support_with_distribution": step4b_by_distribution
            }
            compiled_results.append(final_result)
        
        return compiled_results

    def run_experiments_from_step(self, resume_step: str):
        """
        Resume experiment execution from a specified step.
        
        This is a generalized method that can resume from any step: step1, step2, step3, step4a, or step4b.
        It automatically loads required dependencies and runs all subsequent steps.
        
        Args:
            resume_step: The step to resume from ('step1', 'step2', 'step3', 'step4a', 'step4b')
            
        Raises:
            ValueError: If not resuming from a checkpoint or if resume_step is invalid
            RuntimeError: If required checkpoint files are missing
        """
        if not self.is_resuming:
            raise ValueError("run_experiments_from_step() requires resuming from a checkpoint")
        
        # Validate resume_step
        valid_steps = ["step1", "step2", "step3", "step4a", "step4b"]
        if resume_step not in valid_steps:
            raise ValueError(f"Invalid resume_step '{resume_step}'. Must be one of: {', '.join(valid_steps)}")
        
        if not self._check_resume_feasibility(resume_step):
            raise RuntimeError(f"Cannot resume from {resume_step}: Missing required checkpoint files")
        
        print("\n" + "="*80)
        print(f"RESUMING EXPERIMENTS FROM {resume_step.upper()}")
        print("="*80)
        
        # Handle debug mode
        if self.debug:
            print("\n[DEBUG] Debug mode enabled for resumption.")
            print("[DEBUG] Will process limited subset of data for testing.")
        
        # Load prompt templates
        print("\nLoading prompt templates...")
        self.load_prompt_templates()
        
        # Initialize LLM
        print(f"Initializing LLM: {self.model_name}...")
        self.initialize_llm()
        
        # Initialize variables
        step1_results = None
        step1_metadata = None
        step2_results = None
        step3_results = None
        step3_metadata = None
        step4a_results = None
        step4a_metadata = None
        step4b_results = None
        step4b_metadata = None
        step_base_metadata = None
        step1_yes_ratio = None
        debug_personas = None
        debug_proposals = None

        def _load_step2_results_aligned(current_step1_metadata: List[Dict]) -> Optional[List[Dict]]:
            """Load Step 2 checkpoint results aligned to current Step 1 metadata order."""
            step2_results_full, step2_metadata = self._load_step_results("step2")
            if step2_results_full is None:
                return None

            # Fallback to original behavior when metadata is missing
            if not current_step1_metadata or not step2_metadata:
                aligned = []
                for result in step2_results_full:
                    aligned.append({
                        "predicted_percentage": result.get("predicted_percentage"),
                        "raw_response": result.get("generated_text", ""),
                    })
                return aligned

            step2_lookup = {}
            for i, meta in enumerate(step2_metadata):
                key = (meta.get("persona"), meta.get("proposal"))
                step2_lookup[key] = step2_results_full[i]

            aligned = []
            for meta in current_step1_metadata:
                key = (meta.get("persona"), meta.get("proposal"))
                result = step2_lookup.get(key, {})
                aligned.append({
                    "predicted_percentage": result.get("predicted_percentage"),
                    "raw_response": result.get("generated_text", ""),
                })

            return aligned
        
        # Load and prepare data based on resume_step
        if resume_step in ["step2", "step3", "step4a", "step4b"]:
            # Load Step 1 results
            print("\nLoading Step 1 results...")
            step1_results, step1_metadata = self._load_step_results("step1_phase2")
            if step1_results is None:
                raise RuntimeError("Failed to load Step 1 Phase 2 results")
            
            # Apply debug mode restrictions if needed
            if self.debug:
                print("\n[DEBUG] Limiting Step 1 data for debugging:")
                # Use the same debug selection logic as run_experiments()
                all_personas = self.data_loader.get_personas(include_none=True)
                candidate_personas = [all_personas[0], all_personas[1], all_personas[33]] if len(all_personas) > 33 else all_personas[:3]

                all_unique_proposals = self.data_loader.get_unique_proposals()
                candidate_proposals = [proposal for _, proposal in all_unique_proposals[:2]]

                available_personas = {meta['persona'] for meta in step1_metadata}
                available_proposals = {meta['proposal'] for meta in step1_metadata}

                selected_personas = [p for p in candidate_personas if p in available_personas]
                selected_proposals = [p for p in candidate_proposals if p in available_proposals]

                # Fallback when checkpoint data does not fully overlap with default debug subset
                if not selected_personas:
                    selected_personas = list(dict.fromkeys(meta['persona'] for meta in step1_metadata))[:3]
                if not selected_proposals:
                    selected_proposals = list(dict.fromkeys(meta['proposal'] for meta in step1_metadata))[:2]

                debug_personas = set(selected_personas)
                debug_proposals = set(selected_proposals)

                filtered_indices = [
                    i for i, meta in enumerate(step1_metadata)
                    if meta['persona'] in debug_personas and meta['proposal'] in debug_proposals
                ]
                step1_results = [step1_results[i] for i in filtered_indices]
                step1_metadata = [step1_metadata[i] for i in filtered_indices]

                print(
                    f"[DEBUG] Using {len(debug_personas)} personas x {len(debug_proposals)} proposals "
                    f"-> {len(step1_metadata)} Step 1 pairs"
                )
        
        if resume_step in ["step3", "step4a", "step4b"]:
            # Compute step1_yes_ratio for distribution inference
            print("Computing Step 1 Yes ratios...")
            step1_yes_ratio = self._compute_step1_yes_ratio(step1_results, step1_metadata)
            print(f"✓ Computed Yes ratios for {len(step1_yes_ratio)} proposals")
        
        if resume_step in ["step4a", "step4b"]:
            # Load Step 3 Phase 2 metadata to reconstruct step_base_metadata
            print("Loading Step 3 metadata...")
            _, step3_metadata = self._load_step_results("step3_phase2")
            if step3_metadata is None:
                raise RuntimeError("Failed to load Step 3 Phase 2 metadata")
            step_base_metadata = step3_metadata

            if self.debug and debug_personas is not None and debug_proposals is not None:
                original_len = len(step_base_metadata)
                step_base_metadata = [
                    meta for meta in step_base_metadata
                    if meta['persona'] in debug_personas and meta['proposal'] in debug_proposals
                ]
                print(
                    f"[DEBUG] Limited Step 3/4 base metadata from {original_len} "
                    f"to {len(step_base_metadata)} dialogues"
                )
        
        # Execute steps from resume_step onwards
        steps_to_run = []
        if resume_step == "step1":
            steps_to_run = ["step1", "step2", "step3", "step4a", "step4b"]
        elif resume_step == "step2":
            steps_to_run = ["step2", "step3", "step4a", "step4b"]
        elif resume_step == "step3":
            steps_to_run = ["step3", "step4a", "step4b"]
        elif resume_step == "step4a":
            steps_to_run = ["step4a", "step4b"]
        elif resume_step == "step4b":
            steps_to_run = ["step4b"]
        
        # Step 2: Run if needed
        if "step2" in steps_to_run:
            print("\nStarting Step 2...")
            step2_results = self.run_step2(step1_metadata)
        else:
            # Load existing Step 2 results if not running it
            if step1_metadata is not None:
                print("Loading Step 2 results...")
                step2_results = _load_step2_results_aligned(step1_metadata)
        
        # Step 3: Run if needed
        if "step3" in steps_to_run:
            # Build step_base_metadata if not already loaded
            if step_base_metadata is None:
                print("Building Step 3/4 base metadata from Step 1...")
                # Extract personas and proposals from step1_metadata
                personas_set = set()
                proposals_set = set()
                for meta in step1_metadata:
                    personas_set.add(meta['persona'])
                    proposals_set.add((meta['category'], meta['proposal']))
                
                personas = list(personas_set)
                unique_proposals = list(proposals_set)
                
                # Build step_base_metadata: (persona, proposal, action) combinations
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
            
            print("\nStarting Step 3...")
            step3_results, step3_metadata = self.run_step3(step_base_metadata)
        else:
            # Load existing Step 3 results if not running it
            print("Loading Step 3 results...")
            step3_results, step3_metadata = self._load_step_results("step3_phase2")

            if self.debug and step3_results is not None and step3_metadata is not None and debug_personas is not None and debug_proposals is not None:
                filtered_indices = [
                    i for i, meta in enumerate(step3_metadata)
                    if meta['persona'] in debug_personas and meta['proposal'] in debug_proposals
                ]
                step3_results = [step3_results[i] for i in filtered_indices]
                step3_metadata = [step3_metadata[i] for i in filtered_indices]
                print(f"[DEBUG] Limited Step 3 checkpoint results to {len(step3_metadata)} dialogues")
            
            # Build step_base_metadata if not already loaded
            if step_base_metadata is None and step3_metadata is not None:
                step_base_metadata = step3_metadata
        
        # Step 4a and 4b: Run if needed
        if "step4a" in steps_to_run or "step4b" in steps_to_run:
            if "step4a" in steps_to_run:
                print("\nStarting Step 4a...")
                step4a_results, step4a_metadata = self.run_step4a(step_base_metadata, step1_yes_ratio)
            else:
                # Load existing Step 4a results if not running it
                print("Loading Step 4a results...")
                step4a_results, _ = self._load_step_results("step4a_phase2")
                _, step4a_metadata = self._load_step_results("step4a_phase2")
            
            if "step4b" in steps_to_run:
                print("\nStarting Step 4b...")
                step4b_results, step4b_metadata = self.run_step4b(step_base_metadata, step1_yes_ratio)
            else:
                # Load existing Step 4b results if not running it
                print("Loading Step 4b results...")
                step4b_results, _ = self._load_step_results("step4b_phase2")
                _, step4b_metadata = self._load_step_results("step4b_phase2")
        
        # Compile results
        print("\n=== Compiling Final Results ===")
        
        # Ensure all required data is loaded for compilation
        if step1_results is None:
            print("Loading Step 1 results...")
            step1_results, step1_metadata = self._load_step_results("step1_phase2")
        
        if step2_results is None and step1_metadata is not None:
            print("Loading Step 2 results...")
            step2_results = _load_step2_results_aligned(step1_metadata)
        
        if step3_results is None:
            print("Loading Step 3 results...")
            step3_results, step3_metadata = self._load_step_results("step3_phase2")
        
        if step4a_results is None:
            print("Loading Step 4a results...")
            step4a_results, step4a_metadata = self._load_step_results("step4a_phase2")
        
        if step4b_results is None:
            print("Loading Step 4b results...")
            step4b_results, step4b_metadata = self._load_step_results("step4b_phase2")
        
        if step_base_metadata is None:
            print("Loading Step 3 metadata for base metadata...")
            _, step3_metadata = self._load_step_results("step3_phase2")
            step_base_metadata = step3_metadata
        
        # Compile all results
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
        print(f"\n✓ Completed {len(self.results)} experiments")
    
    def run_experiments(self, personas: List[str] = None,
                       unique_proposals: List[Tuple[str, str]] = None,
                       max_experiments: int = None):
        """
        Run logprob belief experiments.
        
        This method orchestrates all steps using the two-phase approach:
        - Step 1: First-order belief (persona's own opinion on policy)
        - Step 2: Second-order belief (prediction of population opinion)
        - Step 3: Action support WITHOUT distribution information
        - Step 4a: First-order belief with distribution information
        - Step 4b: Action support WITH distribution information
        
        Args:
            personas: List of personas to test (None = all)
            unique_proposals: List of (category, proposal) tuples (None = all)
            max_experiments: Maximum number of base experiments (None = all)
        """
        # Handle debug mode
        if self.debug:
            print("Debug mode enabled: Limiting to 2 proposals with ALL actions.")
            all_unique_proposals = self.data_loader.get_unique_proposals()
            unique_proposals = all_unique_proposals[:2]
            print(f"Debug: Using {len(unique_proposals)} unique proposals")
            max_experiments = None

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
            personas = [personas[0], personas[1], personas[33]] if len(personas) > 33 else personas[:3]
        
        print(f"Testing {len(personas)} personas...")
        
        # Get unique proposals for Step 1 and Step 2
        if unique_proposals is None:
            unique_proposals = self.data_loader.get_unique_proposals()
        
        print(f"Testing {len(unique_proposals)} unique proposals...")
        
        # Calculate Step 1/2 experiments (no redundancy)
        step12_count = len(personas) * len(unique_proposals)
        print(f"Total Step 1/2 experiments (unique persona-proposal pairs): {step12_count}")
        
        # Run Step 1: First-order Belief
        step1_results, step1_metadata = self.run_step1(personas, unique_proposals)
        
        # Run Step 2: Second-order Belief
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
        
        # Limit to max_experiments if specified
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
    """Main entry point for the logprob experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run political belief experiments with LLM (Logprob-based)"
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name (must be compatible with vLLM)")
    
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
                        help="Maximum tokens to generate for thinking phase")
    parser.add_argument("--logprobs", type=int, default=20,
                        help="Number of top logprobs to extract")
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
    parser.add_argument("--prompt-type", type=str, default="logprob",
                        choices=["logprob", "verbalize"],
                        help="Type of prompts to use")
    parser.add_argument("--prompts-dir", type=str, default=None,
                        help="Custom prompts directory (overrides prompt-type)")
    
    # Resume configuration
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint (experiment prefix, e.g., 'meta-llama_Llama-3.1-8B-Instruct_20260303_095200')")
    parser.add_argument("--resume-step", type=str, default="step4a", 
                        choices=["step1", "step2", "step3", "step4a", "step4b"],
                        help="Which step to resume from (default: step4a). Supports: step1, step2, step3, step4a, step4b")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List available checkpoints and exit")
    
    cmd = ["--debug"]
    args = parser.parse_args(cmd)
    
    # Create experiment runner
    runner = LogprobExperimentRunner(
        model_name=args.model,
        results_dir=args.results_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        seed=args.seed,
        debug=args.debug,
        use_api=args.use_api,
        prompts_dir=args.prompts_dir,
        prompt_type=args.prompt_type,
        resume_from_checkpoint=args.resume_from
    )
    
    # Handle checkpoint listing
    if args.list_checkpoints:
        print("\n=== Available Checkpoints ===")
        checkpoints = runner._list_available_checkpoints()
        for step_name, exists in checkpoints.items():
            status = "✓" if exists else "✗"
            print(f"  [{status}] {step_name}")
        return
    
    # Handle resumption
    if args.resume_from:
        print(f"\n=== Resuming from checkpoint: {args.resume_from} ===")
        print(f"Resume step: {args.resume_step}")
        try:
            runner.run_experiments_from_step(args.resume_step)
        except Exception as e:
            print(f"Error during resumption: {e}")
            import traceback
            traceback.print_exc()
        finally:
            runner.cleanup()
        print("Resumption completed!")
        return
    
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
