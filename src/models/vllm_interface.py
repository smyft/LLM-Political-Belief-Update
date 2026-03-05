"""
vLLM Interface Module.

This module implements a VLLMInterface class for interacting with LLMs via vLLM.
It supports loading models locally, generating responses, and extracting logprob information.
"""

import gc
import re
from typing import List, Dict, Optional, Any, Union

import numpy as np
from tqdm import tqdm

# vLLM imports - these will only work if vllm is installed
try:
    import torch
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. VLLMInterface will not be available.")


def free_gpu_memory(llm_instance=None):
    """
    Aggressively cleans up GPU memory to allow loading the next model.
    vLLM is persistent, so we must delete the object and run garbage collection.
    """
    print("Cleaning up GPU memory...")

    if llm_instance:
        del llm_instance

    # Python garbage collection
    gc.collect()

    # PyTorch CUDA cache cleanup
    if VLLM_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("GPU memory cleaned.")


def extract_thinking_process(text: str) -> str:
    """
    Extracts the thinking process from the Model's output.
    
    Handles:
    - Explicit <think>...</think> tags
    - Answer filtering from beginning, middle, and end of text
    
    Args:
        text: The raw model output
        
    Returns:
        The extracted thinking process
    """
    THINK_START_TAG = "<think>"
    THINK_END_TAG = "</think>"
    
    # Strategy 1: Explicit Tags - Check for complete pair: <think> ...Content... </think>
    complete_pattern = re.compile(
        f"{re.escape(THINK_START_TAG)}(.*?){re.escape(THINK_END_TAG)}", 
        re.DOTALL | re.IGNORECASE
    )
    match_complete = complete_pattern.search(text)
    if match_complete:
        return match_complete.group(1).strip()
    
    # Strategy 2: Check for unclosed tag: <think> ...Content (End of Text)
    incomplete_pattern = re.compile(
        f"{re.escape(THINK_START_TAG)}(.*)$",
        re.DOTALL | re.IGNORECASE
    )
    match_incomplete = incomplete_pattern.search(text)
    if match_incomplete:
        return match_incomplete.group(1).strip()
    
    # Strategy 3: No tags - return the original text with answer filtering
    return filter_answer_from_text(text)


def filter_answer_from_text(text: str) -> str:
    """
    Filter out answer-related content from the text.
    This helps extract pure reasoning without the final answer.
    
    Args:
        text: The raw text
        
    Returns:
        Filtered text without answer statements
    """
    # Remove explicit answer patterns at the end
    answer_patterns = [
        # "Yes" or "No" at the end
        r'\s*(?:So,?\s+)?(?:the\s+)?(?:final\s+)?answer(?:\s+is)?(?:\s*:)?\s*(?:Yes|No)\.?\s*$',
        # JSON-like answer at the end
        r'\s*\{[^}]*"answer"\s*:\s*"(?:Yes|No)"[^}]*\}\s*$',
        # "My answer is Yes/No"
        r'\s*(?:My\s+)?answer(?:\s+is)?(?:\s*:)?\s*(?:Yes|No)\.?\s*$',
        # "Therefore, Yes/No"
        r'\s*(?:Therefore|Thus|Hence),?\s*(?:Yes|No)\.?\s*$',
    ]
    
    result = text.strip()
    for pattern in answer_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    return result.strip()


def extract_yes_no_logprobs(top_logprobs: Dict[int, Any]) -> Dict[str, float]:
    """
    Extracts logprobs for Yes and No from vLLM's Logprob objects.
    
    Args:
        top_logprobs: Dictionary mapping {token_id: LogprobStruct}.
                      The LogprobStruct contains .logprob and .decoded_token
                      
    Returns:
        Dictionary with "Yes" and "No" logprobs
    """
    choice_logprobs_list = {"Yes": [], "No": []}
    
    # Iterate through the top-K tokens returned by vLLM
    for token_id, logprob_obj in top_logprobs.items():
        # Access the decoded string provided by vLLM
        token_str = logprob_obj.decoded_token
        
        if token_str:
            clean_str = token_str.strip().lower()
            
            # Check for Yes/No tokens (case insensitive)
            if clean_str in ["yes", "y"]:
                choice_logprobs_list["Yes"].append(logprob_obj.logprob)
            elif clean_str in ["no", "n"]:
                choice_logprobs_list["No"].append(logprob_obj.logprob)
    
    choice_scores = {}
    
    for choice, logprobs in choice_logprobs_list.items():
        if not logprobs:
            # Set unseen choices to a very small number (effectively 0 probability)
            choice_scores[choice] = -999.0
        else:
            # Sum logprobs if multiple tokens map to same choice (e.g., "Yes" and "yes")
            choice_scores[choice] = np.logaddexp.reduce(logprobs)
    
    return choice_scores


def compute_yes_no_probabilities(logprobs: Dict[str, float]) -> Dict[str, float]:
    """
    Compute normalized probabilities from Yes/No logprobs.
    
    Args:
        logprobs: Dictionary with "Yes" and "No" logprobs
        
    Returns:
        Dictionary with "Yes" and "No" probabilities that sum to 1
    """
    yes_logprob = logprobs.get("Yes", -999.0)
    no_logprob = logprobs.get("No", -999.0)
    
    # Check for empty/garbage output
    if yes_logprob == -999.0 and no_logprob == -999.0:
        return {"Yes": 0.5, "No": 0.5}
    
    # Apply softmax to get probabilities
    max_logprob = max(yes_logprob, no_logprob)
    yes_exp = np.exp(yes_logprob - max_logprob)
    no_exp = np.exp(no_logprob - max_logprob)
    
    total = yes_exp + no_exp
    
    return {
        "Yes": float(yes_exp / total),
        "No": float(no_exp / total)
    }


class VLLMInterface:
    """
    Interface for interacting with LLMs via vLLM.
    
    This class supports:
    - Loading models locally
    - Generating responses via chat
    - Extracting logprob information for probability belief estimation
    """
    
    def __init__(self, 
                 model_name: str, 
                 gpu_memory_utilization: float = 0.9,
                 tensor_parallel_size: int = 1,
                 trust_remote_code: bool = True,
                 dtype: str = "auto",
                 enforce_eager: bool = True,
                 **kwargs):
        """
        Initialize the vLLM interface.
        
        Args:
            model_name: Name or path of the model to load
            gpu_memory_utilization: Fraction of GPU memory to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            trust_remote_code: Whether to trust remote code for model loading
            dtype: Data type for model weights
            enforce_eager: Whether to use eager mode (recommended for compatibility)
            **kwargs: Additional arguments for vLLM
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.extra_kwargs = kwargs
        
        self.llm = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model into memory."""
        if self.llm is not None:
            print(f"Model {self.model_name} is already loaded.")
            return
        
        print(f"Loading model: {self.model_name}...")
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
            **self.extra_kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded successfully.")

    def _normalize_and_validate_dialogues(
        self,
        dialogue_history: Union[List[Dict], List[List[Dict]]],
        require_assistant_last: bool = False
    ) -> List[List[Dict]]:
        """
        Normalize input to batched OpenAI-style dialogues and validate schema.

        Args:
            dialogue_history: A single dialogue ([{role, content}, ...]) or
                              a batch of dialogues ([[{role, content}, ...], ...])
            require_assistant_last: Whether the final message must be assistant

        Returns:
            Normalized list of dialogues.
        """
        if not isinstance(dialogue_history, list) or not dialogue_history:
            raise TypeError("dialogue_history must be a non-empty list of OpenAI-style messages/dialogues.")

        # Single dialogue: List[Dict]
        if isinstance(dialogue_history[0], dict):
            dialogues = [dialogue_history]
        # Batch dialogue: List[List[Dict]]
        elif isinstance(dialogue_history[0], list):
            dialogues = dialogue_history
        else:
            raise TypeError(
                "dialogue_history must be List[Dict] or List[List[Dict]]. "
                "Passing plain string prompts is not supported."
            )

        for d_idx, dialogue in enumerate(dialogues):
            if not isinstance(dialogue, list) or not dialogue:
                raise TypeError(f"Dialogue at index {d_idx} must be a non-empty list of messages.")

            for m_idx, msg in enumerate(dialogue):
                if not isinstance(msg, dict):
                    raise TypeError(f"Message at dialogue[{d_idx}][{m_idx}] must be a dict.")
                if "role" not in msg or "content" not in msg:
                    raise ValueError(f"Message at dialogue[{d_idx}][{m_idx}] must contain 'role' and 'content'.")
                if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                    raise TypeError(
                        f"'role' and 'content' at dialogue[{d_idx}][{m_idx}] must both be strings."
                    )

            if require_assistant_last and dialogue[-1]["role"].strip().lower() != "assistant":
                raise ValueError(
                    f"Dialogue at index {d_idx} must end with an assistant message when "
                    "using continuation mode."
                )

        return dialogues
    
    def chat(self, 
             dialogue_history: Union[List[Dict], List[List[Dict]]],
             temperature: float = 0,
             max_tokens: int = 1024,
             seed: int = 42,
             logprobs: int = None,
             show_progress: bool = True,
             desc: str = "Processing",
             **kwargs) -> List[Dict]:
        """
        Generate responses using the loaded model.
        
        Args:
            dialogue_history: Single dialogue or list of dialogues (each dialogue is a list of message dicts)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed
            logprobs: Number of top logprobs to return (None for no logprobs)
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            **kwargs: Additional arguments for SamplingParams
            
        Returns:
            List of response dictionaries containing generated_text and optionally logprobs
        """
        if self.llm is None:
            self.load_model()
        
        # Normalize input to list of dialogues and validate OpenAI-style schema
        dialogues = self._normalize_and_validate_dialogues(dialogue_history)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            skip_special_tokens=False,
            **kwargs
        )
        
        # Generate responses
        outputs = self.llm.chat(dialogues, sampling_params)
        
        # Process outputs
        results = []
        iterator = outputs
        if show_progress:
            iterator = tqdm(outputs, desc=desc, unit="response")
        
        for output in iterator:
            result = {
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason if hasattr(output.outputs[0], 'finish_reason') else None
            }
            
            # Include logprobs if requested
            if logprobs and output.outputs[0].logprobs:
                result["logprobs"] = output.outputs[0].logprobs
            
            results.append(result)
        
        return results
    
    def chat_with_continuation(self,
                               dialogue_history: Union[List[Dict], List[List[Dict]]],
                               temperature: float = 0,
                               max_tokens: int = 1,
                               seed: int = 42,
                               logprobs: int = 20,
                               show_progress: bool = True,
                               desc: str = "Processing",
                               **kwargs) -> List[Dict]:
        """
        Generate responses by continuing the last assistant message.
        
        This is used for Phase 2 of the logprob experiment where we want to
        extract the next token probabilities after the thinking process.
        
        Args:
            dialogue_history: List of dialogues where the last message is from assistant
                             and should be continued
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (usually 1 for logprob extraction)
            seed: Random seed
            logprobs: Number of top logprobs to return
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            **kwargs: Additional arguments for SamplingParams
            
        Returns:
            List of response dictionaries containing generated_text, logprobs, and probabilities
        """
        if self.llm is None:
            self.load_model()
        
        # Normalize input to list of dialogues and validate OpenAI-style schema
        dialogues = self._normalize_and_validate_dialogues(
            dialogue_history,
            require_assistant_last=True
        )
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            skip_special_tokens=False,
            **kwargs
        )
        
        # Generate with continuation mode
        try:
            outputs = self.llm.chat(
                dialogues, 
                sampling_params=sampling_params,
                add_generation_prompt=False,
                continue_final_message=True
            )
        except TypeError as e:
            # Fallback for older vLLM versions
            print(f"Warning: vLLM chat with continuation failed: {e}")
            print("Falling back to standard chat mode.")
            outputs = self.llm.chat(dialogues, sampling_params)
        
        # Process outputs
        results = []
        iterator = outputs
        if show_progress:
            iterator = tqdm(outputs, desc=desc, unit="response")
        
        for output in iterator:
            result = {
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason if hasattr(output.outputs[0], 'finish_reason') else None
            }
            
            # Extract logprobs for the first token
            if logprobs and output.outputs[0].logprobs:
                first_token_logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else None
                
                if first_token_logprobs:
                    # Store the raw, unprocessed logprobs from vLLM for debug purposes
                    # This is the complete logprobs dictionary as returned by vLLM
                    result["logprobs_raw_vllm"] = first_token_logprobs
                    
                    # Extract Yes/No logprobs for analysis
                    yes_no_logprobs = extract_yes_no_logprobs(first_token_logprobs)
                    yes_no_probs = compute_yes_no_probabilities(yes_no_logprobs)
                    
                    result["logprobs_raw"] = yes_no_logprobs
                    result["probabilities"] = yes_no_probs
                else:
                    result["logprobs_raw_vllm"] = {}
                    result["logprobs_raw"] = {"Yes": -999.0, "No": -999.0}
                    result["probabilities"] = {"Yes": 0.5, "No": 0.5}
            
            results.append(result)
        
        return results
    
    def extract_thinking(self, response_text: str) -> str:
        """
        Extract the thinking process from a response.
        
        Args:
            response_text: The raw response text
            
        Returns:
            The extracted thinking process
        """
        return extract_thinking_process(response_text)
    
    def free_memory(self):
        """Free GPU memory by deleting the model."""
        free_gpu_memory(self.llm)
        self.llm = None
        self.tokenizer = None


if __name__ == "__main__":
    # Test the vLLM interface (only works if vLLM is installed)
    if VLLM_AVAILABLE:
        print("Testing VLLMInterface...")
        
        # Test with a small model
        model_name = "Qwen/Qwen3-0.6B"
        
        try:
            interface = VLLMInterface(model_name)
            interface.load_model()
            
            # Test chat
            dialogues = [
                [
                    {"role": "user", "content": "Is 2+2=4? Answer Yes or No."},
                ],
            ]
            
            results = interface.chat(dialogues, max_tokens=1024, desc="Testing")
            print("Chat result:", results)
            
            # Test chat with logprobs
            results_with_logprobs = interface.chat(
                dialogues, 
                max_tokens=1024, 
                logprobs=None,
                desc="Testing Phase 1 thinking"
            )
            print("Phase 1 thinking:", results_with_logprobs)
            
            # Test continuation for logprob extraction
            continuation_dialogues = [
                [
                    {"role": "user", "content": "Is 2+2=4? Answer Yes or No."},
                    {"role": "assistant", "content": "Let me think. 2+2 equals 4. So my answer is"},
                ],
            ]
            
            continuation_results = interface.chat_with_continuation(
                continuation_dialogues,
                max_tokens=1,
                logprobs=20,
                desc="Testing continuation"
            )
            print("Continuation result:", continuation_results)
            
            interface.free_memory()
            print("Test completed successfully!")
            
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("vLLM not available. Skipping tests.")
