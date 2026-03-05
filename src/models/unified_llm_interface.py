import os
import json
import time
import requests
import re
from typing import List, Dict, Optional, Any, Union
from dotenv import load_dotenv

# Import tqdm for progress display
from tqdm import tqdm

# Try to import VLLMInterface, but don't fail if vllm is not installed (for API-only usage)
try:
    from src.models.vllm_interface import VLLMInterface, VLLM_AVAILABLE
except ImportError:
    VLLMInterface = None
    VLLM_AVAILABLE = False
    print("Warning: VLLMInterface could not be imported. Only API usage will be available.")

from openai import OpenAI
import openai

# def extract_belief_from_response(response_text: str) -> Optional[str]:
#     """
#     Extract belief (Yes/No) from the model's response.
    
#     Expected formats based on verbalize prompts:
#     - JSON: {"thinking": "...", "answer": "Yes"} or {"answer": "Yes"}
#     - Markdown JSON: ```json {"answer": "Yes"} ```
#     - Plain text: "Yes" or "No" (short responses)
#     """
#     if not response_text:
#         return None
    
#     # Try to parse as JSON first
#     try:
#         # Try direct JSON parsing
#         data = json.loads(response_text)
#         if isinstance(data, dict):
#             answer = data.get("answer")
#             if answer and isinstance(answer, str):
#                 answer_lower = answer.lower()
#                 if "yes" in answer_lower:
#                     return "Yes"
#                 if "no" in answer_lower:
#                     return "No"
#     except (json.JSONDecodeError, TypeError):
#         pass
    
#     # Try to find JSON in markdown code blocks
#     json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
#     if json_match:
#         try:
#             data = json.loads(json_match.group(1))
#             if isinstance(data, dict):
#                 answer = data.get("answer")
#                 if answer and isinstance(answer, str):
#                     answer_lower = answer.lower()
#                     if "yes" in answer_lower:
#                         return "Yes"
#                     if "no" in answer_lower:
#                         return "No"
#         except (json.JSONDecodeError, TypeError):
#             pass
    
#     # Look for "Answer: Yes" or "Answer: No" (case insensitive)
#     match = re.search(r"answer:\s*(yes|no)", response_text, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()
    
#     # Fallback: Look for just "Yes" or "No" if it's a short response (e.g. < 50 chars)
#     clean_text = response_text.strip().lower()
#     if len(clean_text) < 50:
#         if "yes" in clean_text and "no" not in clean_text:
#             return "Yes"
#         if "no" in clean_text and "yes" not in clean_text:
#             return "No"
            
#     return None


class APIInterface:
    """
    Interface for interacting with closed-source LLMs via API (OpenAI compatible).
    """
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, **kwargs):
        load_dotenv()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        
        if OpenAI is None:
            raise ImportError("openai package is required for APIInterface")
            
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def chat(self, dialogue_history: List[Dict], temperature: float = 0, max_tokens: int = 1000, 
             seed: int = 42, show_progress: bool = True, desc: str = "Processing", **kwargs) -> List[Dict]:
        """
        Generate responses using the API.
        Supports both single dialogue and batch (list of dialogues).
        
        Args:
            dialogue_history: Single dialogue or list of dialogues
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed
            show_progress: Whether to show progress bar for batch processing
            desc: Description for the progress bar
            **kwargs: Additional arguments for the API
            
        Returns:
            List of response dictionaries
        """
        # Handle batching if dialogue_history is a list of lists (batch processing)
        # Note: OpenAI API doesn't support batching in a single request like vLLM, 
        # so we iterate. For true parallelism, we would need async or threads.
        if isinstance(dialogue_history[0], list):
            results = []
            iterator = dialogue_history
            if show_progress:
                iterator = tqdm(dialogue_history, desc=desc, unit="dialogue")
            for dialogue in iterator:
                result = self._chat_single(dialogue, temperature, max_tokens, seed, **kwargs)
                results.append(result)
            return results
        else:
            # Single dialogue
            return [self._chat_single(dialogue_history, temperature, max_tokens, seed, **kwargs)]

    def _chat_single(self, dialogue: List[Dict], temperature: float, max_tokens: int, seed: int, **kwargs) -> Dict:
        max_retries = 8
        base_delay = 1
        retryable_errors = (
            openai.APIConnectionError, 
            openai.RateLimitError,
            openai.APIError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        )

        retries = 0
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=dialogue,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return {
                    "generated_text": completion.choices[0].message.content,
                    "finish_reason": completion.choices[0].finish_reason
                }
            except Exception as e:
                if retries >= max_retries:
                    print(f"Request failed. Reach max retry: {max_retries}")
                    return {"generated_text": "", "error": str(e)}
                
                # Check if error is retryable
                is_retryable = isinstance(e, retryable_errors)
                if not is_retryable:
                    # If it's not a connection/rate limit error, it might be a bad request
                    # But for robustness, we'll retry anyway unless we're sure
                    pass

                delay = base_delay * (2 ** retries)
                print(f"Retry {retries+1}/{max_retries} in {delay}s: {type(e).__name__} - {str(e)}")
                time.sleep(delay)
                retries += 1
    
    def free_memory(self):
        # API doesn't need memory cleanup
        pass


class UnifiedLLMInterface:
    """
    Unified interface for both vLLM and API backends.
    """
    def __init__(self, model_name: str, use_api: bool = False, **kwargs):
        self.use_api = use_api
        self.model_name = model_name
        
        if use_api:
            print(f"Initializing API Interface for model: {model_name}")
            self.interface = APIInterface(model_name, **kwargs)
        else:
            print(f"Initializing vLLM Interface for model: {model_name}")
            if VLLMInterface is None or not VLLM_AVAILABLE:
                raise ImportError("vLLM not installed or VLLMInterface not found. Please install vLLM with: pip install vllm")
            # Filter kwargs for VLLMInterface
            vllm_kwargs = {k: v for k, v in kwargs.items() if k in ['gpu_memory_utilization', 'tensor_parallel_size', 'trust_remote_code', 'dtype', 'enforce_eager']}
            self.interface = VLLMInterface(model_name, **vllm_kwargs)
            self.interface.load_model()

    def chat(self, dialogue_history: Union[List[Dict], List[List[Dict]]], 
             show_progress: bool = True, desc: str = "Processing", **kwargs) -> List[Dict]:
        """
        Unified chat method.
        
        Args:
            dialogue_history: Single dialogue or list of dialogues
            show_progress: Whether to show progress bar for batch processing
            desc: Description for the progress bar
            **kwargs: Additional arguments for the chat method
            
        Returns:
            List of response dictionaries
        """
        return self.interface.chat(dialogue_history, show_progress=show_progress, desc=desc, **kwargs)
    
    def chat_with_continuation(self, dialogue_history: Union[List[Dict], List[List[Dict]]], 
                               show_progress: bool = True, desc: str = "Processing", **kwargs) -> List[Dict]:
        """
        Chat with continuation for logprob extraction (vLLM only).
        
        This method continues the last assistant message to extract logprobs for the next token.
        Only available when using vLLM backend.
        
        Args:
            dialogue_history: Single dialogue or list of dialogues where the last message
                             is from assistant and should be continued
            show_progress: Whether to show progress bar for batch processing
            desc: Description for the progress bar
            **kwargs: Additional arguments for the chat method
            
        Returns:
            List of response dictionaries containing generated_text, logprobs, and probabilities
        """
        if self.use_api:
            raise NotImplementedError("chat_with_continuation is only available with vLLM backend")
        
        return self.interface.chat_with_continuation(dialogue_history, show_progress=show_progress, desc=desc, **kwargs)
    
    def extract_thinking(self, response_text: str) -> str:
        """
        Extract thinking process from response text.
        Only available when using vLLM backend.
        
        Args:
            response_text: The raw response text
            
        Returns:
            Extracted thinking process
        """
        if self.use_api:
            # For API, just return the text as-is (or implement basic filtering)
            return response_text
        
        return self.interface.extract_thinking(response_text)
    
    def free_memory(self):
        if hasattr(self.interface, 'free_memory'):
            self.interface.free_memory()


if __name__ == "__main__":
    model_name = 'openai/gpt-4o-mini'

    interface = UnifiedLLMInterface(model_name, use_api=True)

    dialogue_histories = [
        [
            {"role": "user", "content": "2+2=?"},
        ],
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "Please breifly introduce OpenAI."},
        ]
    ]

    results = interface.chat(dialogue_histories, desc="Testing API")

    print(results)
