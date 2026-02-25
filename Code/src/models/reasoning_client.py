# src/models/reasoning_client.py
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .base import BaseModel

class ReasoningAPIWrapper(BaseModel):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, max_workers: int = 5, **kwargs):
        """
        Client for reasoning models (e.g., o1-preview, o1-mini, o3-mini, DeepSeek-R1).

        Characteristics:
        1. Does not support logprobs (confidence is set to None).
        2. Uses max_completion_tokens in some APIs.
        3. Typically enforces temperature=1 or default settings.
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.max_workers = max_workers

    def _request_single_prompt(self, prompt: str, max_new_tokens: int) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            
            choice = response.choices[0]
            reasoning_content = getattr(choice.message, "reasoning_content", None)
            text = choice.message.content.strip()
            
            return {
                "reasoning_content": reasoning_content,
                "text": text,
                "probability": None
            }
            
        except Exception:
            return {"text": "Error", "probability": None}

    def predict_batch(self, prompts: List[str], max_new_tokens=4000) -> List[str]:
        results = self.predict_batch_with_prob(prompts, max_new_tokens)
        return [r['text'] for r in results]

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens=4000) -> List[Dict]:
        """
        Execute concurrent requests.

        Note: Reasoning models are typically slower; use conservative concurrency.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            iterable = [(prompt, max_new_tokens) for prompt in prompts]
            future_results = list(executor.map(lambda p: self._request_single_prompt(*p), iterable))
            
        return future_results