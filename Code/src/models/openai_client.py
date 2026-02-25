# src/models/openai_client.py
import os
import time
from typing import List, Dict, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from .base import BaseModel


class OpenAIWrapper(BaseModel):
    """
    OpenAI-compatible Chat Completions wrapper.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 50,
        max_retries: int = 5,
        request_timeout: Optional[float] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=request_timeout,
        )
        self.max_workers = int(max_workers)
        self.max_retries = int(max_retries)

    def predict_batch(self, prompts: List[str], max_new_tokens: int = 500) -> List[str]:
        results = self.predict_batch_with_prob(prompts, max_new_tokens=max_new_tokens)
        return [r["text"] for r in results]

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens: int = 500) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            iterable = [(prompt, max_new_tokens) for prompt in prompts]
            results = list(executor.map(lambda p: self._request_single_prompt(*p), iterable))
        return results

    def _is_qwen3(self) -> bool:
        model_id = (self.model_name or "").strip().lower()
        return model_id.startswith("qwen3")

    def _build_request_kwargs(self, prompt: str, max_new_tokens: int) -> Dict:
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            stream=False,
        )

        if self._is_qwen3():
            kwargs["extra_body"] = {"enable_thinking": False}

        return kwargs

    def _extract_text_and_prob(self, response) -> Dict:
        choice = response.choices[0]
        text = (choice.message.content or "").strip()

        probability = 0.0
        try:
            if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
                token_logprobs = [t.logprob for t in choice.logprobs.content if hasattr(t, "logprob")]
                if token_logprobs:
                    avg_log_prob = float(np.mean(token_logprobs))
                    probability = float(np.exp(avg_log_prob))
        except Exception:
            probability = 0.0

        return {"text": text, "probability": round(probability, 6)}

    def _request_single_prompt(self, prompt: str, max_new_tokens: int) -> Dict:
        retry_count = 0
        backoff_time = 2

        request_kwargs = self._build_request_kwargs(prompt, max_new_tokens)

        while retry_count <= self.max_retries:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return self._extract_text_and_prob(response)

            except RateLimitError:
                retry_count += 1
                if retry_count > self.max_retries:
                    break
                time.sleep(backoff_time)
                backoff_time *= 2

            except APIStatusError as e:
                status = getattr(e, "status_code", None)

                if status in (400, 401, 403, 404):
                    return {"text": "Error", "probability": 0.0}

                if status == 429:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        break
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue

                retry_count += 1
                if retry_count > self.max_retries:
                    break
                time.sleep(backoff_time)
                backoff_time *= 2

            except APIConnectionError:
                retry_count += 1
                if retry_count > self.max_retries:
                    break
                time.sleep(backoff_time)
                backoff_time *= 2

            except Exception:
                return {"text": "Error", "probability": 0.0}

        return {"text": "Error", "probability": 0.0}