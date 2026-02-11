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

    Supports:
    - OpenAI official
    - DashScope (Alibaba) OpenAI-compatible endpoint
    - Other OpenAI-compatible gateways

    Key fix:
    - DashScope Qwen3 non-streaming requires enable_thinking=false, otherwise 400.
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
            timeout=request_timeout,  # OpenAI SDK supports timeout
        )
        self.max_workers = int(max_workers)
        self.max_retries = int(max_retries)

    # ---------------------------
    # Public APIs
    # ---------------------------
    def predict_batch(self, prompts: List[str], max_new_tokens: int = 500) -> List[str]:
        results = self.predict_batch_with_prob(prompts, max_new_tokens=max_new_tokens)
        return [r["text"] for r in results]

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens: int = 500) -> List[Dict]:
        # Thread pool for I/O bound requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            iterable = [(prompt, max_new_tokens) for prompt in prompts]
            results = list(executor.map(lambda p: self._request_single_prompt(*p), iterable))
        return results

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _is_qwen3(self) -> bool:
        """
        Detect Qwen3 model IDs like: qwen3-8b, qwen3-14b, qwen3-32b, etc.
        """
        model_id = (self.model_name or "").strip().lower()
        return model_id.startswith("qwen3")

    def _build_request_kwargs(self, prompt: str, max_new_tokens: int) -> Dict:
        """
        Build kwargs for chat.completions.create(...)
        """
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            stream=False,  # IMPORTANT: you are using non-streaming pipeline
            # temperature=1.0,
            # logprobs=True,
            # top_logprobs=1,
        )

        # DashScope/Qwen3 rule: enable_thinking must be false for non-streaming calls.
        # Put it into extra_body (safe for OpenAI-compatible endpoints).
        if self._is_qwen3():
            kwargs["extra_body"] = {"enable_thinking": False}

        return kwargs

    def _extract_text_and_prob(self, response) -> Dict:
        choice = response.choices[0]
        text = (choice.message.content or "").strip()

        # sequence confidence (optional): exp(avg(log_probs))
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

        # Build once (for stable params). If you need per-retry mutation, move inside loop.
        request_kwargs = self._build_request_kwargs(prompt, max_new_tokens)

        while retry_count <= self.max_retries:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return self._extract_text_and_prob(response)

            except RateLimitError as e:
                # 429: retry with exponential backoff
                retry_count += 1
                if retry_count > self.max_retries:
                    break
                print(f"  [Rate Limit] {self.model_name} | Retry {retry_count}/{self.max_retries} | Waiting {backoff_time}s...")
                time.sleep(backoff_time)
                backoff_time *= 2

            except APIStatusError as e:
                # APIStatusError provides HTTP status code
                status = getattr(e, "status_code", None)

                # 400/401/403 are usually non-retriable (bad params / auth / permission)
                if status in (400, 401, 403, 404):
                    print(f"  [API Error] {self.model_name} | HTTP {status} | {e}")
                    return {"text": "Error", "probability": 0.0}

                # 429 treated as rate limit
                if status == 429:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        break
                    print(f"  [Rate Limit] {self.model_name} | Retry {retry_count}/{self.max_retries} | Waiting {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue

                # 5xx / others: retry
                retry_count += 1
                if retry_count > self.max_retries:
                    break
                print(f"  [API Error] {self.model_name} | HTTP {status} | Retrying in {backoff_time}s... | {e}")
                time.sleep(backoff_time)
                backoff_time *= 2

            except APIConnectionError as e:
                # Network/connection: retry
                retry_count += 1
                if retry_count > self.max_retries:
                    break
                print(f"  [API Connection Error] {self.model_name} | Retrying in {backoff_time}s... | {e}")
                time.sleep(backoff_time)
                backoff_time *= 2

            except Exception as e:
                # Unknown error: do not retry forever; return placeholder
                print(f"  [API Error] Model: {self.model_name} | Error: {e}")
                return {"text": "Error", "probability": 0.0}

        # exhausted retries
        return {"text": "Error", "probability": 0.0}
