import os
import numpy as np
import time
from typing import List, Dict
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from concurrent.futures import ThreadPoolExecutor
from .base import BaseModel

class OpenAIWrapper(BaseModel):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, max_workers: int = 50, **kwargs):
        """
        model_name: 例如 'gpt-4', 'gpt-3.5-turbo'
        api_key: 如果不传，默认读取环境变量 OPENAI_API_KEY
        base_url: 可选，用于支持第三方中转或 vLLM
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.max_workers = max_workers
        self.max_retries = 5

    def predict_batch(self, prompts: List[str], max_new_tokens=500) -> List[str]:
        # 复用带概率的方法，只取 text
        results = self.predict_batch_with_prob(prompts, max_new_tokens)
        return [r['text'] for r in results]
    
    def _request_single_prompt(self, prompt: str, max_new_tokens: int) -> Dict:
        """
        内部私有方法：处理单条 Prompt 的请求与解析
        """
        retry_count = 0
        backoff_time = 2

        while retry_count <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    # temperature=1.0,
                    # The following line use in Qwen3
                    extra_body={"enable_thinking": False},
                    # logprobs=True,
                    # top_logprobs=1
                )
                choice = response.choices[0]
                text = choice.message.content.strip()

                # 计算序列置信度: P = exp(avg(log_probs))
                if choice.logprobs and choice.logprobs.content:
                    token_logprobs = [token.logprob for token in choice.logprobs.content]
                    avg_log_prob = np.mean(token_logprobs)
                    probability = np.exp(avg_log_prob)
                else:
                    probability = 0.0

                return {
                    "text": text,
                    "probability": round(float(probability), 6)
                }
            
            except RateLimitError as e:
                # 429 错误：限流
                retry_count += 1
                if retry_count > self.max_retries: break
                print(f"  [Rate Limit] {self.model_name} | Retry {retry_count}/{self.max_retries} | Waiting {backoff_time}s...")
                time.sleep(backoff_time)
                backoff_time *= 2  # 指数增加等待时间

            except (APIConnectionError, APIStatusError) as e:
                # 网络连接错误或服务器 500 错误
                retry_count += 1
                if retry_count > self.max_retries: break
                print(f"  [API Error] {self.model_name} | {e} | Retrying in {backoff_time}s...")
                time.sleep(backoff_time)
                backoff_time *= 2
            except Exception as e:
                # 增加基本的重试逻辑或错误占位
                print(f"  [API Error] Model: {self.model_name} | Error: {e}")
                return {"text": "Error", "probability": 0.0}

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens=500) -> List[Dict]:

        # 使用线程池加速 I/O 绑定任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map 会严格按照 prompts 的顺序返回结果
            # 相比于 as_completed，map 更适合需要保持输入输出顺序一致的评估任务
            iterable = [(prompt, max_new_tokens) for prompt in prompts]
            
            # 使用 lambda 包装函数以传递多个参数
            future_results = list(executor.map(lambda p: self._request_single_prompt(*p), iterable))
            
        return future_results