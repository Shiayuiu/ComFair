import os
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, APIError
from .base import BaseModel

class ReasoningAPIWrapper(BaseModel):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, max_workers: int = 5, **kwargs):
        """
        专门用于 Reasoning 模型 (如 o1-preview, o1-mini, o3-mini, DeepSeek-R1) 的客户端
        特点: 
        1. 不支持 logprobs (置信度只能设为 None)
        2. 参数使用 max_completion_tokens
        3. 强制 temperature=1 或使用 default
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.max_workers = max_workers

    def _request_single_prompt(self, prompt: str, max_new_tokens: int) -> Dict:
        try:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=1000,      
            )
            
            choice = response.choices[0]
            reasoning_content = choice.message.reasoning_content
            text = choice.message.content.strip()
            
            # Reasoning 内容提取 (如果模型支持返回思维链，通常在 reasoning_content 字段，但标准接口只返回 content)
            # 这里我们只取最终结果
            
            return {
                "reasoning_content": reasoning_content, 
                "text": text,
                "probability": None  # Reasoning 模型无法计算置信度
            }
            
        except Exception as e:
            print(f"  [Reasoning API Error] {self.model_name} | {e}")
            # 简单的重试逻辑或返回空
            return {"text": "Error", "probability": None}

    def predict_batch(self, prompts: List[str], max_new_tokens=10) -> List[str]:
        results = self.predict_batch_with_prob(prompts, max_new_tokens)
        return [r['text'] for r in results]

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens=1000) -> List[Dict]:
        """
        并发请求
        注意：Reasoning 模型通常较慢，并发数不宜过高，且超时时间可能需要更长
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            iterable = [(prompt, max_new_tokens) for prompt in prompts]
            future_results = list(executor.map(lambda p: self._request_single_prompt(*p), iterable))
            
        return future_results