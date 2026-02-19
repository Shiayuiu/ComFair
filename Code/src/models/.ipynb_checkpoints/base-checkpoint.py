from abc import ABC, abstractmethod
from typing import List, Dict

class BaseModel(ABC):
    @abstractmethod
    def predict_batch(self, prompts: List[str], max_new_tokens: int = 10) -> List[str]:
        """
        仅返回文本结果
        """
        pass

    @abstractmethod
    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens: int = 10) -> List[Dict]:
        """
        返回文本结果 + 置信度
        Return format: [{'text': 'A', 'probability': 0.95}, ...]
        """
        pass