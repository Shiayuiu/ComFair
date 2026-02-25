from abc import ABC, abstractmethod
from typing import List, Dict

class BaseModel(ABC):
    @abstractmethod
    def predict_batch(self, prompts: List[str], max_new_tokens: int = 10) -> List[str]:
        """
        Return generated text only.
        """
        pass

    @abstractmethod
    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens: int = 10) -> List[Dict]:
        """
        Return generated text along with confidence scores.
        Return format: [{'text': 'A', 'probability': 0.95}, ...]
        """
        pass