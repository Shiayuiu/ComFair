<<<<<<< HEAD
from .llm_client import LLMClient, BaseLLMClient, HuggingFaceClient, OpenAIClient, AzureOpenAIClient

__all__ = [
    'LLMClient',
    'BaseLLMClient',
    'HuggingFaceClient',
    'OpenAIClient',
    'AzureOpenAIClient'
    ]
=======
from .huggingface import HFModelWrapper
from .openai_client import OpenAIWrapper
from .reasoning_client import ReasoningAPIWrapper

def load_model(model_type: str, model_path_or_name: str, **kwargs):
    """
    统一模型加载入口
    
    Args:
        model_type: 'hf' (本地) 或 'api' (OpenAI兼容接口)
        model_path_or_name: 本地路径 或 API 模型名称 (如 'gpt-4')
        kwargs: 透传参数 (如 device_map, api_key 等)
    """
    if model_type.lower() == 'hf':
        return HFModelWrapper(model_path=model_path_or_name, **kwargs)
    
    elif model_type.lower() == 'api':
        return OpenAIWrapper(model_name=model_path_or_name, **kwargs)
    
    elif model_type.lower() == 'reasoning_api':
        max_workers = kwargs.pop('max_workers', 3) # Reasoning 模型通常限流更严，默认并发设低一点
        return ReasoningAPIWrapper(model_name=model_path_or_name, max_workers=max_workers, **kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'hf' or 'api'.")
>>>>>>> ac22061 (sync latest local version)
