#!/usr/bin/env python3
"""
LLM客户端抽象层
支持多种后端：OpenAI、Anthropic、Azure、HuggingFace本地模型
"""

import os
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path

import torch

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    @abstractmethod
    def get_prediction(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """获取模型预测"""
        pass
    
    def get_cached_prediction(self, prompt: str) -> Optional[Dict[str, Any]]:
        """获取缓存的预测结果"""
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def _save_to_cache(self, prompt: str, prediction: Dict[str, Any]):
        """保存预测结果到缓存"""
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(prediction, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _get_cache_key(self, prompt: str) -> str:
        """生成缓存键"""
        content = f"{self.__class__.__name__}:{prompt}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    @abstractmethod
    def supports_classification(self) -> bool:
        """是否支持分类任务"""
        pass


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace本地模型客户端"""
    
    def __init__(self, model_path: str, model_name: Optional[str] = None, 
                 task: str = "text-classification", **kwargs):
        super().__init__(kwargs.get('cache_dir'))
        
        self.model_path = model_path
        self.model_name = model_name or os.path.basename(model_path)
        self.task = task
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = kwargs.get('batch_size', 8)
        
        # 延迟加载模型
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        
        logger.info(f"初始化HuggingFace客户端: {model_path} (device: {self.device})")
    
    @property
    def pipeline(self):
        """延迟加载pipeline"""
        if self._pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                
                if self.task == "text-classification":
                    self._pipeline = pipeline(
                        self.task,
                        model=self.model_path,
                        tokenizer=self.model_path,
                        device=self.device,
                        batch_size=self.batch_size
                    )
                else:
                    # 对于生成式模型
                    self._pipeline = pipeline(
                        self.task,
                        model=self.model_path,
                        tokenizer=self.model_path,
                        device=self.device
                    )
                    
            except ImportError:
                raise ImportError("请安装transformers: pip install transformers")
            except Exception as e:
                raise RuntimeError(f"加载模型失败: {e}")
        
        return self._pipeline
    
    def get_prediction(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """获取本地模型预测"""
        
        # 检查缓存
        cached = self.get_cached_prediction(prompt)
        if cached:
            return cached
        
        try:
            if self.task == "text-classification":
                result = self._classify_text(prompt, **kwargs)
            else:
                result = self._generate_text(prompt, **kwargs)
            
            # 保存到缓存
            self._save_to_cache(prompt, result)
            return result
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """文本分类"""
        try:
            results = self.pipeline(prompt, **kwargs)
            
            # 处理单样本和批量预测的返回格式
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # 批量预测返回 [[{...}]] 格式
                    result = results[0][0]
                else:
                    # 单样本预测返回 [{...}] 格式
                    result = results[0]
            else:
                result = results
            
            return {
                'label': str(result['label']),
                'confidence': float(result['score']),
                'raw_output': result
            }
            
        except Exception as e:
            logger.error(f"文本分类失败: {e}")
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """文本生成"""
        try:
            # 对于生成式模型，需要后处理来提取分类标签
            max_length = kwargs.get('max_length', 50)
            result = self.pipeline(prompt, max_length=max_length, **kwargs)
            
            generated_text = result[0]['generated_text'] if isinstance(result, list) else result
            
            # 简单的后处理：提取第一个单词作为标签
            label = generated_text.replace(prompt, '').strip().split()[0] if generated_text else 'UNKNOWN'
            
            return {
                'label': label,
                'confidence': 0.5,  # 生成模型难以提供置信度
                'raw_output': generated_text,
                'generated_text': generated_text
            }
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def supports_classification(self) -> bool:
        return self.task == "text-classification"


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(kwargs.get('cache_dir'))
        
        self.model_name = model_name
        self.api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供")
        
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            openai.api_key = self.api_key
            return openai
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
    
    def get_prediction(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """获取OpenAI模型预测"""
        
        cached = self.get_cached_prediction(prompt)
        if cached:
            return cached
        
        try:
            response = self.client.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 10),
                temperature=kwargs.get('temperature', 0.0)
            )
            
            content = response.choices[0].message.content.strip()
            
            result = {
                'label': content,
                'confidence': 1.0,  # OpenAI不提供置信度
                'raw_output': response
            }
            
            self._save_to_cache(prompt, result)
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def supports_classification(self) -> bool:
        return True

'''
class AnthropicClient(BaseLLMClient):
    """Anthropic Claude客户端"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(kwargs.get('cache_dir'))
        
        self.model_name = model_name
        self.api_key = kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API密钥未提供")
        
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装anthropic: pip install anthropic")
    
    def get_prediction(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """获取Claude模型预测"""
        
        cached = self.get_cached_prediction(prompt)
        if cached:
            return cached
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', 10),
                temperature=kwargs.get('temperature', 0.0),
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            result = {
                'label': content,
                'confidence': 1.0,  # Anthropic不提供置信度
                'raw_output': response
            }
            
            self._save_to_cache(prompt, result)
            return result
            
        except Exception as e:
            logger.error(f"Anthropic API调用失败: {e}")
            return {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def supports_classification(self) -> bool:
        return True
'''

class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI客户端"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.api_version = kwargs.get('api_version', '2023-12-01-preview')
        self.api_base = kwargs.get('api_base') or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not self.api_base:
            raise ValueError("Azure OpenAI端点未提供")
    
    def _initialize_client(self):
        """初始化Azure OpenAI客户端"""
        try:
            import openai
            openai.api_type = "azure"
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            openai.api_version = self.api_version
            return openai
        except ImportError:
            raise ImportError("请安装openai: pip install openai")


class LLMClient:
    """LLM客户端工厂类"""
    
    @staticmethod
    def create(backend: str, **kwargs) -> BaseLLMClient:
        """
        创建LLM客户端
        
        Args:
            backend: 后端类型 ('openai', 'anthropic', 'azure', 'huggingface')
            **kwargs: 后端特定参数
        
        Returns:
            BaseLLMClient实例
        """
        backend = backend.lower()
        
        if backend == 'openai':
            return OpenAIClient(**kwargs)
        # elif backend == 'anthropic':
            # return AnthropicClient(**kwargs)
        elif backend == 'azure':
            return AzureOpenAIClient(**kwargs)
        elif backend == 'huggingface':
            # 对于HuggingFace，需要model_path参数
            if 'model_path' not in kwargs:
                raise ValueError("HuggingFace后端需要model_path参数")
            return HuggingFaceClient(**kwargs)
        else:
            raise ValueError(f"不支持的LLM后端: {backend}")
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """获取可用的后端列表"""
        return ['openai', 'azure', 'huggingface']