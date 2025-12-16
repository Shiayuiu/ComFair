# src/models/huggingface.py
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class HFModelWrapper:
    def __init__(self, model_path: str, device_map="auto", load_in_4bit=False):
        print(f"Loading HF Model from: {model_path}...")
        
        # 根据需要配置量化等参数
        model_kwargs = {"dtype": torch.bfloat16}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs=model_kwargs,
            device_map=device_map,
            tokenizer=model_path    # 通常 tokenizer 和 model 路径一致
        )
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer.padding_side = 'left'


    def predict_batch(self, prompts: List[str], max_new_tokens=10) -> List[str]:
        """
        批量预测
        """
        # 设置 pad_token_id 避免警告
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        outputs = self.pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            return_full_text=False, # 关键：只返回生成部分，方便解析
            truncation=True,
            pad_token_id=self.tokenizer.pad_token_id,
            batch_size=len(prompts) # 利用 pipeline 的内置 batching
        )
        
        # 提取文本
        result_texts = [out[0]['generated_text'].strip() for out in outputs]
        return result_texts