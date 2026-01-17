# src/models/huggingface.py
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from .base import BaseModel

class HFModelWrapper(BaseModel):
    def __init__(self, model_path: str, device_map="auto", load_in_4bit=False):
        print(f"Loading HF Model from: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        
        model_kwargs = {"torch_dtype": torch.bfloat16}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        # 注意：这里我们直接加载 Model 实例，以便更好地控制生成参数
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device_map, 
            trust_remote_code=True,
            **model_kwargs
        )
        self.device = self.model.device

        # Classfication任务上的优化
        self.label_token_ids = {
            "A": self.tokenizer.encode("A", add_special_tokens=False)[0],
            "B": self.tokenizer.encode("B", add_special_tokens=False)[0],
            }

    # For Task Classification
    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens) -> List[Dict]:
        """
        批量预测并返回文本和置信度 (Probability)
        """
        # 1. Tokenize
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # 2. Generate with Scores
        # output_scores=True: 返回每一步的 logits
        # return_dict_in_generate=True: 返回结构化对象
        max_new_tokens = 1
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample = False,
                temperature = None,
                top_p = None,
                top_k = None,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # 3. 解析结果
        results = []
        
        # outputs.scores: List[length=1] of logits
        # shape of scores[0]: (batch_size, vocab_size)
        first_step_logits = outputs.scores[0]
        first_step_probs = torch.softmax(first_step_logits, dim=-1)

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        
        for i in range(generated_tokens.shape[0]):
            token_id = generated_tokens[i, 0].item()
            gen_text = self.tokenizer.decode(
                token_id,
                skip_special_tokens=True
            ).strip()

            # 只取 A / B 的概率
            if gen_text in self.label_token_ids:
                prob = first_step_probs[i, token_id].item()
            else:
                # 非法输出（如空、特殊符号）
                prob = 0.0

            PROB_PRECISION = 6
            results.append({
                "text": gen_text,
                "probability": round(prob, PROB_PRECISION)  # 严格定义的分类概率
            })
            
        return results