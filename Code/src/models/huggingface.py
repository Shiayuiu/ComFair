# src/models/huggingface.py
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple

class HFModelWrapper:
    def __init__(self, model_path: str, device_map="auto", load_in_4bit=False):
        print(f"Loading HF Model from: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
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

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens=10) -> List[Dict]:
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
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # 3. 解析结果
        results = []
        
        # compute_transition_scores 可以计算生成序列的 log_probs
        # shape: (batch_size, sequence_length)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        
        for i, scores in enumerate(transition_scores):
            # 获取生成的纯文本
            gen_text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip()
            
            # 计算置信度：取生成序列中所有 token 概率的几何平均值 (Geometric Mean)
            # 也可以简单地取 sum(log_probs) 然后 exp，或者取第一个 token 的 prob
            # 这里我们采用 exp(mean(log_probs)) 作为序列置信度
            
            # 过滤掉 pad token 的分数 (pad token 的 log_prob 通常是 0 或 -inf，视处理而定，这里简单处理)
            valid_scores = scores[scores != -float('inf')]
            if len(valid_scores) > 0:
                avg_log_prob = valid_scores.mean().item()
                probability = np.exp(avg_log_prob)
            else:
                probability = 0.0
                
            results.append({
                "text": gen_text,
                "probability": round(probability, 6) # 保留6位小数
            })
            
        return results