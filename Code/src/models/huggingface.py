# src/models/huggingface.py
import torch
import numpy as np
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel

class HFModelWrapper(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        HuggingFace 本地模型包装器
        
        Args:
            model_path: 本地路径或 HF Hub ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            device: "cuda" or "cpu" (如果使用 device_map="auto" 则忽略此参数)
            kwargs: 其他传给 AutoModel 的参数 (如 load_in_8bit)
        """
        self.model_path = model_path
        self.device = device
        
        print(f">>> Loading HF Model from: {model_path}")
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # 修正 Padding Token (很多新模型默认没有 pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 生成任务通常需要左侧 Padding，否则 pad token 会混入输入导致输出异常
        self.tokenizer.padding_side = "left"

        # 2. 加载模型
        # 默认使用 bfloat16 以节省显存并保持精度 (如果硬件支持)
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",  # 自动分配多卡或 CPU/GPU
            trust_remote_code=True,
            **kwargs
        )
        self.model.eval() # 设置为评估模式

    def predict_batch(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        """简化的预测接口，仅返回文本"""
        results = self.predict_batch_with_prob(prompts, max_new_tokens)
        return [r['text'] for r in results]

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens: int = 500) -> List[Dict]:
        """
        执行批量推理，并返回生成的文本及其对应的置信度。
        格式与 API Wrapper 保持一致。
        """
        # 1. Tokenization
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048 # 防止过长输入 OOM
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_len = input_ids.shape[1]

        # 2. Generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # Greedy Decoding (确定性输出)
                return_dict_in_generate=True,
                output_scores=True,    # 必须开启以计算概率
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 3. 结果解析
        results = []
        
        # transition_scores: 计算每一步生成的 log_prob
        # shape: (batch_size, generated_len)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        generated_tokens = outputs.sequences[:, input_len:]
        
        for i, (gen_seq, scores) in enumerate(zip(generated_tokens, transition_scores)):
            # --- 文本解码 ---
            # skip_special_tokens=True 会去掉 EOS 和 PAD
            text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
            
            # --- 概率计算 ---
            # 1. 过滤掉 Padding Token 的分数 (pad token 的 log_prob 通常没有意义)
            # 注意: transition_scores 已经对应生成的 tokens
            valid_mask = (gen_seq != self.tokenizer.pad_token_id)
            valid_scores = scores[valid_mask]
            
            if len(valid_scores) > 0:
                # 计算平均 Log Probability
                avg_log_prob = torch.mean(valid_scores).item()
                # 转换为线性概率 P = exp(log_P)
                # 这种计算方式表示 "生成整个序列的几何平均置信度"
                probability = np.exp(avg_log_prob)
            else:
                probability = 0.0

            results.append({
                "text": text.strip(),
                "probability": round(float(probability), 6)
            })

        # 清理显存 (可选，但在大 Batch 循环中很有用)
        del inputs, outputs, transition_scores
        torch.cuda.empty_cache()
        
        return results