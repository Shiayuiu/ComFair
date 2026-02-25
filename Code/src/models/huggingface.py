# src/models/huggingface.py
import torch
import numpy as np
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel

class HFModelWrapper(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        HuggingFace local model wrapper.

        Args:
            model_path: Local path or HF Hub ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            device: "cuda" or "cpu" (ignored if device_map="auto")
            kwargs: Additional arguments passed to AutoModel (e.g., load_in_8bit)
        """
        self.model_path = model_path
        self.device = device
        
        print(f">>> Loading HF Model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Ensure pad token exists (some models do not define one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Use left padding for generation tasks
        self.tokenizer.padding_side = "left"

        # Load model
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **kwargs
        )
        self.model.eval()

    def predict_batch(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        """Return generated text only."""
        results = self.predict_batch_with_prob(prompts, max_new_tokens)
        return [r['text'] for r in results]

    def predict_batch_with_prob(self, prompts: List[str], max_new_tokens: int = 500) -> List[Dict]:
        """
        Run batch inference and return generated text with confidence scores.
        Output format matches API wrapper.
        """
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        results = []

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        generated_tokens = outputs.sequences[:, input_len:]
        
        for i, (gen_seq, scores) in enumerate(zip(generated_tokens, transition_scores)):
            text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
            
            valid_mask = (gen_seq != self.tokenizer.pad_token_id)
            valid_scores = scores[valid_mask]
            
            if len(valid_scores) > 0:
                avg_log_prob = torch.mean(valid_scores).item()
                probability = np.exp(avg_log_prob)
            else:
                probability = 0.0

            results.append({
                "text": text.strip(),
                "probability": round(float(probability), 6)
            })

        del inputs, outputs, transition_scores
        torch.cuda.empty_cache()
        
        return results