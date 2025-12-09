# Inference pipeline for Llama3-8B using Triton

import torch
from transformers import AutoTokenizer

from models.llama3_model import Llama3ForCausalLM
from configs.llama3_8b_config import LLAMA3_8B_CONFIG
from utils.helpers import get_attention_mask

class Llama3Inferencer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = Llama3ForCausalLM(LLAMA3_8B_CONFIG).to(self.device)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_model(self, model_path):
        """
        加载预训练模型权重
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50):
        """
        文本生成函数
        """
        # 编码输入
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"].to(self.device)
        
        generated_ids = input_ids.clone()
        
        # 生成新token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 获取当前序列长度
                seq_len = generated_ids.shape[-1]
                
                # 如果超过最大长度，停止生成
                if seq_len > self.config["max_seq_length"]:
                    break
                
                # 前向传播
                _, logits = self.model(generated_ids)
                
                # 获取最后一个token的logits
                next_token_logits = logits[:, -1, :]
                
                # 应用温度缩放
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保持至少一个token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = -float("inf")
                
                # Top-k采样
                if top_k > 0:
                    next_token_logits = torch.topk(next_token_logits, top_k)[0]
                
                # 计算概率分布
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 检查是否生成了终止token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # 添加到生成的序列中
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def batch_generate(self, prompts, max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50):
        """
        批量文本生成函数
        """
        # 编码输入
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        )["input_ids"].to(self.device)
        
        generated_ids = inputs.clone()
        batch_size = inputs.shape[0]
        
        # 生成新token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 获取当前序列长度
                seq_len = generated_ids.shape[-1]
                
                # 如果超过最大长度，停止生成
                if seq_len > self.config["max_seq_length"]:
                    break
                
                # 前向传播
                _, logits = self.model(generated_ids)
                
                # 获取最后一个token的logits
                next_token_logits = logits[:, -1, :]
                
                # 应用温度缩放
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits == indices_to_remove.unsqueeze(1),
                        -float("inf")
                    )
                
                # Top-k采样
                if top_k > 0:
                    next_token_logits = torch.topk(next_token_logits, top_k)[0]
                
                # 计算概率分布
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成的序列中
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
        
        # 解码生成的文本
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        
        return generated_texts
