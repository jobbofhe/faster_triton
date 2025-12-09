# Model definition for Llama3-8B using Triton

import torch
import torch.nn as nn
import triton
import triton.language as tl

from kernels.triton_kernels import fused_attention, fused_mlp, rms_norm, rotary_pos_emb, rotate_half, apply_rotary_pos_emb, get_attention_mask
from utils.helpers import RMSNorm, RotaryPositionalEmbedding

class Llama3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config["max_position_embeddings"]
        
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 初始化旋转位置编码
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config["rope_theta"])
    
    def forward(self, hidden_states):
        # 使用 Triton 优化的融合注意力
        output = fused_attention(
            hidden_states, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.o_proj.weight,
            self.rotary_emb
        )
        return output

class Llama3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, hidden_states):
        # 使用 Triton 优化的融合MLP
        return fused_mlp(hidden_states, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)

class Llama3Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.self_attn = Llama3Attention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.mlp = Llama3MLP(config)
    
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon)
        hidden_states = self.self_attn(hidden_states)
        hidden_states += residual
        
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        
        return hidden_states

class Llama3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([Llama3Layer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
    
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = rms_norm(hidden_states, self.norm.weight, self.norm.variance_epsilon)
        return hidden_states

class Llama3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Llama3Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # 权重共享
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, input_ids, labels=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config["vocab_size"]), labels.view(-1))
        
        return loss, logits
