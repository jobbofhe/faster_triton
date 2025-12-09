# Helper functions for Llama3-8B

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation using Triton.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # 创建CPU上的inv_freq，但在forward中移到正确的设备
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    def forward(self, x, seq_len=None):
        from kernels.triton_kernels import rotary_pos_emb
        
        if seq_len is None:
            seq_len = x.shape[1]
        
        # 将inv_freq移动到与输入张量相同的设备
        inv_freq = self.inv_freq.to(x.device)
        
        # 使用Triton风格的rotary_pos_emb函数
        cos_emb, sin_emb = rotary_pos_emb(inv_freq, seq_len)
        
        return cos_emb, sin_emb

def rotate_half(x):
    """
    Helper function for RoPE using Triton.
    """
    from kernels.triton_kernels import rotate_half as triton_rotate_half
    return triton_rotate_half(x)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embedding to query and key tensors using Triton.
    """
    from kernels.triton_kernels import apply_rotary_pos_emb as triton_apply_rotary_pos_emb
    return triton_apply_rotary_pos_emb(q, k, cos, sin)

def get_attention_mask(seq_len, device):
    """
    Generate causal attention mask using Triton.
    """
    from kernels.triton_kernels import get_attention_mask as triton_get_attention_mask
    return triton_get_attention_mask(seq_len, device)
