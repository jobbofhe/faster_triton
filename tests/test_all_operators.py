#!/usr/bin/env python3

import torch
import sys
import os

# 将当前目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kernels.triton_kernels import (
    rms_norm,
    rotary_pos_emb,
    rotate_half,
    apply_rotary_pos_emb,
    get_attention_mask
)

# 定义测试设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 测试参数
batch_size = 4
seq_len = 128
hidden_size = 4096
dim = 128  # 确保dim是2的幂，适合作为编译时常量

# 测试RMSNorm
def test_rms_norm():
    print("\n=== 测试RMSNorm ===")
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    weight = torch.randn(hidden_size, device=device, requires_grad=True)
    epsilon = 1e-6
    
    # PyTorch实现（用于比较）
    def torch_rms_norm(x, weight, epsilon):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + epsilon)
        return x_normalized * weight
    
    # 计算结果
    torch_result = torch_rms_norm(x, weight, epsilon)
    triton_result = rms_norm(x, weight, epsilon)
    
    # 比较结果
    print(f"PyTorch结果形状: {torch_result.shape}")
    print(f"Triton结果形状: {triton_result.shape}")
    print(f"结果误差: {torch.max(torch.abs(torch_result - triton_result))}")
    print(f"结果是否接近: {torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)}")
    
    return torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)

# 测试RotaryPositionalEmbedding
def test_rotary_pos_emb():
    print("\n=== 测试RotaryPositionalEmbedding ===")
    
    # 创建测试输入
    max_seq_len = seq_len
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # PyTorch实现（用于比较）
    def torch_rotary_pos_emb(inv_freq, max_seq_len):
        t = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return cos_emb, sin_emb
    
    # 计算结果
    torch_cos, torch_sin = torch_rotary_pos_emb(inv_freq, max_seq_len)
    triton_cos, triton_sin = rotary_pos_emb(inv_freq, max_seq_len)
    
    # 比较结果
    print(f"PyTorch cos形状: {torch_cos.shape}, sin形状: {torch_sin.shape}")
    print(f"Triton cos形状: {triton_cos.shape}, sin形状: {triton_sin.shape}")
    print(f"cos误差: {torch.max(torch.abs(torch_cos - triton_cos))}")
    print(f"sin误差: {torch.max(torch.abs(torch_sin - triton_sin))}")
    print(f"cos是否接近: {torch.allclose(torch_cos, triton_cos, rtol=1e-5, atol=1e-5)}")
    print(f"sin是否接近: {torch.allclose(torch_sin, triton_sin, rtol=1e-5, atol=1e-5)}")
    
    return (torch.allclose(torch_cos, triton_cos, rtol=1e-5, atol=1e-5) and 
            torch.allclose(torch_sin, triton_sin, rtol=1e-5, atol=1e-5))

# 测试rotate_half
def test_rotate_half():
    print("\n=== 测试rotate_half ===")
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    
    # PyTorch实现（用于比较）
    def torch_rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # 计算结果
    torch_result = torch_rotate_half(x)
    triton_result = rotate_half(x)
    
    # 比较结果
    print(f"PyTorch结果形状: {torch_result.shape}")
    print(f"Triton结果形状: {triton_result.shape}")
    print(f"结果误差: {torch.max(torch.abs(torch_result - triton_result))}")
    print(f"结果是否接近: {torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)}")
    
    return torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)

# 测试apply_rotary_pos_emb
def test_apply_rotary_pos_emb():
    print("\n=== 测试apply_rotary_pos_emb ===")
    
    # 创建测试输入
    q = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    
    # 创建sin和cos嵌入
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_emb = emb.cos()
    sin_emb = emb.sin()
    
    # PyTorch实现（用于比较）
    def torch_apply_rotary_pos_emb(q, k, cos_emb, sin_emb):
        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_rot = q * cos_emb + rotate_half(q) * sin_emb
        k_rot = k * cos_emb + rotate_half(k) * sin_emb
        return q_rot, k_rot
    
    # 计算结果
    torch_q, torch_k = torch_apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
    triton_q, triton_k = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
    
    # 比较结果
    print(f"PyTorch q形状: {torch_q.shape}, k形状: {torch_k.shape}")
    print(f"Triton q形状: {triton_q.shape}, k形状: {triton_k.shape}")
    print(f"q误差: {torch.max(torch.abs(torch_q - triton_q))}")
    print(f"k误差: {torch.max(torch.abs(torch_k - triton_k))}")
    print(f"q是否接近: {torch.allclose(torch_q, triton_q, rtol=1e-5, atol=1e-5)}")
    print(f"k是否接近: {torch.allclose(torch_k, triton_k, rtol=1e-5, atol=1e-5)}")
    
    return (torch.allclose(torch_q, triton_q, rtol=1e-5, atol=1e-5) and 
            torch.allclose(torch_k, triton_k, rtol=1e-5, atol=1e-5))

# 测试get_attention_mask
def test_get_attention_mask():
    print("\n=== 测试get_attention_mask ===")
    
    # PyTorch实现（用于比较）
    def torch_get_attention_mask(seq_len):
        mask = torch.full((seq_len, seq_len), -float('inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
    
    # 计算结果
    torch_result = torch_get_attention_mask(seq_len)
    triton_result = get_attention_mask(seq_len, device)
    
    # 比较结果
    print(f"PyTorch结果形状: {torch_result.shape}")
    print(f"Triton结果形状: {triton_result.shape}")
    print(f"结果误差: {torch.max(torch.abs(torch_result - triton_result))}")
    print(f"结果是否接近: {torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)}")
    
    return torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)

# 运行所有测试
def run_all_tests():
    print("开始测试所有Triton风格算子...")
    
    tests = [
        test_rms_norm,
        test_rotary_pos_emb,
        test_rotate_half,
        test_apply_rotary_pos_emb,
        test_get_attention_mask
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 失败: {e}")
            results.append(False)
    
    print("\n=== 测试总结 ===")
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"测试 {i}: {test.__name__} - {status}")
    
    if all(results):
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败！")
        return 1

if __name__ == "__main__":
    exit(run_all_tests())
