#!/usr/bin/env python3

import torch
import sys
import os

# 将当前目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kernels.triton_kernels import (
    rotary_pos_emb,
    rotate_half,
    apply_rotary_pos_emb,
    get_attention_mask
)

# 定义测试设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 测试参数
batch_size = 2
seq_len = 4
dim = 8

# 测试rotate_half
def test_rotate_half():
    print("\n=== 测试rotate_half ===")
    
    # 创建测试输入 (使用固定的dim=8，适合作为编译时常量)
    x = torch.randn(2, 4, 8, device="cuda")
    print(f"输入形状: {x.shape}")
    print(f"输入值: {x}")
    
    try:
        result = rotate_half(x)
        print(f"输出形状: {result.shape}")
        print(f"输出值: {result}")
        print("✅ rotate_half测试成功！")
        return True
    except Exception as e:
        print(f"❌ rotate_half测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 测试apply_rotary_pos_emb
def test_apply_rotary_pos_emb():
    print("\n=== 测试apply_rotary_pos_emb ===")
    
    # 创建测试输入 (使用固定的dim=8，适合作为编译时常量)
    q = torch.randn(2, 4, 8, device="cuda")
    k = torch.randn(2, 4, 8, device="cuda")
    
    # 创建sin和cos嵌入
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 8, 2, device="cuda").float() / 8))
    t = torch.arange(4, device="cuda", dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_emb = emb.cos()
    sin_emb = emb.sin()
    
    print(f"q形状: {q.shape}")
    print(f"k形状: {k.shape}")
    print(f"cos_emb形状: {cos_emb.shape}")
    print(f"sin_emb形状: {sin_emb.shape}")
    
    try:
        result_q, result_k = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
        print(f"输出q形状: {result_q.shape}")
        print(f"输出k形状: {result_k.shape}")
        print("✅ apply_rotary_pos_emb测试成功！")
        return True
    except Exception as e:
        print(f"❌ apply_rotary_pos_emb测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 测试get_attention_mask
def test_get_attention_mask():
    print("\n=== 测试get_attention_mask ===")
    
    seq_len = 4
    device = "cuda"
    print(f"序列长度: {seq_len}")
    
    try:
        result = get_attention_mask(seq_len, device)
        print(f"输出形状: {result.shape}")
        print(f"输出值: {result}")
        print("✅ get_attention_mask测试成功！")
        return True
    except Exception as e:
        print(f"❌ get_attention_mask测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 运行测试
def run_tests():
    print("开始测试其他Triton风格算子...")
    
    tests = [
        test_rotate_half,
        test_apply_rotary_pos_emb,
        test_get_attention_mask
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
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
    exit(run_tests())
