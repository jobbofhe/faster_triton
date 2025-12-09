#!/usr/bin/env python3

import torch
import sys
import os

# 将当前目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kernels.triton_kernels import rms_norm

# 测试RMSNorm的简单脚本
print("正在测试RMSNorm...")

# 创建简单的测试输入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 4, 8, device=device)
weight = torch.randn(8, device=device)
epsilon = 1e-6

print(f"输入形状: {x.shape}")
print(f"权重形状: {weight.shape}")
print(f"使用设备: {device}")

try:
    print("调用rms_norm函数...")
    result = rms_norm(x, weight, epsilon)
    print(f"输出形状: {result.shape}")
    print(f"输出结果: {result}")
    
    # 简单验证结果是否合理
    if torch.isnan(result).any():
        print("❌ RMSNorm测试失败: 结果包含NaN值")
    elif torch.isinf(result).any():
        print("❌ RMSNorm测试失败: 结果包含Inf值")
    else:
        print("✅ RMSNorm测试成功！")
except Exception as e:
    print(f"❌ RMSNorm测试失败: {e}")
    import traceback
    traceback.print_exc()
