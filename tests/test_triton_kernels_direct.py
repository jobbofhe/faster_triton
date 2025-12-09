# 直接测试Triton内核功能

import os
import sys
import torch

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入Triton内核
from kernels.triton_kernels import fused_mlp, rms_norm

def test_silu_kernel():
    """测试SiLU Triton内核"""
    print("\n=== 测试SiLU Triton内核 ===")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 创建测试输入
        batch_size = 1
        seq_len = 128
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size).cuda()
        
        # 使用PyTorch原生SiLU
        y_pytorch = torch.nn.functional.silu(x)
        
        # 使用Triton内核
        y_triton = torch.empty_like(x)
        from kernels.triton_kernels import _silu_kernel
        
        # 配置网格
        grid = (x.shape[0], x.shape[1],)
        
        # 调用内核
        _silu_kernel[grid](
            x_ptr=x,
            y_ptr=y_triton,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            BLOCK_SIZE=128
        )
        
        # 验证结果
        max_diff = torch.max(torch.abs(y_pytorch - y_triton)).item()
        print(f"PyTorch和Triton结果的最大差异: {max_diff}")
        
        if max_diff < 1e-5:
            print("✅ SiLU Triton内核测试通过")
            return True
        else:
            print("❌ SiLU Triton内核测试失败，差异过大")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rms_norm_kernel():
    """测试RMS Norm Triton内核"""
    print("\n=== 测试RMS Norm Triton内核 ===")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 创建测试输入
        batch_size = 1
        seq_len = 128
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size).cuda()
        weight = torch.randn(hidden_size).cuda()
        eps = 1e-5
        
        # 使用PyTorch实现RMS Norm
        def pytorch_rms_norm(x, weight, eps):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return x * weight
        
        y_pytorch = pytorch_rms_norm(x, weight, eps)
        
        # 使用Triton内核
        y_triton = rms_norm(x, weight, eps)
        
        # 验证结果
        max_diff = torch.max(torch.abs(y_pytorch - y_triton)).item()
        print(f"PyTorch和Triton结果的最大差异: {max_diff}")
        
        if max_diff < 1e-5:
            print("✅ RMS Norm Triton内核测试通过")
            return True
        else:
            print("❌ RMS Norm Triton内核测试失败，差异过大")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fused_mlp():
    """测试融合MLP"""
    print("\n=== 测试融合MLP ===")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 创建测试输入和权重
        batch_size = 1
        seq_len = 128
        hidden_size = 4096
        intermediate_size = 11008  # Llama3-8B的中间层大小
        
        x = torch.randn(batch_size, seq_len, hidden_size).cuda()
        gate_proj_w = torch.randn(intermediate_size, hidden_size).cuda()
        up_proj_w = torch.randn(intermediate_size, hidden_size).cuda()
        down_proj_w = torch.randn(hidden_size, intermediate_size).cuda()
        
        # 执行融合MLP
        output = fused_mlp(x, gate_proj_w, up_proj_w, down_proj_w)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        # 检查输出形状
        if output.shape == x.shape:
            print("✅ 融合MLP测试通过")
            return True
        else:
            print("❌ 融合MLP测试失败，输出形状不正确")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试Triton内核功能...")
    
    results = []
    results.append(test_silu_kernel())
    results.append(test_rms_norm_kernel())
    results.append(test_fused_mlp())
    
    print("\n=== 测试总结 ===")
    if all(results):
        print("✅ 所有Triton内核测试通过！")
    else:
        print("❌ 部分Triton内核测试失败！")