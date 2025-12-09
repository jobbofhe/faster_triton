# 测试Triton内核与模型的集成

import os
import sys
import torch

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def test_triton_kernels():
    """测试Triton内核"""
    print("开始测试Triton内核...")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 测试_silu_kernel
        print("\n测试_silu_kernel...")
        batch_size = 1
        seq_len = 128
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size).cuda()
        y = torch.empty_like(x)
        
        from kernels.triton_kernels import _silu_kernel
        grid = (batch_size, seq_len,)
        _silu_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            batch_size=batch_size,
            seq_len=seq_len,
            BLOCK_SIZE=128,
            HIDDEN_SIZE=hidden_size
        )
        print("✅ _silu_kernel测试通过")
        
        # 测试rms_norm
        print("\n测试rms_norm...")
        from kernels.triton_kernels import rms_norm
        weight = torch.randn(hidden_size).cuda()
        y = rms_norm(x, weight, eps=1e-5)
        print("✅ rms_norm测试通过")
        
        # 测试fused_mlp
        print("\n测试fused_mlp...")
        from kernels.triton_kernels import fused_mlp
        intermediate_size = 11008
        gate_proj_w = torch.randn(intermediate_size, hidden_size).cuda()
        up_proj_w = torch.randn(intermediate_size, hidden_size).cuda()
        down_proj_w = torch.randn(hidden_size, intermediate_size).cuda()
        output_mlp = fused_mlp(x, gate_proj_w, up_proj_w, down_proj_w)
        print("✅ fused_mlp测试通过")
        
        print("\n✅ 所有Triton内核测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_triton_kernels()