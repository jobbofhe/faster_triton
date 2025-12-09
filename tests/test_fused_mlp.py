import torch
import torch.nn as nn
from kernels.triton_kernels import fused_mlp

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_fused_mlp():
    """
    测试fused_mlp函数
    """
    print("开始测试fused_mlp...")
    
    # 设置测试参数
    batch_size = 2
    seq_len = 16
    hidden_size = 512
    intermediate_size = 2048
    
    # 创建输入张量
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 创建投影权重
    gate_proj_w = torch.randn(intermediate_size, hidden_size, device=device)
    up_proj_w = torch.randn(intermediate_size, hidden_size, device=device)
    down_proj_w = torch.randn(hidden_size, intermediate_size, device=device)
    
    # 调用fused_mlp函数
    try:
        output = fused_mlp(
            hidden_states, gate_proj_w, up_proj_w, down_proj_w
        )
        print(f"✅ fused_mlp输出形状: {output.shape}")
        print("✅ fused_mlp测试通过")
        return True
    except Exception as e:
        print(f"❌ fused_mlp测试失败: {e}")
        print(f"hidden_states形状: {hidden_states.shape}")
        print(f"gate_proj_w形状: {gate_proj_w.shape}")
        print(f"up_proj_w形状: {up_proj_w.shape}")
        print(f"down_proj_w形状: {down_proj_w.shape}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fused_mlp()