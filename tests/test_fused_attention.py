import torch
import torch.nn as nn
from kernels.triton_kernels import fused_attention
from utils.helpers import RotaryPositionalEmbedding

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_fused_attention():
    """
    测试fused_attention函数
    """
    print("开始测试fused_attention...")
    
    # 设置测试参数
    batch_size = 2
    seq_len = 16
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    # 创建输入张量
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 创建投影权重
    q_proj_w = torch.randn(num_heads * head_dim, hidden_size, device=device)
    k_proj_w = torch.randn(num_heads * head_dim, hidden_size, device=device)
    v_proj_w = torch.randn(num_heads * head_dim, hidden_size, device=device)
    o_proj_w = torch.randn(hidden_size, num_heads * head_dim, device=device)
    
    # 创建旋转嵌入
    rotary_emb = RotaryPositionalEmbedding(head_dim, base=10000.0)
    
    # 调用fused_attention函数
    try:
        output = fused_attention(
            hidden_states, q_proj_w, k_proj_w, v_proj_w, o_proj_w, rotary_emb
        )
        print(f"✅ fused_attention输出形状: {output.shape}")
        print(f"o_proj_w形状: {o_proj_w.shape}")
        print("✅ fused_attention测试通过")
        return True
    except Exception as e:
        print(f"❌ fused_attention测试失败: {e}")
        print(f"hidden_states形状: {hidden_states.shape}")
        print(f"q_proj_w形状: {q_proj_w.shape}")
        print(f"k_proj_w形状: {k_proj_w.shape}")
        print(f"v_proj_w形状: {v_proj_w.shape}")
        print(f"o_proj_w形状: {o_proj_w.shape}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fused_attention()