# Minimal test script to verify core components
import torch
import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Test 1: Import basic utilities
    print("1. 测试基础工具导入...")
    from utils.helpers import RMSNorm, RotaryPositionalEmbedding
    print("✅ 成功导入工具类")
    
    # Test 2: Test RMSNorm
    print("\n2. 测试RMSNorm...")
    rms_norm = RMSNorm(hidden_size=4096)
    input_tensor = torch.randn(1, 10, 4096)
    output = rms_norm(input_tensor)
    print(f"✅ RMSNorm测试成功，输入形状: {input_tensor.shape}, 输出形状: {output.shape}")
    
    # Test 3: Test RotaryPositionalEmbedding
    print("\n3. 测试旋转位置编码...")
    rope = RotaryPositionalEmbedding(dim=128, base=500000.0)
    cos, sin = rope(input_tensor[:, :, :128])
    print(f"✅ 旋转位置编码测试成功，cos/sin形状: {cos.shape}")
    
    # Test 4: Test Triton kernels import
    print("\n4. 测试Triton内核导入...")
    from kernels.triton_kernels import fused_attention, fused_mlp, rms_norm
    print("✅ Triton内核导入成功")
    
    print("\n🎉 所有基础组件测试通过！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")
