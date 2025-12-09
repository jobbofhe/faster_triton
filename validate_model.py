# Simple validation script to test model functionality
import torch
import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Import model and config
    from models.llama3_model import Llama3ForCausalLM
    from configs.llama3_8b_config import LLAMA3_8B_CONFIG
    
    print("✅ 成功导入模型和配置")
    
    # Test model initialization
    print("\n测试模型初始化...")
    model = Llama3ForCausalLM(LLAMA3_8B_CONFIG)
    print("✅ 模型初始化成功")
    
    # Test model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数: {total_params/1e9:.2f}B")
    
    # Test forward pass
    print("\n测试前向传播...")
    input_ids = torch.randint(0, LLAMA3_8B_CONFIG["vocab_size"], (1, 10))
    labels = torch.randint(0, LLAMA3_8B_CONFIG["vocab_size"], (1, 10))
    
    try:
        loss, logits = model(input_ids, labels)
        print(f"✅ 前向传播成功，loss: {loss.item():.4f}")
        print(f"   logits形状: {logits.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n验证完成")
