# 简单测试模型前向传播

import os
import sys
import torch

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入模型
from models.llama3_model import Llama3ForCausalLM
from configs.llama3_8b_config import LLAMA3_8B_CONFIG

def test_model_forward():
    """测试模型前向传播"""
    print("测试模型前向传播...")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 初始化模型
        print("初始化模型...")
        model = Llama3ForCausalLM(LLAMA3_8B_CONFIG).cuda()
        print("✅ 模型初始化成功")
        
        # 创建测试输入
        batch_size = 1
        seq_len = 128
        input_ids = torch.randint(0, LLAMA3_8B_CONFIG["vocab_size"], (batch_size, seq_len)).cuda()
        labels = input_ids.clone()
        
        print(f"测试输入形状: {input_ids.shape}")
        
        # 前向传播
        print("执行前向传播...")
        loss, logits = model(input_ids, labels)
        
        print("✅ 前向传播成功")
        print(f"损失值: {loss.item()}")
        print(f"Logits形状: {logits.shape}")
        
        # 反向传播
        print("执行反向传播...")
        loss.backward()
        
        print("✅ 反向传播成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_forward()