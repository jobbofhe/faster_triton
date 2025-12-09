# 测试模型加载的最小化脚本

import os
import sys
import torch

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Current Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA memory allocated:", torch.cuda.memory_allocated())
    print("CUDA memory cached:", torch.cuda.memory_reserved())

# 尝试导入并加载模型
print("\n尝试导入模型相关模块...")
try:
    from configs.llama3_8b_config import LLAMA3_8B_CONFIG
    print("成功导入模型配置")
    print("模型配置:", LLAMA3_8B_CONFIG)
    
    from models.llama3_model import Llama3ForCausalLM
    print("成功导入Llama3ForCausalLM")
    
    print("\n尝试创建模型实例...")
    model = Llama3ForCausalLM(LLAMA3_8B_CONFIG)
    print("成功创建模型实例")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    print("\n尝试将模型移动到GPU...")
    model = model.to("cuda")
    print("成功将模型移动到GPU")
    
    if torch.cuda.is_available():
        print("CUDA memory allocated after model loading:", torch.cuda.memory_allocated())
        print("CUDA memory cached after model loading:", torch.cuda.memory_reserved())
        
    print("\n模型加载测试成功！")
    
    # 测试前向传播
    print("\n测试前向传播...")
    try:
        # 创建一个简单的输入张量 [batch_size, seq_length]
        input_ids = torch.randint(0, LLAMA3_8B_CONFIG["vocab_size"], (2, 512), device="cuda")
        
        # 前向传播（无labels参数）
        loss, logits = model(input_ids)
        print("前向传播成功")
        print("logits形状:", logits.shape)
        print("损失值:", loss)
        
        if torch.cuda.is_available():
            print("CUDA memory after forward:", torch.cuda.memory_allocated())
    except Exception as e:
        print(f"前向传播错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 测试反向传播和优化器
    print("\n测试反向传播和优化器...")
    try:
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        print("优化器创建成功")
        
        # 使用有labels的前向传播来计算损失
        labels = torch.randint(0, LLAMA3_8B_CONFIG["vocab_size"], (2, 512), device="cuda")
        loss, logits = model(input_ids, labels=labels)
        print("带标签的前向传播成功")
        print("损失值:", loss)
        
        # 反向传播
        loss.backward()
        print("反向传播成功")
        
        if torch.cuda.is_available():
            print("CUDA memory after backward:", torch.cuda.memory_allocated())
        
        # 更新参数
        optimizer.step()
        print("参数更新成功")
        
        optimizer.zero_grad()
        print("梯度清零成功")
        
        print("\n反向传播和优化器测试成功！")
    except Exception as e:
        print(f"反向传播或优化器错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n所有测试都已通过！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
