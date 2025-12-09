# 测试Triton内核的功能

import os
import sys
import torch
import triton

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入测试所需的模块
from kernels.triton_kernels import fused_mlp, rms_norm
from training.trainer import Llama3Trainer

def test_fused_mlp():
    """测试融合MLP的Triton内核"""
    print("\n=== Testing Fused MLP with Triton Kernel ===")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 1024
    hidden_size = 4096
    intermediate_size = 11008  # Llama3-8B的配置
    
    # 随机生成输入张量
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
    
    # 创建权重矩阵
    gate_proj_w = torch.randn(intermediate_size, hidden_size, device="cuda")
    up_proj_w = torch.randn(intermediate_size, hidden_size, device="cuda")
    down_proj_w = torch.randn(hidden_size, intermediate_size, device="cuda")
    
    try:
        # 使用Triton内核的MLP
        output_triton = fused_mlp(hidden_states, gate_proj_w, up_proj_w, down_proj_w)
        print("✅ Triton MLP运行成功")
        print(f"   输出形状: {output_triton.shape}")
        
        # 使用PyTorch实现的MLP作为参考
        def pytorch_mlp(hidden_states, gate_proj_w, up_proj_w, down_proj_w):
            gate = torch.matmul(hidden_states, gate_proj_w.t())
            up = torch.matmul(hidden_states, up_proj_w.t())
            gate = torch.nn.functional.silu(gate)
            intermediate = gate * up
            output = torch.matmul(intermediate, down_proj_w.t())
            return output
        
        output_pytorch = pytorch_mlp(hidden_states, gate_proj_w, up_proj_w, down_proj_w)
        print("✅ PyTorch MLP运行成功")
        
        # 比较结果
        max_diff = torch.max(torch.abs(output_triton - output_pytorch)).item()
        print(f"   最大差异: {max_diff}")
        
        if max_diff < 1e-5:
            print("✅ Triton MLP与PyTorch MLP结果一致")
        else:
            print("❌ Triton MLP与PyTorch MLP结果不一致")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_rms_norm():
    """测试RMS Norm的Triton内核"""
    print("\n=== Testing RMS Norm with Triton Kernel ===")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 1024
    hidden_size = 4096
    
    # 随机生成输入张量和权重
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
    weight = torch.randn(hidden_size, device="cuda")
    eps = 1e-5
    
    try:
        # 使用Triton内核的RMS Norm
        output_triton = rms_norm(x, weight, eps)
        print("✅ Triton RMS Norm运行成功")
        print(f"   输出形状: {output_triton.shape}")
        
        # 使用PyTorch实现的RMS Norm作为参考
        def pytorch_rms_norm(x, weight, eps=1e-5):
            # 计算RMS
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps)
            # 应用权重
            return x_norm * weight
        
        output_pytorch = pytorch_rms_norm(x, weight, eps)
        print("✅ PyTorch RMS Norm运行成功")
        
        # 比较结果
        max_diff = torch.max(torch.abs(output_triton - output_pytorch)).item()
        print(f"   最大差异: {max_diff}")
        
        if max_diff < 1e-5:
            print("✅ Triton RMS Norm与PyTorch RMS Norm结果一致")
        else:
            print("❌ Triton RMS Norm与PyTorch RMS Norm结果不一致")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_triton_installation():
    """测试Triton安装"""
    print("\n=== Testing Triton Installation ===")
    
    try:
        # 测试Triton基本功能
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements):
            idx = tl.program_id(0) * tl.num_programs(0) + tl.program_id(1)
            if idx < n_elements:
                tl.store(output_ptr + idx, tl.load(x_ptr + idx) + tl.load(y_ptr + idx))
        
        # 创建测试数据
        n = 1024
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        output = torch.empty_like(x)
        
        # 配置网格大小
        grid = (triton.cdiv(n, 1024),)
        
        # 运行内核
        add_kernel[grid](x, y, output, n)
        
        # 验证结果
        expected = x + y
        max_diff = torch.max(torch.abs(output - expected)).item()
        
        if max_diff < 1e-5:
            print("✅ Triton基本功能正常")
        else:
            print("❌ Triton基本功能异常")
            
    except Exception as e:
        print(f"❌ Triton安装测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_integration():
    """测试模型与Triton内核的集成"""
    print("\n=== Testing Model Integration with Triton Kernels ===")
    
    # 创建简单的配置
    config = {
        "learning_rate": 5e-5,
        "weight_decay": 0.1,
        "batch_size": 2,
        "num_epochs": 1,
        "max_seq_length": 512,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 1,
        "tokenizer_path": "bert-base-uncased",
        "dataset_path": "non_existent_dataset",  # 使用mock数据
        "output_model_path": "test_model.pt"
    }
    
    try:
        # 初始化训练器
        trainer = Llama3Trainer(config)
        print("✅ 训练器初始化成功")
        
        # 加载mock数据
        trainer.load_data(config["dataset_path"])
        print("✅ Mock数据加载成功")
        
        # 获取一个batch的数据
        batch = next(iter(trainer.train_dataloader))
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        
        # 测试前向传播
        loss, logits = trainer.model(input_ids, labels)
        print("✅ 模型前向传播成功")
        print(f"   损失值: {loss.item()}")
        print(f"   输出logits形状: {logits.shape}")
        
        # 测试反向传播
        loss.backward()
        print("✅ 模型反向传播成功")
        
    except Exception as e:
        print(f"❌ 模型集成测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法测试Triton内核")
        sys.exit(1)
    
    print("开始测试Triton内核功能...")
    
    # 测试Triton安装
    test_triton_installation()
    
    # 测试RMS Norm内核
    test_rms_norm()
    
    # 测试融合MLP内核
    test_fused_mlp()
    
    # 测试模型集成
    test_model_integration()
    
    print("\n=== 所有测试完成 ===")
