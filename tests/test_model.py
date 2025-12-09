#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Llama3-8B 模型测试脚本
用于验证模型定义和 Triton 内核是否正确集成
"""

import os
import sys
import torch
import json
from models.llama3_model import Llama3ForCausalLM
from configs.llama3_8b_config import LLAMA3_8B_CONFIG

def test_model_initialization():
    """测试模型初始化"""
    print("\n=== 测试模型初始化 ===")
    
    try:
        # 初始化模型
        model = Llama3ForCausalLM(LLAMA3_8B_CONFIG)
        
        # 检查模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型初始化成功")
        print(f"   总参数数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise

def test_forward_pass(model):
    """测试模型前向传播"""
    print("\n=== 测试模型前向传播 ===")
    
    try:
        # 创建随机输入
        batch_size = 2
        seq_len = 128
        vocab_size = model.config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        print(f"输入形状: {input_ids.shape}")
        print(f"标签形状: {labels.shape}")
        
        # 前向传播
        loss, logits = model(input_ids, labels=labels)
        
        print(f"✅ 前向传播成功")
        print(f"   损失值: {loss.item():.4f}")
        print(f"   Logits 形状: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_usage(model):
    """测试 GPU 加速"""
    print("\n=== 测试 GPU 加速 ===")
    
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            
            # 创建 GPU 输入
            batch_size = 2
            seq_len = 128
            vocab_size = model.config.vocab_size
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            # 前向传播
            loss, logits = model(input_ids, labels=labels)
            
            print(f"✅ GPU 加速测试成功")
            print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"   损失值: {loss.item():.4f}")
            
            return True
        else:
            print(f"⚠️  未检测到 GPU，跳过 GPU 加速测试")
            return False
    except Exception as e:
        print(f"❌ GPU 加速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triton_kernels():
    """测试 Triton 内核是否正确导入"""
    print("\n=== 测试 Triton 内核导入 ===")
    
    try:
        from kernels.triton_kernels import fused_attention, fused_mlp, rms_norm
        
        print(f"✅ Triton 内核导入成功")
        print(f"   可用内核: fused_attention, fused_mlp, rms_norm")
        
        return True
    except Exception as e:
        print(f"❌ Triton 内核导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Llama3-8B 模型测试")
    print("=" * 50)
    
    try:
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 运行测试
        results = []
        
        # 测试 1: Triton 内核导入
        results.append(test_triton_kernels())
        
        # 测试 2: 模型初始化
        model = test_model_initialization()
        results.append(True)
        
        # 测试 3: 前向传播
        results.append(test_forward_pass(model))
        
        # 测试 4: GPU 加速
        results.append(test_gpu_usage(model))
        
        # 汇总结果
        print("\n" + "=" * 50)
        print("📊 测试结果汇总")
        print(f"总测试数: {len(results)}")
        print(f"通过测试: {sum(results)}")
        print(f"失败测试: {len(results) - sum(results)}")
        
        if all(results):
            print("🎉 所有测试通过！模型可以正常工作。")
            sys.exit(0)
        else:
            print("💥 部分测试失败，模型需要进一步调试。")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)