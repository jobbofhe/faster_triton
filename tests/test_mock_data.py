# 测试mock数据生成功能

import os
import sys

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.trainer import Llama3Trainer

# 创建测试配置
config = {
    "learning_rate": 5e-5,
    "weight_decay": 0.1,
    "batch_size": 2,
    "num_epochs": 1,
    "max_seq_length": 512,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 1,
    "tokenizer_path": "bert-base-uncased",  # 使用一个常用的tokenizer
    "dataset_path": "non_existent_dataset",  # 不存在的数据集路径，用于模拟下载失败
    "output_model_path": "llama3_8b_mock_trained.pt"
}

try:
    # 初始化训练器
    trainer = Llama3Trainer(config)
    
    # 加载数据（应该会触发mock数据生成）
    print("Testing mock data generation...")
    trainer.load_data(config["dataset_path"])
    
    # 检查是否成功创建了数据加载器
    if hasattr(trainer, "train_dataloader"):
        print("✅ Mock data generation successful!")
        print(f"   Train dataloader created with {len(trainer.train_dataloader)} batches")
        
        # 尝试获取一个batch的数据
        batch = next(iter(trainer.train_dataloader))
        print(f"   Batch keys: {batch.keys()}")
        print(f"   Input IDs shape: {batch['input_ids'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print("✅ Batch data generated successfully!")
    else:
        print("❌ Failed to create train dataloader")
        
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()