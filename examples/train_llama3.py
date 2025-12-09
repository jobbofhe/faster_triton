# 训练 Llama3-8B 模型的示例脚本

import os
import sys
import argparse
import json
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

try:
    from training.trainer import Llama3Trainer
    print("Successfully imported Llama3Trainer")
except Exception as e:
    print("Failed to import Llama3Trainer:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    print("Starting main function...")
    try:
        parser = argparse.ArgumentParser(description="Train Llama3-8B model using Triton")
        parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
        args = parser.parse_args()
        print(f"Parsed arguments: {args}")
        
        # 加载配置
        print(f"Loading configuration from: {args.config}")
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded configuration: {config}")
        
        # 初始化训练器
        print("Initializing Llama3Trainer...")
        trainer = Llama3Trainer(config)
        print("Successfully initialized Llama3Trainer")
        
        # 加载数据
        print("Loading training data...")
        trainer.load_data(config["dataset_path"])
        print("Successfully loaded training data")
        
        # 开始训练
        print("Starting training...")
        trainer.train()
        print("Training completed successfully")
        
        # 暂时注释掉模型保存，避免权限问题
        # trainer.save_model(config["output_model_path"])
        # print(f"Final model saved to {config['output_model_path']}")
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
