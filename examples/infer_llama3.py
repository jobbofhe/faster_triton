# 推理 Llama3-8B 模型的示例脚本

import argparse
import json

from inference.inferencer import Llama3Inferencer


def main():
    parser = argparse.ArgumentParser(description="Infer with Llama3-8B model using Triton")
    parser.add_argument("--config", type=str, required=True, help="Path to inference configuration file")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 初始化推理器
    inferencer = Llama3Inferencer(config)
    
    # 加载模型
    print(f"Loading model from {config["model_path"]}...")
    inferencer.load_model(config["model_path"])
    
    # 生成文本
    print(f"Generating text with prompt: '{args.prompt}'")
    generated_text = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # 输出结果
    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
