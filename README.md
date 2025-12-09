# Triton Llama3-8B Training & Inference Project

这是一个使用 Triton 实现 Llama3-8B 模型完整训练和推理流程的项目。

# 操作指南
``` bash
# cd faster_triton
python3 examples/train_llama3.py  --config examples/training_config.json

# 生成的模型文件会保存在 output/ 目录下
```