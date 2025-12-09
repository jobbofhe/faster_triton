# Simplified Llama3 Model Configuration for Low Memory Environments

LLAMA3_8B_CONFIG = {
    "hidden_size": 256,  # 进一步减少到256以降低内存使用
    "intermediate_size": 1024,  # 保持不变
    "num_attention_heads": 4,  # 进一步减少到4以降低内存使用
    "num_key_value_heads": 2,  # 进一步减少到2以降低内存使用
    "num_hidden_layers": 1,
    "vocab_size": 128256,
    "max_position_embeddings": 512,  # 从8192减少到512
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "rope_theta": 500000.0,
}
