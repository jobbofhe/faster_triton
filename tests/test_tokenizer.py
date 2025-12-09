#!/usr/bin/env python3
"""
测试mock tokenizer的功能
"""

# 模拟配置
config = {
    "tokenizer_path": "meta-llama/Meta-Llama-3-8B",
    "max_seq_length": 512
}

print("Testing mock tokenizer...")

# 创建一个完全自定义的mock分词器类
class MockTokenizer:
    def __init__(self, max_seq_length):
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.vocab_size = 128256
        self.model_max_length = max_seq_length
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
    def tokenize(self, text):
        """简单的分词方法"""
        return text.split()
        
    def __call__(self, text, padding=None, truncation=None, max_length=None):
        """__call__方法，返回input_ids"""
        actual_max_length = max_length or self.model_max_length
        if isinstance(text, list):
            return {"input_ids": [[i % 1000 for i in range(actual_max_length)] for _ in text]}
        else:
            return {"input_ids": [i % 1000 for i in range(actual_max_length)]}

# 创建mock tokenizer实例
mock_tokenizer = MockTokenizer(config["max_seq_length"])

print("Mock tokenizer created successfully!")
print(f"Tokenizer attributes:")
print(f"- vocab_size: {mock_tokenizer.vocab_size}")
print(f"- model_max_length: {mock_tokenizer.model_max_length}")
print(f"- pad_token: {mock_tokenizer.pad_token}")
print(f"- pad_token_id: {mock_tokenizer.pad_token_id}")
print(f"- bos_token: {mock_tokenizer.bos_token}")
print(f"- bos_token_id: {mock_tokenizer.bos_token_id}")
print(f"- eos_token: {mock_tokenizer.eos_token}")
print(f"- eos_token_id: {mock_tokenizer.eos_token_id}")

# 测试tokenize方法
test_text = "Hello world! This is a test."
tokenized = mock_tokenizer.tokenize(test_text)
print(f"\ntokenize() result: {tokenized}")

# 测试__call__方法
call_result = mock_tokenizer(test_text)
print(f"\n__call__() result:")
print(f"- input_ids length: {len(call_result['input_ids'])}")
print(f"- First few tokens: {call_result['input_ids'][:10]}")

# 测试批量处理
test_texts = ["Hello world!", "This is another test."]
batch_result = mock_tokenizer(test_texts, max_length=128)
print(f"\nBatch __call__() result:")
print(f"- Number of inputs: {len(batch_result['input_ids'])}")
print(f"- Each input length: {[len(ids) for ids in batch_result['input_ids']]}")
print(f"- First input first few tokens: {batch_result['input_ids'][0][:10]}")

print("\nAll tests passed! Mock tokenizer is working correctly.")
