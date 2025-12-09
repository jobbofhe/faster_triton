# Debug script to check DataCollatorForLanguageModeling behavior
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from transformers import DataCollatorForLanguageModeling
import torch

# Create a simple tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.vocab_size = 1000
        self.model_max_length = 10

# Test DataCollatorForLanguageModeling
print("Testing DataCollatorForLanguageModeling with MLM=False")
print("="*50)

# Create tokenizer and collator
tokenizer = SimpleTokenizer()
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create a batch of input_ids
input_ids = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
encoded_inputs = {"input_ids": input_ids}

# Apply the collator - note: collator expects a list of dicts
batch = collator([{"input_ids": ids} for ids in input_ids])

print(f"Input_ids: {input_ids}")
print(f"Collated input_ids: {batch['input_ids']}")
print(f"Collated labels: {batch['labels']}")
print(f"Are input_ids and labels the same? {torch.allclose(batch['input_ids'], batch['labels'])}")

# Test with padding
print("\nTesting with padding")
print("="*50)

input_ids_padded = [[1, 2, 3], [4, 5, 6, 7, 8]]
encoded_inputs_padded = {"input_ids": input_ids_padded}
# Apply the collator
batch_padded = collator([{"input_ids": ids} for ids in input_ids_padded])

print(f"Input_ids: {input_ids_padded}")
print(f"Collated input_ids: {batch_padded['input_ids']}")
print(f"Collated labels: {batch_padded['labels']}")
print(f"Are input_ids and labels the same? {torch.allclose(batch_padded['input_ids'], batch_padded['labels'])}")

# Test with mock tokenizer similar to the one in trainer.py
print("\nTesting with mock tokenizer similar to trainer.py")
print("="*50)

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
        return text.split()
        
    def __call__(self, text, padding=None, truncation=None, max_length=None):
        actual_max_length = max_length or self.model_max_length
        import random
        if isinstance(text, list):
            return {"input_ids": [[random.randint(0, 999) for _ in range(actual_max_length)] for _ in text]}
        else:
            return {"input_ids": [random.randint(0, 999) for _ in range(actual_max_length)]}
    
    def pad(self, encoded_inputs, padding="longest", max_length=None, pad_to_multiple_of=None, return_tensors=None):
        if isinstance(encoded_inputs, dict):
            input_ids = encoded_inputs.get("input_ids", [])
        else:
            input_ids = [enc.get("input_ids", []) for enc in encoded_inputs]
        
        if max_length is not None:
            padded_length = max_length
        elif padding == "longest":
            padded_length = max(len(ids) for ids in input_ids)
        elif padding == "max_length":
            padded_length = self.model_max_length
        else:
            padded_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        for ids in input_ids:
            padded_ids = ids + [self.pad_token_id] * (padded_length - len(ids))
            padded_input_ids.append(padded_ids)
        
        result = {"input_ids": padded_input_ids}
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"])
        
        return result

# Create mock tokenizer and collator
max_seq_length = 5
mock_tokenizer = MockTokenizer(max_seq_length)
mock_collator = DataCollatorForLanguageModeling(tokenizer=mock_tokenizer, mlm=False)

# Test with mock tokenizer
texts = ["hello world", "test text"]
tokenized = mock_tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_length)
print(f"Mock tokenizer output: {tokenized}")

# Apply collator - need to convert to list of dicts
input_ids_list = tokenized["input_ids"]
batch_mock = mock_collator([{"input_ids": ids} for ids in input_ids_list])
print(f"\nInput_ids (mock): {batch_mock['input_ids']}")
print(f"Labels (mock): {batch_mock['labels']}")
print(f"Are input_ids and labels the same? {torch.allclose(batch_mock['input_ids'], batch_mock['labels'])}")
