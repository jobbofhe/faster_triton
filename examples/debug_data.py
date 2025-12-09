# Debug script to test tokenizer and data collator behavior
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from transformers import DataCollatorForLanguageModeling
import torch

# Create the mock tokenizer
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
                attention_mask = encoded_inputs.get("attention_mask", None)
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
            
            if pad_to_multiple_of is not None:
                padded_length = ((padded_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            
            padded_input_ids = []
            attention_masks = []
            
            for ids in input_ids:
                padded_ids = ids + [self.pad_token_id] * (padded_length - len(ids))
                padded_input_ids.append(padded_ids)
                
                mask = [1] * len(ids) + [0] * (padded_length - len(ids))
                attention_masks.append(mask)
            
            result = {"input_ids": padded_input_ids, "attention_mask": attention_masks}
            
            if isinstance(encoded_inputs, dict):
                for key, value in encoded_inputs.items():
                    if key not in ["input_ids", "attention_mask"]:
                        if isinstance(value, list) and isinstance(value[0], list):
                            padded_values = []
                            for v in value:
                                padded_v = v + [0] * (padded_length - len(v))
                                padded_values.append(padded_v)
                            result[key] = padded_values
            
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor(result["input_ids"])
                result["attention_mask"] = torch.tensor(result["attention_mask"])
            
            return result

# Test the tokenizer and data collator
max_seq_length = 128
mock_tokenizer = MockTokenizer(max_seq_length)

# Create a sample input
sample_text = ["Hello world", "This is a test"]
tokenized = mock_tokenizer(sample_text, padding="max_length", truncation=True, max_length=max_seq_length)
print(f"Tokenized output: {tokenized}")
print(f"Input_ids shape: {len(tokenized['input_ids'])}, {len(tokenized['input_ids'][0])}")

# Test the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=mock_tokenizer,
    mlm=False
)

# Prepare batch
batch = tokenized
batch_with_labels = data_collator(batch)

print(f"\nBatch with labels: {batch_with_labels.keys()}")
print(f"Input_ids shape: {batch_with_labels['input_ids'].shape}")
print(f"Labels shape: {batch_with_labels['labels'].shape}")
print(f"Input_ids sample: {batch_with_labels['input_ids'][0][:10]}")
print(f"Labels sample: {batch_with_labels['labels'][0][:10]}")
print(f"Are input_ids and labels the same? {torch.allclose(batch_with_labels['input_ids'], batch_with_labels['labels'])}")

# Test with the mock dataset
from datasets import DatasetDict, Dataset

# Generate mock data
def generate_random_text(length=50):
    import string
    import random
    letters = string.ascii_letters + string.digits + string.punctuation + " "
    return ''.join(random.choice(letters) for _ in range(length))

train_data = [{"text": generate_random_text()} for _ in range(3)]
dataset = DatasetDict({"train": Dataset.from_list(train_data)})

# Tokenize dataset
def tokenize_function(examples):
    return mock_tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Test data collator on tokenized dataset
batch = tokenized_datasets["train"][:2]
batch_with_labels = data_collator(batch)

print(f"\nFrom tokenized dataset:")
print(f"Input_ids: {batch_with_labels['input_ids']}")
print(f"Labels: {batch_with_labels['labels']}")
print(f"Are they the same? {torch.allclose(batch_with_labels['input_ids'], batch_with_labels['labels'])}")
