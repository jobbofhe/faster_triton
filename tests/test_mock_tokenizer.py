#!/usr/bin/env python3
"""
Test script for MockTokenizer functionality
"""

import sys
import os
import torch

# Add the parent directory to the path so we can import from training module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import Llama3Trainer from the module
from training.trainer import Llama3Trainer

# Test 1: Create a MockTokenizer instance
def test_tokenizer_creation():
    print("=== Test 1: Tokenizer Creation ===")
    # Provide all required configuration parameters
    config = {
        "learning_rate": 5e-05,
        "weight_decay": 0.1,
        "batch_size": 2,
        "num_epochs": 1,
        "max_seq_length": 512,
        "max_grad_norm": 1.0,
        "logging_steps": 100,
        "save_steps": 1,
        "tokenizer_path": "mock-tokenizer",
        "dataset_path": "mock-dataset",
        "output_model_path": "mock-model.pt"
    }
    trainer = Llama3Trainer(config)
    tokenizer = trainer.tokenizer
    print(f"✓ MockTokenizer created successfully")
    print(f"  - vocab_size: {tokenizer.vocab_size}")
    print(f"  - model_max_length: {tokenizer.model_max_length}")
    print(f"  - pad_token_id: {tokenizer.pad_token_id}")
    print(f"  - bos_token_id: {tokenizer.bos_token_id}")
    print(f"  - eos_token_id: {tokenizer.eos_token_id}")
    return tokenizer

# Test 2: Test tokenize method
def test_tokenize_method(tokenizer):
    print("\n=== Test 2: Tokenize Method ===")
    text = "Hello world! This is a test."
    tokens = tokenizer.tokenize(text)
    print(f"✓ Tokenize method works: {tokens}")

# Test 3: Test __call__ method (single text)
def test_call_single(tokenizer):
    print("\n=== Test 3: __call__ Method (Single Text) ===")
    text = "Hello world!"
    result = tokenizer(text)
    print(f"✓ __call__ single text works")
    print(f"  - input_ids shape: {len(result['input_ids'])}")

# Test 4: Test __call__ method (batch)
def test_call_batch(tokenizer):
    print("\n=== Test 4: __call__ Method (Batch) ===")
    texts = ["Hello world!", "This is a test."]
    result = tokenizer(texts)
    print(f"✓ __call__ batch works")
    print(f"  - batch size: {len(result['input_ids'])}")
    print(f"  - each input_ids shape: {len(result['input_ids'][0])}, {len(result['input_ids'][1])}")

# Test 5: Test pad method (without return_tensors)
def test_pad_method(tokenizer):
    print("\n=== Test 5: Pad Method (Without return_tensors) ===")
    # Create some mock inputs
    inputs = {
        "input_ids": [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10]
        ]
    }
    padded = tokenizer.pad(inputs, padding="longest")
    print(f"✓ Pad method works")
    print(f"  - input_ids: {padded['input_ids']}")
    print(f"  - attention_mask: {padded['attention_mask']}")
    assert len(padded["input_ids"]) == 3
    assert len(padded["input_ids"][0]) == 5  # longest sequence length
    assert len(padded["attention_mask"]) == 3
    assert len(padded["attention_mask"][0]) == 5

# Test 6: Test pad method (with return_tensors="pt")
def test_pad_with_tensors(tokenizer):
    print("\n=== Test 6: Pad Method (With return_tensors='pt') ===")
    # Create some mock inputs
    inputs = {
        "input_ids": [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10]
        ]
    }
    padded = tokenizer.pad(inputs, padding="longest", return_tensors="pt")
    print(f"✓ Pad method with return_tensors works")
    print(f"  - input_ids shape: {padded['input_ids'].shape}")
    print(f"  - attention_mask shape: {padded['attention_mask'].shape}")
    assert isinstance(padded["input_ids"], torch.Tensor)
    assert isinstance(padded["attention_mask"], torch.Tensor)
    assert padded["input_ids"].shape == (3, 5)
    assert padded["attention_mask"].shape == (3, 5)

# Test 7: Test pad method with max_length and pad_to_multiple_of
def test_pad_with_params(tokenizer):
    print("\n=== Test 7: Pad Method (With max_length and pad_to_multiple_of) ===")
    # Create some mock inputs
    inputs = {
        "input_ids": [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10]
        ]
    }
    padded = tokenizer.pad(inputs, max_length=10, pad_to_multiple_of=4, return_tensors="pt")
    print(f"✓ Pad method with max_length and pad_to_multiple_of works")
    print(f"  - input_ids shape: {padded['input_ids'].shape}")  # should be (3, 12) since pad_to_multiple_of=4
    print(f"  - attention_mask shape: {padded['attention_mask'].shape}")
    assert padded["input_ids"].shape == (3, 12)  # 10 rounded up to next multiple of 4 is 12
    assert padded["attention_mask"].shape == (3, 12)

if __name__ == "__main__":
    print("Testing MockTokenizer functionality...")
    try:
        tokenizer = test_tokenizer_creation()
        test_tokenize_method(tokenizer)
        test_call_single(tokenizer)
        test_call_batch(tokenizer)
        test_pad_method(tokenizer)
        test_pad_with_tensors(tokenizer)
        test_pad_with_params(tokenizer)
        print("\n🎉 All tests passed! MockTokenizer is fully functional.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)