#!/usr/bin/env python3
"""
Test script to diagnose OOM issues with low memory Llama3 configuration
"""

import os
import sys
import torch

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.llama3_model import Llama3ForCausalLM
from configs.llama3_8b_config import LLAMA3_8B_CONFIG

def test_model_memory():
    """Test model memory usage"""
    print("=== Testing Low Memory Model ===")
    
    # Print current memory status
    print("\n1. Initial Memory Status:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"   Free GPU memory: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
        print(f"   Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # Print model configuration
    print("\n2. Model Configuration:")
    for key, value in LLAMA3_8B_CONFIG.items():
        print(f"   {key}: {value}")
    
    # Calculate expected model size
    vocab_size = LLAMA3_8B_CONFIG["vocab_size"]
    hidden_size = LLAMA3_8B_CONFIG["hidden_size"]
    num_layers = LLAMA3_8B_CONFIG["num_hidden_layers"]
    
    # Embedding layer: vocab_size * hidden_size
    emb_size = vocab_size * hidden_size
    # Each layer has: attention + mlp + norms
    layer_size = hidden_size * hidden_size * 4  # Rough estimate
    total_params = emb_size + (num_layers * layer_size)
    
    # Convert to GB (float32 is 4 bytes per parameter)
    expected_size_gb = (total_params * 4) / (1024**3)
    print(f"\n3. Expected Model Size:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Expected GPU memory (float32): {expected_size_gb:.2f} GB")
    
    # Initialize model
    print("\n4. Initializing Model...")
    model = Llama3ForCausalLM(LLAMA3_8B_CONFIG)
    
    # Check memory after model creation
    print("\n5. Memory After Model Creation:")
    if torch.cuda.is_available():
        print(f"   Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"   Reserved GPU memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    # Move model to GPU
    print("\n6. Moving Model to GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Check memory after moving to GPU
    print("\n7. Memory After Moving to GPU:")
    if torch.cuda.is_available():
        print(f"   Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"   Reserved GPU memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    # Test forward pass with small batch
    print("\n8. Testing Forward Pass...")
    batch_size = 2
    seq_length = 512
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Memory after creating input: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # Forward pass
    with torch.no_grad():
        loss, logits = model(input_ids, labels=input_ids)
    
    print(f"   Forward pass successful!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Memory after forward pass: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # Test with actual training step
    print("\n9. Testing Training Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    loss, logits = model(input_ids, labels=input_ids)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    print(f"   Training step successful!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Memory after training step: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    print("\n🎉 All tests completed successfully!")
    print("The model is working correctly with the low memory configuration.")
    print("If you still get OOM errors in training, check:")
    print("1. Batch size (try smaller)")
    print("2. Sequence length (try shorter)")
    print("3. Triton kernel implementation")
    print("4. Other processes using GPU memory")

if __name__ == "__main__":
    test_model_memory()