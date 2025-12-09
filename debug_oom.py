#!/usr/bin/env python3
"""
Debug script to trace memory usage during backward pass
"""

import os
import sys
import torch
import gc

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.llama3_model import Llama3ForCausalLM
from configs.llama3_8b_config import LLAMA3_8B_CONFIG

def print_memory(title):
    """Print current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        print(f"{title}")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Free: {free_gb:.2f} GB")
        print(f"   Total: {total / (1024**3):.2f} GB")
        return allocated, reserved, free_gb
    return 0, 0, 0

def debug_backward_pass():
    """Debug backward pass memory usage"""
    print("=== Debugging Backward Pass OOM ===")
    
    # Set CUDA memory configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    print_memory("1. Initial Memory")
    
    # Create model
    print("\n2. Creating Model...")
    model = Llama3ForCausalLM(LLAMA3_8B_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print_memory("3. After Model to GPU")
    
    # Create dummy input
    batch_size = 2
    seq_len = 512
    vocab_size = LLAMA3_8B_CONFIG["vocab_size"]
    
    print(f"\n4. Creating Input: batch_size={batch_size}, seq_len={seq_len}")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print_memory("5. After Input Creation")
    
    # Forward pass with memory tracing
    print("\n6. Forward Pass...")
    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        loss, logits = model(input_ids, labels=input_ids)
    
    print_memory("7. After Forward Pass")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Logits shape: {logits.shape}")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    print_memory("8. After Cache Clear")
    
    # Test different batch sizes and seq lengths
    test_cases = [
        (1, 256),  # Small batch, small seq
        (1, 128),  # Small batch, very small seq
        (1, 64),   # Tiny batch, tiny seq
    ]
    
    for batch_size, seq_len in test_cases:
        print(f"\n=== Testing with batch_size={batch_size}, seq_len={seq_len} ===")
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        print_memory("  Initial Memory")
        
        # Create new input
        input_ids_test = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        print_memory("  After Input Creation")
        
        try:
            # Forward pass
            loss, _ = model(input_ids_test, labels=input_ids_test)
            print_memory("  After Forward Pass")
            
            # Backward pass
            print("  Backward Pass...")
            loss.backward()
            print("  ✅ Backward pass successful!")
            print_memory("  After Backward Pass")
            
            break  # If successful, exit loop
            
        except torch.OutOfMemoryError as e:
            print(f"  ❌ OOM Error: {e}")
            print_memory("  After OOM Error")
            
            # Clear cache for next test
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n=== Analysis ===")
    print("If even the smallest test case fails, there might be:")
    print("1. A memory leak in the model implementation")
    print("2. Inefficient memory management in Triton kernels")
    print("3. Incorrect memory statistics reporting")
    print("4. Other processes using GPU memory")
    
    # Check what processes are using GPU memory
    print("\n=== GPU Memory Usage by Process ===")
    try:
        os.system("nvidia-smi")
    except:
        print("Failed to run nvidia-smi")

if __name__ == "__main__":
    debug_backward_pass()