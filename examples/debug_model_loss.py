# Simple debug script to test model loss calculation
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from models.llama3_model import Llama3ForCausalLM
from configs.llama3_8b_config import LLAMA3_8B_CONFIG

# Create a smaller config for faster testing
test_config = LLAMA3_8B_CONFIG.copy()
test_config["hidden_size"] = 256
test_config["num_attention_heads"] = 4
test_config["num_key_value_heads"] = 2
test_config["intermediate_size"] = 512
test_config["num_hidden_layers"] = 2

# Initialize the model
model = Llama3ForCausalLM(test_config).cuda()
print(f"Model initialized with config: {test_config}")

# Create random input_ids and labels
batch_size = 1
max_seq_length = 128
vocab_size = test_config["vocab_size"]

# Generate random input_ids and labels
input_ids = torch.randint(0, 1000, (batch_size, max_seq_length)).cuda()
labels = torch.randint(0, 1000, (batch_size, max_seq_length)).cuda()

print(f"Input_ids shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Input_ids sample: {input_ids[0][:10]}")
print(f"Labels sample: {labels[0][:10]}")

# Test forward pass without labels
with torch.no_grad():
    _, logits_no_labels = model(input_ids)
    print(f"\nLogits without labels shape: {logits_no_labels.shape}")
    print(f"Logits without labels sample: {logits_no_labels[0][:2]}")

# Test forward pass with labels
with torch.no_grad():
    loss, logits_with_labels = model(input_ids, labels=labels)
    print(f"\nLogits with labels shape: {logits_with_labels.shape}")
    print(f"Logits with labels sample: {logits_with_labels[0][:2]}")
    print(f"Loss: {loss.item()}")

# Test with matching input_ids and labels (should have non-zero loss)
matching_labels = input_ids.clone()
with torch.no_grad():
    loss_matching, _ = model(input_ids, labels=matching_labels)
    print(f"\nLoss with matching labels: {loss_matching.item()}")

# Test with different labels (should have different loss)
different_labels = input_ids.clone()
different_labels[0, 0] = (different_labels[0, 0] + 1) % 1000
with torch.no_grad():
    loss_different, _ = model(input_ids, labels=different_labels)
    print(f"Loss with different labels: {loss_different.item()}")

# Test with all labels set to the same value
same_labels = torch.full_like(labels, 0).cuda()
with torch.no_grad():
    loss_same, _ = model(input_ids, labels=same_labels)
    print(f"Loss with all labels 0: {loss_same.item()}")

# Test loss function directly
logits = logits_with_labels.view(-1, vocab_size)
targets = labels.view(-1)
loss_fct = torch.nn.CrossEntropyLoss()
direct_loss = loss_fct(logits, targets)
print(f"\nDirect cross entropy loss: {direct_loss.item()}")
print(f"Model loss matches direct loss: {abs(loss.item() - direct_loss.item()) < 1e-6}")
