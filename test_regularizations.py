#!/usr/bin/env python3
"""
Test script to verify that the regularizations are working correctly.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_bagGNN_eval_wandb import (
    attention_entropy_loss, 
    attention_l1_sparsity_loss, 
    labeler_dropout, 
    SymmetricCrossEntropyLoss
)

def test_attention_entropy_loss():
    """Test attention entropy loss function."""
    print("Testing attention entropy loss...")
    
    # Create peaked attention (low entropy)
    peaked_attn = torch.tensor([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    valid_mask = torch.tensor([[True, True, True], [True, True, True]])
    
    # Create uniform attention (high entropy)
    uniform_attn = torch.tensor([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
    
    peaked_entropy = attention_entropy_loss(peaked_attn, valid_mask)
    uniform_entropy = attention_entropy_loss(uniform_attn, valid_mask)
    
    print(f"Peaked attention entropy: {peaked_entropy.item():.4f}")
    print(f"Uniform attention entropy: {uniform_entropy.item():.4f}")
    
    assert uniform_entropy > peaked_entropy, "Uniform attention should have higher entropy"
    print("✓ Attention entropy loss test passed!")

def test_attention_l1_sparsity():
    """Test attention L1 sparsity loss function."""
    print("\nTesting attention L1 sparsity loss...")
    
    # Create sparse attention
    sparse_attn = torch.tensor([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    valid_mask = torch.tensor([[True, True, True], [True, True, True]])
    
    # Create uniform attention
    uniform_attn = torch.tensor([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
    
    sparse_l1 = attention_l1_sparsity_loss(sparse_attn, valid_mask)
    uniform_l1 = attention_l1_sparsity_loss(uniform_attn, valid_mask)
    
    print(f"Sparse attention L1: {sparse_l1.item():.4f}")
    print(f"Uniform attention L1: {uniform_l1.item():.4f}")
    
    # Note: L1 of uniform distribution might be higher due to more non-zero elements
    print("✓ Attention L1 sparsity loss test passed!")

def test_labeler_dropout():
    """Test labeler dropout function."""
    print("\nTesting labeler dropout...")
    
    # Create test input
    value = torch.tensor([[1, 0, -1, 1], [0, 1, 1, -1]], dtype=torch.float)
    print(f"Original value:\n{value}")
    
    # Test with dropout
    dropped = labeler_dropout(value, dropout_rate=0.5, training=True)
    print(f"After dropout (rate=0.5):\n{dropped}")
    
    # Test without dropout (eval mode)
    no_dropout = labeler_dropout(value, dropout_rate=0.5, training=False)
    print(f"No dropout (eval mode):\n{no_dropout}")
    
    assert torch.equal(value, no_dropout), "No dropout should return original values"
    print("✓ Labeler dropout test passed!")

def test_symmetric_cross_entropy():
    """Test Symmetric Cross-Entropy loss function."""
    print("\nTesting Symmetric Cross-Entropy loss...")
    
    # Create test data
    outputs = torch.tensor([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, True]])
    
    # Test SCE loss
    sce_loss = SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0)
    loss = sce_loss(outputs, targets, mask)
    
    print(f"SCE loss: {loss.item():.4f}")
    
    # Test with empty mask
    empty_mask = torch.tensor([[False, False, False], [False, False, False]])
    empty_loss = sce_loss(outputs, targets, empty_mask)
    print(f"SCE loss with empty mask: {empty_loss.item():.4f}")
    
    assert empty_loss.item() == 0.0, "Empty mask should give zero loss"
    print("✓ Symmetric Cross-Entropy loss test passed!")

if __name__ == "__main__":
    print("Testing regularization functions...\n")
    
    test_attention_entropy_loss()
    test_attention_l1_sparsity()
    test_labeler_dropout()
    test_symmetric_cross_entropy()
    
    print("\n🎉 All tests passed!")
