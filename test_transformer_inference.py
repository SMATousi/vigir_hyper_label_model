#!/usr/bin/env python
"""Test script to verify transformer model inference works correctly"""

import numpy as np
import torch
from transformer_models import LELATransformerWrapper
from data import load_dataset_wrench

def test_transformer_inference():
    """Test that transformer model can handle inference without sampling issues"""
    
    # Load a transformer checkpoint if it exists, otherwise create a new model
    try:
        model = LELATransformerWrapper(
            checkpoint_path="./model_checkpoints/model_transformer_0.pt",
            max_seq_len=800,
            inference_max_seq_len=5000  # Allow larger sequences during inference
        )
        print("Loaded trained transformer model")
    except:
        model = LELATransformerWrapper(
            checkpoint_path=None,
            max_seq_len=800,
            inference_max_seq_len=5000
        )
        print("Created new transformer model (no checkpoint found)")
    
    # Test on a small dataset
    print("\nTesting on semeval dataset...")
    X, y = load_dataset_wrench("datasets/semeval")
    
    # Remove cols and rows with all abstentions
    non_zero_cols = np.sum(X >= 0, axis=0) != 0
    X = X[:, non_zero_cols]
    non_zero = np.sum(X >= 0, axis=1) != 0
    X = X[non_zero, :]
    y = y[non_zero]
    
    print(f"Dataset shape: {X.shape}")
    
    # Test prediction
    try:
        # Test predict_prob first
        print("Testing predict_prob...")
        probs = model.predict_prob(X)
        print(f"Probability predictions shape: {probs.shape}")
        print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Test predict
        print("\nTesting predict...")
        preds = model.predict(X)
        print(f"Predictions shape: {preds.shape}")
        print(f"Unique predicted labels: {np.unique(preds)}")
        
        # Check consistency
        assert preds.shape[0] == X.shape[0], "Prediction count mismatch!"
        assert probs.shape[0] == X.shape[0], "Probability prediction count mismatch!"
        
        print("\n✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        raise

if __name__ == "__main__":
    test_transformer_inference()
