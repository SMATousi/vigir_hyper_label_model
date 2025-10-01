#!/bin/bash

# Example usage of train_transformer_eval_wandb.py with different configurations

echo "=== Example 1: Quick test run ==="
python train_transformer_eval_wandb.py \
    --num_runs 2 \
    --num_epochs 10 \
    --batch_size 16 \
    --max_n_lfs 30 \
    --max_examples 1000 \
    --eval_frequency 50 \
    --project_name "lela-quick-test"

echo "=== Example 2: Full training run ==="
python train_transformer_eval_wandb.py \
    --num_runs 10 \
    --num_epochs 100 \
    --batch_size 32 \
    --max_n_lfs 60 \
    --max_examples 2000 \
    --num_layers 3 \
    --eval_frequency 100 \
    --project_name "lela-full-training"

echo "=== Example 3: Large scale experiment ==="
python train_transformer_eval_wandb.py \
    --num_runs 5 \
    --num_epochs 200 \
    --batch_size 64 \
    --max_n_lfs 100 \
    --max_examples 5000 \
    --num_layers 4 \
    --num_workers 4 \
    --max_seq_len 1000000 \
    --eval_frequency 200 \
    --project_name "lela-large-scale"

echo "=== To see all available options, run: ==="
echo "python train_transformer_eval_wandb.py --help"
