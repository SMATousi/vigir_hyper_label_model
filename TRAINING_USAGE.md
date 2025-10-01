# LELA Transformer Training with Evaluation

## Usage

The `train_transformer_eval_wandb.py` script now accepts command-line arguments to configure all training parameters.

### Basic Usage

```bash
python train_transformer_eval_wandb.py [OPTIONS]
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_runs` | int | 10 | Number of training runs |
| `--num_epochs` | int | 100 | Number of epochs per run |
| `--data_size` | int | 100 | Dataset size for DataLoader |
| `--max_n_lfs` | int | 60 | Maximum number of labeling functions |
| `--max_examples` | int | 2000 | Maximum number of examples |
| `--batch_size` | int | 20 | Batch size |
| `--num_workers` | int | 0 | Number of DataLoader workers |
| `--num_layers` | int | 2 | Number of transformer layers |
| `--max_seq_len` | int | 800000 | Maximum sequence length |
| `--eval_frequency` | int | 100 | Evaluation frequency in iterations |
| `--project_name` | str | "lela-transformer-training" | Wandb project name |

### Example Commands

#### Quick Test Run
```bash
python train_transformer_eval_wandb.py \
    --num_runs 2 \
    --num_epochs 10 \
    --batch_size 16 \
    --max_n_lfs 30 \
    --max_examples 1000 \
    --eval_frequency 50 \
    --project_name "lela-quick-test"
```

#### Full Training Run
```bash
python train_transformer_eval_wandb.py \
    --num_runs 10 \
    --num_epochs 100 \
    --batch_size 32 \
    --max_n_lfs 60 \
    --max_examples 2000 \
    --num_layers 3 \
    --eval_frequency 100 \
    --project_name "lela-full-training"
```

#### Large Scale Experiment
```bash
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
```

### Features

- **Configurable Training**: All major training parameters can be set via command line
- **Dataset Evaluation**: Evaluates on 14 real-world datasets at the end of each epoch
- **Wandb Logging**: Comprehensive logging of training and evaluation metrics
- **Multiple Runs**: Supports running multiple training runs with different random seeds
- **Flexible Architecture**: Configurable number of transformer layers

### Evaluation Datasets

The script evaluates on the following 14 datasets with noise power = 0:
- census, imdb, yelp, youtube, sms, spouse, cdr, commercial
- tennis, basketball, agnews, trec, semeval, chemprot

### Wandb Metrics

The script logs the following metrics:
- **Training**: `train/loss`, `train/synthetic_val_acc`, `train/iteration`
- **Evaluation**: `eval/{dataset}_{f1|accuracy}`, `eval/{dataset}_time`
- **Aggregates**: `eval/avg_f1`, `eval/avg_accuracy`, `eval/avg_overall`

### Help

To see all available options:
```bash
python train_transformer_eval_wandb.py --help
```
