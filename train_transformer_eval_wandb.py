import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from data import DatasetOnlineGen
from transformer_models import LELATransformer
from bag_attention_model import LELATransformerBag, StackedLELATransformerBag, LELATransformerBagWrapper
from loss import  BCEMask
import numpy as np
from data import SytheticValidation, load_dataset_wrench
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import time
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train LELA Transformer with evaluation')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of training runs (default: 10)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs per run (default: 100)')
    parser.add_argument('--data_size', type=int, default=100, help='Dataset size for DataLoader (default: 100)')
    parser.add_argument('--max_n_lfs', type=int, default=60, help='Maximum number of labeling functions (default: 60)')
    parser.add_argument('--max_examples', type=int, default=2000, help='Maximum number of examples (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size (default: 20)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers (default: 2)')
    parser.add_argument('--max_seq_len', type=int, default=800000, help='Maximum sequence length (default: 800000)')
    parser.add_argument('--eval_frequency', type=int, default=100, help='Evaluation frequency in iterations (default: 100)')
    parser.add_argument('--project_name', type=str, default='lela-transformer-training', help='Wandb project name')
    return parser.parse_args()

args = parse_args()
NUM_RUNS = args.num_runs

# Evaluation datasets (same as in performance_exp_transformer.py)
EVAL_DATASETS = [
    'census', 
    'imdb', 
    "yelp", 
    'youtube',
    "sms", 
    "spouse",
    'cdr', 
    'commercial', 
    'tennis', 
    'basketball',
    "agnews", 
    "trec", 
    "semeval", 
    "chemprot",
]

def evaluate_on_datasets(model, epoch, run_id):
    """Evaluate model on all datasets with noise power = 0 and log to wandb"""
    model.eval()
    eval_results = {}
    
    # Create a temporary wrapper for evaluation
    temp_checkpoint_path = f"temp_eval_checkpoint_run_{run_id}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
    }, temp_checkpoint_path)
    
    # Create wrapper for evaluation
    eval_wrapper = LELATransformerBagWrapper(
        checkpoint_path=temp_checkpoint_path, 
        max_lf_id=0, 
        use_lf_reliability=False
    )
    
    for dataset_name in EVAL_DATASETS:
        try:
            print(f"Evaluating on {dataset_name}...")
            X, y = load_dataset_wrench(f"datasets/{dataset_name}")
            
            # No noise applied (NOISE_POWER = 0)
            # Remove cols and rows with all abstentions
            non_zero_cols = np.sum(X >= 0, axis=0) != 0
            X = X[:, non_zero_cols]
            non_zero = np.sum(X >= 0, axis=1) != 0
            X = X[non_zero, :]
            y = y[non_zero]
            
            if X.shape[0] == 0:  # Skip if no valid data
                continue
                
            y_ind = np.arange(0, X.shape[0])
            
            # Get predictions
            t_s = time.time()
            pred_round = eval_wrapper.predict(X)
            delta_t = time.time() - t_s
            
            # Calculate accuracy
            gt_labels = np.copy(y).flatten().astype(int)
            y_ind_ = y_ind[gt_labels >= 0]  # only use data points with non-abstention gt labels
            gt_labels = gt_labels[gt_labels >= 0]
            pred_round = pred_round[y_ind_]
            
            if len(gt_labels) == 0:  # Skip if no valid labels
                continue
                
            # Use F1 score for specific datasets, accuracy for others
            if dataset_name in ["sms", "spouse", 'cdr', 'commercial', 'tennis', 'basketball', 'census']:
                score = f1_score(gt_labels, pred_round)
                metric_name = "f1"
            else:
                score = accuracy_score(gt_labels, pred_round)
                metric_name = "accuracy"
            
            eval_results[f"eval/{dataset_name}_{metric_name}"] = score
            eval_results[f"eval/{dataset_name}_time"] = delta_t
            
            print(f"{dataset_name}: {metric_name}={score:.4f}, time={delta_t:.2f}s")
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
    
    # Clean up temporary checkpoint
    if os.path.exists(temp_checkpoint_path):
        os.remove(temp_checkpoint_path)
    
    # Log to wandb
    if eval_results:
        wandb.log({**eval_results, "epoch": epoch})
        
        # Calculate and log average scores
        f1_scores = [v for k, v in eval_results.items() if k.endswith('_f1')]
        acc_scores = [v for k, v in eval_results.items() if k.endswith('_accuracy')]
        
        if f1_scores:
            wandb.log({"eval/avg_f1": np.mean(f1_scores), "epoch": epoch})
        if acc_scores:
            wandb.log({"eval/avg_accuracy": np.mean(acc_scores), "epoch": epoch})
        if f1_scores or acc_scores:
            all_scores = f1_scores + acc_scores
            wandb.log({"eval/avg_overall": np.mean(all_scores), "epoch": epoch})
    
    model.train()
    return eval_results
for i_run in range(NUM_RUNS): #train LELA model with configurable parameters
    # Initialize wandb for this run
    wandb.init(
        project=args.project_name,
        name=f"run_{i_run}",
        config={
            "run_id": i_run,
            "num_epochs": args.num_epochs,
            "max_seq_len": args.max_seq_len,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "max_n_lfs": args.max_n_lfs,
            "max_examples": args.max_examples,
            "data_size": args.data_size,
            "num_workers": args.num_workers,
            "eval_frequency": args.eval_frequency,
        }
    )
    device = "cuda:0"
    # Use sequence length limit to prevent memory explosion
    max_seq_len = args.max_seq_len  # Configurable via command line
    # net = LELATransformer(max_seq_len=max_seq_len)
    # net = LELATransformerBag()
    net = StackedLELATransformerBag(num_layers=args.num_layers)

    optimizer = optim.Adam(
        net.parameters(),
        amsgrad=True,
    )
    net.to(device)

    criterion = BCEMask()

    dataset = DatasetOnlineGen(
        size=args.data_size, # Configurable dataset size
        max_n_lfs=args.max_n_lfs,
        max_example=args.max_examples,
    )


    valid = SytheticValidation()# the sythetic validation set


    collate_fn = dataset.collate

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    writer = SummaryWriter(comment="run_"+str(i_run))
    n_iter = 0
    optimizer.zero_grad()
    n_not_improved = 0
    min_loss = np.inf
    losses = []
    val_accs = []

    torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg':np.mean(val_accs),
            },
            "model_checkpoints/stacked_bag_model_transformer_random.pt"
            )
    
    epoch = 0
    for _ in tqdm(range(args.num_epochs)):  
        if n_not_improved > 10**4:
            break
        
        epoch += 1
        for i, (index, value, labels) in enumerate(dataloader):

            try:
                # Place tensors on GPU
                index = index.int()
                index, value, labels = index.squeeze().to(
                    device), value.squeeze().to(device), labels.squeeze().to(device)

                outputs, _, aux = net(index, value)     # outputs: (B, E_max)
                B, E_max = outputs.shape
                E = labels.shape[1]
                if E < E_max:
                    pad = torch.full((B, E_max - E), -1, dtype=labels.dtype, device=labels.device)
                    labels = torch.cat([labels, pad], dim=1)

                # print(outputs.shape, (labels != -1).shape, aux["example_mask"].shape)
                mask = aux["example_mask"] & (labels != -1)
                loss = criterion(outputs, labels.float(), mask)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                l_value = loss.item()

                n_iter += 1

                eval_fre = args.eval_frequency
                losses.append(l_value)
                if n_iter%eval_fre==0:
                    net.eval()
                    with torch.no_grad():
                        test_score_sythetic_ind = valid.get_avg_score_sythetic(
                            net)
                    writer.add_scalar('score/test_syth_acc_ind',
                                        test_score_sythetic_ind, n_iter)
                    val_accs.append(test_score_sythetic_ind)
                    l_avg = np.mean(losses[-1000:])
                    writer.add_scalar('score/train_loss_iter',l_avg, n_iter)
                    
                    # Log to wandb
                    wandb.log({
                        "train/loss": l_avg,
                        "train/synthetic_val_acc": test_score_sythetic_ind,
                        "train/iteration": n_iter
                    })
                    if l_avg< min_loss:
                        min_loss = l_avg
                        n_not_improved = 0
                    else:
                        n_not_improved+=eval_fre
                    net.train()
            except Exception as e:
                print(e)
                continue
        if not os.path.exists("model_checkpoints"): 
            os.mkdir("model_checkpoints")

        # Evaluate on datasets at the end of each epoch
        print(f"\nEvaluating at end of epoch {epoch}...")
        eval_results = evaluate_on_datasets(net, epoch, i_run)
        
        torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg':np.mean(val_accs),
            },
            "model_checkpoints/stacked_bag_model_transformer_" + str(i_run) + ".pt"
            )
    
    # Finish wandb run
    wandb.finish()

#select the best run
val_accs = []
for i in range(NUM_RUNS): 
    checkpoint = torch.load("model_checkpoints/stacked_bag_model_transformer_"+str(i)+".pt", map_location="cpu")
    val_accs.append(checkpoint['val_acc_avg'])
best_run = np.argmax(val_accs)
print('Please use the checkpoint:', "stacked_bag_model_transformer_" + str(best_run) + ".pt")
