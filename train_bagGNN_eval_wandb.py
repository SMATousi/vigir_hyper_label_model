import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import DatasetOnlineGen
from bag_attention_gnn import StackedBagAttentionGNN, BagAttentionGNNModelWrapper, BagAttentionGNNWrapper
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

torch.autograd.set_detect_anomaly(True)

# Regularization functions
def attention_entropy_loss(attention_weights, valid_mask=None):
    """
    Compute entropy penalty to encourage peaked attention.
    Lower entropy = more peaked attention.
    
    Args:
        attention_weights: (num_segs, maxL) attention weights after softmax
        valid_mask: (num_segs, maxL) mask for valid positions
    
    Returns:
        entropy: scalar entropy loss (negative for peaked attention)
    """
    if valid_mask is not None:
        # Only compute entropy over valid positions
        attention_weights = attention_weights * valid_mask
        # Renormalize
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute entropy: H(p) = -sum(p * log(p))
    log_attn = torch.log(attention_weights + 1e-8)
    entropy = -(attention_weights * log_attn).sum(dim=-1)  # (num_segs,)
    return entropy.mean()  # Average over all segments

def attention_l1_sparsity_loss(attention_weights, valid_mask=None):
    """
    Compute L1 sparsity penalty on attention weights.
    
    Args:
        attention_weights: (num_segs, maxL) attention weights after softmax
        valid_mask: (num_segs, maxL) mask for valid positions
    
    Returns:
        l1_loss: scalar L1 sparsity loss
    """
    if valid_mask is not None:
        attention_weights = attention_weights * valid_mask
    
    return attention_weights.abs().sum() / (valid_mask.sum() if valid_mask is not None else attention_weights.numel())

def labeler_dropout(value, dropout_rate=0.2, training=True):
    """
    Apply labeler dropout - randomly set entire labeler channels to abstain (-1).
    
    Args:
        value: (B, S) input values
        dropout_rate: probability of dropping each labeler
        training: whether in training mode
    
    Returns:
        value_dropped: (B, S) values with some labelers dropped
    """
    if not training or dropout_rate == 0.0:
        return value
    
    # Create dropout mask - same for all examples in batch
    B, S = value.shape
    dropout_mask = torch.rand(S, device=value.device) > dropout_rate  # (S,)
    dropout_mask = dropout_mask.unsqueeze(0).expand(B, -1)  # (B, S)
    
    # Apply dropout by setting dropped positions to -1 (abstain)
    value_dropped = value.clone()
    value_dropped[~dropout_mask] = -1
    
    return value_dropped

class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross-Entropy Loss for robust training with noisy labels.
    SCE = alpha * CE + beta * RCE
    where RCE is Reverse Cross-Entropy: -sum(y_pred * log(y_true))
    """
    def __init__(self, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, outputs, targets, mask):
        """
        Args:
            outputs: (B, E_max) predicted probabilities
            targets: (B, E_max) target labels (0 or 1)
            mask: (B, E_max) valid position mask
        
        Returns:
            loss: scalar SCE loss
        """
        # Apply mask
        outputs_masked = outputs[mask]
        targets_masked = targets[mask]
        
        if outputs_masked.numel() == 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        # Clamp to avoid log(0)
        outputs_clamped = torch.clamp(outputs_masked, 1e-7, 1.0 - 1e-7)
        targets_clamped = torch.clamp(targets_masked, 1e-7, 1.0 - 1e-7)
        
        # Standard Cross-Entropy: -sum(y_true * log(y_pred))
        ce_loss = -(targets_clamped * torch.log(outputs_clamped) + 
                   (1 - targets_clamped) * torch.log(1 - outputs_clamped))
        
        # Reverse Cross-Entropy: -sum(y_pred * log(y_true))
        rce_loss = -(outputs_clamped * torch.log(targets_clamped) + 
                    (1 - outputs_clamped) * torch.log(1 - targets_clamped))
        
        # Symmetric combination
        sce_loss = self.alpha * ce_loss + self.beta * rce_loss
        
        return sce_loss.mean()

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
    parser.add_argument('--eval_frequency', type=int, default=1, help='Evaluation frequency in epochs (default: 1 - evaluate every epoch)')
    parser.add_argument('--project_name', type=str, default='lela-transformer-training', help='Wandb project name')
    parser.add_argument('--log-wandb', type=bool, default=False, help='Log to wandb (default: False)')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers (default: 2)')
    
    # Regularization hyperparameters
    parser.add_argument('--lambda_entropy', type=float, default=0.01, help='Attention entropy regularization weight (default: 0.01)')
    parser.add_argument('--lambda_l1', type=float, default=0.001, help='L1 sparsity on attention weights (default: 0.001)')
    parser.add_argument('--labeler_dropout_rate', type=float, default=0.2, help='Labeler dropout rate (default: 0.2)')
    parser.add_argument('--alpha_sce', type=float, default=0.1, help='Alpha parameter for Symmetric Cross-Entropy (default: 0.1)')
    parser.add_argument('--beta_sce', type=float, default=1.0, help='Beta parameter for Symmetric Cross-Entropy (default: 1.0)')
    parser.add_argument('--use_sce_loss', action='store_true', help='Use Symmetric Cross-Entropy loss instead of BCE')
    
    return parser.parse_args()

args = parse_args()
NUM_RUNS = args.num_runs
LOG_WANDB = args.log_wandb
log_wandb = args.log_wandb

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

def evaluate_on_datasets(model, epoch, run_id, log_wandb, num_gnn_layers=2, num_layers=2):
    """Evaluate model on all datasets with noise power = 0 and log to wandb"""
    model.eval()
    eval_results = {}
    
    # Create a temporary wrapper for evaluation
    temp_checkpoint_path = f"temp_eval_checkpoint_run_{run_id}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
    }, temp_checkpoint_path)
    
    # Create wrapper for evaluation with matching architecture
    eval_wrapper = BagAttentionGNNWrapper(
        checkpoint_path=temp_checkpoint_path, 
        max_lf_id=0, 
        num_gnn_layers=num_gnn_layers,  # Match training model architecture
        num_layers=num_layers,
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
    
    # Calculate overall score for model selection
    overall_score = -1.0
    if eval_results:
        # Calculate and log average scores
        f1_scores = [v for k, v in eval_results.items() if k.endswith('_f1')]
        acc_scores = [v for k, v in eval_results.items() if k.endswith('_accuracy')]
        
        if f1_scores or acc_scores:
            all_scores = f1_scores + acc_scores
            overall_score = np.mean(all_scores)
            eval_results['overall_score'] = overall_score
        
        if log_wandb:
            # Log to wandb
            wandb.log({**eval_results, "epoch": epoch})
            
            if f1_scores:
                wandb.log({"eval/avg_f1": np.mean(f1_scores), "epoch": epoch})
            if acc_scores:
                wandb.log({"eval/avg_accuracy": np.mean(acc_scores), "epoch": epoch})
            if f1_scores or acc_scores:
                wandb.log({"eval/avg_overall": overall_score, "epoch": epoch})
    
    model.train()
    return eval_results
for i_run in range(NUM_RUNS): #train LELA model with configurable parameters
    # Initialize wandb for this run
    if LOG_WANDB:
        wandb.init(
            project=args.project_name,
            name=f"run_{i_run}",
            config={
                "run_id": i_run,
                "num_epochs": args.num_epochs,
                "data_size": args.data_size,
                "max_n_lfs": args.max_n_lfs,
                "max_examples": args.max_examples,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "num_layers": args.num_layers,
                "max_seq_len": args.max_seq_len,
                "eval_frequency": args.eval_frequency,
                "num_gnn_layers": args.num_gnn_layers,
                # Regularization hyperparameters
                "lambda_entropy": args.lambda_entropy,
                "lambda_l1": args.lambda_l1,
                "labeler_dropout_rate": args.labeler_dropout_rate,
                "alpha_sce": args.alpha_sce,
                "beta_sce": args.beta_sce,
                "use_sce_loss": args.use_sce_loss
            }
        )
    
    # Track best model for this run
    best_overall_score = -1.0
    best_model_checkpoint = None
    best_epoch = 0
    
    device = "cuda:0"
    # Use sequence length limit to prevent memory explosion
    max_seq_len = args.max_seq_len  # Configurable via command line
    net = StackedBagAttentionGNN(num_gnn_layers=args.num_gnn_layers, num_layers=args.num_layers)

    optimizer = optim.Adam(
        net.parameters(),
        amsgrad=True,
    )
    net.to(device)

    # Initialize loss function based on arguments
    if args.use_sce_loss:
        criterion = SymmetricCrossEntropyLoss(alpha=args.alpha_sce, beta=args.beta_sce)
        print(f"Using Symmetric Cross-Entropy Loss with alpha={args.alpha_sce}, beta={args.beta_sce}")
    else:
        criterion = BCEMask()
        print("Using standard BCE Loss")

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

                # Apply labeler dropout during training
                if net.training:
                    value = labeler_dropout(value, dropout_rate=args.labeler_dropout_rate, training=True)

                outputs, _, aux = net(index, value)     # outputs: (B, E_max)
                # print("outputs.shape", outputs.shape)
                B, E_max = outputs.shape
                E = labels.shape[1]
                if E < E_max:
                    pad = torch.full((B, E_max - E), -1, dtype=labels.dtype, device=labels.device)
                    labels = torch.cat([labels, pad], dim=1)

                # print(outputs.shape, (labels != -1).shape, aux["example_mask"].shape)
                mask = aux["example_mask"] & (labels != -1)
                # print("mask.shape", mask.shape)
                
                # Compute main loss
                main_loss = criterion(outputs, labels.float(), mask)
                
                # Compute regularization losses
                reg_loss = 0.0
                
                # Attention regularization (from all bag attention layers)
                if "all_bag_aux" in aux and aux["all_bag_aux"]:
                    total_entropy_loss = 0.0
                    total_l1_loss = 0.0
                    num_layers = 0
                    
                    for bag_aux in aux["all_bag_aux"]:
                        if bag_aux and "attention_weights" in bag_aux and "valid_masks" in bag_aux:
                            attention_weights = bag_aux["attention_weights"]
                            valid_masks = bag_aux["valid_masks"]
                            
                            # Compute regularization for each batch in this layer
                            for attn_weights, valid_mask in zip(attention_weights, valid_masks):
                                if attn_weights.numel() > 0:
                                    # Entropy regularization (encourage peaked attention)
                                    entropy_loss = attention_entropy_loss(attn_weights, valid_mask)
                                    total_entropy_loss += entropy_loss
                                    
                                    # L1 sparsity regularization
                                    l1_loss = attention_l1_sparsity_loss(attn_weights, valid_mask)
                                    total_l1_loss += l1_loss
                                    
                                    num_layers += 1
                    
                    if num_layers > 0:
                        # Average over all layers and add to regularization
                        avg_entropy_loss = total_entropy_loss / num_layers
                        avg_l1_loss = total_l1_loss / num_layers
                        
                        reg_loss += args.lambda_entropy * avg_entropy_loss
                        reg_loss += args.lambda_l1 * avg_l1_loss
                
                # Total loss
                loss = main_loss + reg_loss
                # print("loss.shape", loss.shape)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                l_value = loss.item()
                main_loss_value = main_loss.item() if 'main_loss' in locals() else l_value
                reg_loss_value = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0.0
                entropy_loss_value = avg_entropy_loss.item() if 'avg_entropy_loss' in locals() else 0.0
                l1_loss_value = avg_l1_loss.item() if 'avg_l1_loss' in locals() else 0.0

                n_iter += 1

                eval_fre = args.eval_frequency
                losses.append(l_value)
                if n_iter%eval_fre==0:
                    net.eval()
                    with torch.no_grad():
                        # Wrap model for data.py compatibility (returns 2 values instead of 3)
                        wrapped_net = BagAttentionGNNModelWrapper(net)
                        test_score_sythetic_ind = valid.get_avg_score_sythetic(
                            wrapped_net)
                    writer.add_scalar('score/test_syth_acc_ind',
                                        test_score_sythetic_ind, n_iter)
                    val_accs.append(test_score_sythetic_ind)
                    l_avg = np.mean(losses[-1000:])
                    writer.add_scalar('score/train_loss_iter',l_avg, n_iter)
                    
                    if LOG_WANDB:
                        # Log to wandb
                        log_dict = {
                            "train/loss": l_avg,
                            "train/main_loss": main_loss_value,
                            "train/reg_loss": reg_loss_value,
                            "train/entropy_loss": entropy_loss_value,
                            "train/l1_loss": l1_loss_value,
                            "train/synthetic_val_acc": test_score_sythetic_ind,
                            "train/iteration": n_iter
                        }
                        
                        wandb.log(log_dict)
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

        # Evaluate on datasets at specified epoch frequency
        eval_results = {}
        current_overall_score = -1.0
        
        if epoch % args.eval_frequency == 0:
            print(f"\nEvaluating at end of epoch {epoch} (evaluation frequency: {args.eval_frequency})...")
            eval_results = evaluate_on_datasets(net, epoch, i_run, log_wandb, args.num_gnn_layers, args.num_layers)
            current_overall_score = eval_results.get('overall_score', -1.0)
            
            # Check if this is the best model so far for this run
            if current_overall_score > best_overall_score:
                best_overall_score = current_overall_score
                best_epoch = epoch
                # Save the best model checkpoint
                best_model_checkpoint = {
                    'n_iter': n_iter,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc_avg': np.mean(val_accs),
                    'eval_results': eval_results,
                    'epoch': epoch,
                    'overall_score': current_overall_score
                }
                print(f"New best model at epoch {epoch} with overall score: {current_overall_score:.4f}")
        else:
            print(f"Skipping evaluation at epoch {epoch} (next evaluation at epoch {((epoch // args.eval_frequency) + 1) * args.eval_frequency})")
        
        # Save regular checkpoint
        torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg':np.mean(val_accs),
                'eval_results': eval_results,
                'epoch': epoch,
                'overall_score': current_overall_score
            },
            "model_checkpoints/stacked_bag_model_transformer_" + str(i_run) + ".pt"
            )
    
    # Final evaluation if the last epoch wasn't evaluated
    if epoch % args.eval_frequency != 0:
        print(f"\nFinal evaluation at end of training (epoch {epoch})...")
        final_eval_results = evaluate_on_datasets(net, epoch, i_run, log_wandb, args.num_gnn_layers, args.num_layers)
        final_overall_score = final_eval_results.get('overall_score', -1.0)
        
        # Check if this final model is the best
        if final_overall_score > best_overall_score:
            best_overall_score = final_overall_score
            best_epoch = epoch
            best_model_checkpoint = {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg': np.mean(val_accs),
                'eval_results': final_eval_results,
                'epoch': epoch,
                'overall_score': final_overall_score
            }
            print(f"Final model is the best with overall score: {final_overall_score:.4f}")
        
        # Update the final checkpoint with evaluation results
        torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg': np.mean(val_accs),
                'eval_results': final_eval_results,
                'epoch': epoch,
                'overall_score': final_overall_score
            },
            "model_checkpoints/stacked_bag_model_transformer_" + str(i_run) + ".pt"
        )
    
    # Save and upload the best model for this run
    if best_model_checkpoint is not None:
        best_model_path = f"model_checkpoints/best_model_run_{i_run}.pt"
        torch.save(best_model_checkpoint, best_model_path)
        
        if LOG_WANDB:
            # Upload best model to wandb
            artifact = wandb.Artifact(
                name=f"best_model_run_{i_run}",
                type="model",
                description=f"Best model from run {i_run} at epoch {best_epoch} with overall score {best_overall_score:.4f}",
                metadata={
                    "run_id": i_run,
                    "best_epoch": best_epoch,
                    "best_overall_score": best_overall_score,
                    "eval_results": best_model_checkpoint['eval_results']
                }
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
        
        if LOG_WANDB:
            print(f"Uploaded best model from run {i_run} to wandb (epoch {best_epoch}, score: {best_overall_score:.4f})")
    
    # Finish wandb run
    if LOG_WANDB:
        wandb.finish()

#select the best run
val_accs = []
overall_scores = []
for i in range(NUM_RUNS): 
    checkpoint = torch.load("model_checkpoints/stacked_bag_model_transformer_"+str(i)+".pt", map_location="cpu")
    val_accs.append(checkpoint['val_acc_avg'])
    overall_scores.append(checkpoint.get('overall_score', -1.0))

# Select best run based on overall evaluation score
best_run = np.argmax(overall_scores)
best_overall_score = overall_scores[best_run]
print('Best run:', best_run, 'with overall score:', f"{best_overall_score:.4f}")
print('Please use the checkpoint:', "stacked_bag_model_transformer_" + str(best_run) + ".pt")

# Upload the overall best model to wandb
if LOG_WANDB:
    wandb.init(
        project=args.project_name,
        name="final_best_model",
        job_type="model_selection"
    )

    # Load the best model from the best run
    best_model_path = f"model_checkpoints/best_model_run_{best_run}.pt"
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location="cpu")
        
        # Create final best model artifact
        final_artifact = wandb.Artifact(
            name="final_best_model",
            type="model",
            description=f"Overall best model from run {best_run} with score {best_overall_score:.4f}",
            metadata={
                "best_run_id": best_run,
                "best_epoch": best_checkpoint.get('epoch', 0),
                "best_overall_score": best_overall_score,
                "all_run_scores": overall_scores,
                "eval_results": best_checkpoint.get('eval_results', {})
            }
        )
        final_artifact.add_file(best_model_path)
        wandb.log_artifact(final_artifact)
        
        print(f"Uploaded final best model to wandb (run {best_run}, score: {best_overall_score:.4f})")
    else:
        print(f"Warning: Best model file not found at {best_model_path}")

    wandb.finish()
