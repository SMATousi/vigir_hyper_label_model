import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import time
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from baselines.baselines import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LELAWrapper
from data import load_dataset_wrench


def calculate_npv(y_true, y_pred):
    """
    Calculate Negative Predictive Value (NPV).
    NPV = TN / (TN + FN)
    """
    cm = confusion_matrix(y_true, y_pred)
    # For binary classification
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fn = cm[1, 0]
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return npv
    else:
        # For multiclass, calculate NPV for each class and average
        npvs = []
        for i in range(cm.shape[0]):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fn = np.sum(cm[i, :]) - cm[i, i]
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            npvs.append(npv)
        return np.mean(npvs)


def run_label_aggregation_methods(
    X: np.ndarray, 
    y: np.ndarray, 
    dataset_name: str = "custom",
    use_f1: bool = False,
    lela_checkpoint: Optional[str] = "./lela_checkpoint.pt",
    methods_to_run: Optional[List] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Run label aggregation methods on provided labeling function results and ground truth.
    
    Args:
        X: Label matrix of shape (n_samples, n_labeling_functions) with values in {-1, 0, 1, ...}
           where -1 indicates abstention
        y: Ground truth labels of shape (n_samples,) with values in {0, 1, ...} or -1 for no ground truth
        dataset_name: Name of the dataset (for display purposes)
        use_f1: If True, use F1 score; otherwise use accuracy
        lela_checkpoint: Path to LELA checkpoint file (None to skip LELA)
        methods_to_run: List of methods to run. If None, runs all methods.
    
    Returns:
        results: Dictionary mapping method names to dictionaries of metrics
                 (Accuracy, F1, Precision, Recall, NPV)
        times: Dictionary mapping method names to running times
    """
    # Initialize LELA if checkpoint provided
    lela = None
    if lela_checkpoint is not None:
        try:
            lela = LELAWrapper(checkpoint_path=lela_checkpoint)
        except Exception as e:
            print(f"Warning: Could not load LELA model: {e}")
    
    # Default methods
    all_methods = [dawid_skene, majority_vote, Metal, flying_squid, data_programming, ebcc, NPLM]
    if lela is not None:
        all_methods.insert(2, lela)  # Insert LELA after majority_vote
    
    if methods_to_run is None:
        methods_to_run = all_methods
    
    # Remove rows and columns with all abstentions
    non_abstain_cols = np.sum(X != -1, axis=0) > 0
    X_filtered = X[:, non_abstain_cols]
    non_abstain_rows = np.sum(X_filtered != -1, axis=1) > 0
    X_filtered = X_filtered[non_abstain_rows, :]
    y_filtered = y[non_abstain_rows]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Shape after filtering: {X_filtered.shape}")
    print(f"Number of samples with ground truth: {np.sum(y_filtered >= 0)}")
    
    num_classes = len(np.unique(y_filtered[y_filtered >= 0]))
    is_binary = num_classes == 2
    
    results = {}
    times = {}
    
    y_ind = np.arange(0, X_filtered.shape[0])
    
    for method in methods_to_run:
        try:
            t_s = time.time()
            
            if isinstance(method, LELAWrapper):
                method_name = "LELA"
                pred_round = method.predict(X_filtered)
            else:
                method_name = method.__name__
                pred = method(X_filtered)
                pred_round = np.argmax(pred, axis=1).flatten()
            
            delta_t = time.time() - t_s
            gt_labels = np.copy(y_filtered).flatten().astype(int)
            
            # Only use data points with non-abstention gt labels to evaluate
            y_ind_ = y_ind[gt_labels >= 0]
            gt_labels_eval = gt_labels[gt_labels >= 0]
            pred_round_eval = pred_round[y_ind_]
            
            if len(gt_labels_eval) == 0:
                print(f"{method_name}: No ground truth labels available")
                continue
            
            # Calculate all metrics
            acc = accuracy_score(gt_labels_eval, pred_round_eval)
            
            # For F1, Precision, Recall: use 'binary' for binary, 'macro' for multiclass
            avg_type = 'binary' if is_binary else 'macro'
            f1 = f1_score(gt_labels_eval, pred_round_eval, average=avg_type, zero_division=0)
            prec = precision_score(gt_labels_eval, pred_round_eval, average=avg_type, zero_division=0)
            rec = recall_score(gt_labels_eval, pred_round_eval, average=avg_type, zero_division=0)
            npv = calculate_npv(gt_labels_eval, pred_round_eval)
            
            # Store all metrics
            results[method_name] = {
                'Accuracy': acc,
                'F1': f1,
                'Precision': prec,
                'Recall': rec,
                'NPV': npv
            }
            times[method_name] = delta_t
            
            print(f"{method_name}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, NPV={npv:.4f} (time: {delta_t:.3f}s)")
        
        except Exception as e:
            print(f"{method_name if 'method_name' in locals() else 'Unknown'}: Error - {e}")
    
    return results, times


def print_results_table(results_dict: Dict[str, Dict[str, Dict[str, float]]], 
                       times_dict: Optional[Dict[str, Dict[str, float]]] = None):
    """
    Print results in a formatted table.
    
    Args:
        results_dict: Dictionary mapping dataset names to method results
                     (which are dicts of metrics)
        times_dict: Optional dictionary mapping dataset names to method times
    """
    # Flatten the nested dictionary structure for display
    # Convert {dataset: {method: {metric: value}}} to separate tables per metric
    
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'NPV']
    
    for metric in metrics:
        metric_data = {}
        for dataset, methods in results_dict.items():
            metric_data[dataset] = {method: vals[metric] for method, vals in methods.items()}
        
        df = pd.DataFrame.from_dict(metric_data, orient='index')
        
        print("\n" + "="*80)
        print(f"{metric.upper()} SCORES")
        print("="*80)
        print(df.to_string())
        print(f"\nMean {metric}:")
        print(df.mean().to_string())
    
    if times_dict is not None:
        time_df = pd.DataFrame.from_dict(times_dict, orient='index')
        print("\n" + "="*80)
        print("RUNNING TIMES (seconds)")
        print("="*80)
        print(time_df.to_string())
        print("\nMean Time:")
        print(time_df.mean().to_string())
    
    # Return the first metric's dataframe for compatibility
    return pd.DataFrame.from_dict(
        {dataset: {method: vals['F1'] for method, vals in methods.items()} 
         for dataset, methods in results_dict.items()}, 
        orient='index'
    )


# =============================================================================
# MAIN EXECUTION - Run on multiple datasets from WRENCH
# =============================================================================
def run_wrench_experiments(run_experiments: bool = False):
    """
    Run experiments on WRENCH datasets. Set run_experiments=True to execute.
    """
    if not run_experiments:
        print("Skipping WRENCH experiments. Set run_experiments=True to run.")
        return
    
    lela = LELAWrapper(checkpoint_path="./lela_checkpoint.pt") #load pretrained LELA model
    
    NOISE_POWER = 0.0  # 10% chance of flipping each non-zero label
    
    datasets = [
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
    
    # Datasets that use F1 score instead of accuracy
    f1_datasets = ["sms", "spouse", 'cdr', 'commercial', 'tennis', 'basketball', 'census']
    
    # dicts to save performance scores and running times 
    all_results = {}
    all_times = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset}")
        print(f"{'='*80}")
        
        X, y = load_dataset_wrench("datasets/" + dataset)
        
        # Apply label flipping noise if needed
        if NOISE_POWER > 0:
            print(f"Applying label flipping noise with probability {NOISE_POWER}")
            X_noisy = X.copy()
            non_abstain_mask = X != -1
            flip_probs = np.random.random(X.shape)
            flip_mask = (flip_probs < NOISE_POWER) & non_abstain_mask
            X_noisy[flip_mask] = 1 - X_noisy[flip_mask]
            num_flipped = np.sum(flip_mask)
            total_non_abstain = np.sum(non_abstain_mask)
            print(f"Flipped {num_flipped}/{total_non_abstain} labels ({num_flipped/total_non_abstain*100:.1f}%)")
            X = X_noisy
        
        # Run methods
        use_f1 = dataset in f1_datasets
        results, times = run_label_aggregation_methods(
            X, y, 
            dataset_name=dataset,
            use_f1=use_f1,
            lela_checkpoint="./lela_checkpoint.pt"
        )
        
        all_results[dataset] = results
        all_times[dataset] = times
    
    # Print and save results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    results_df = print_results_table(all_results, all_times)
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/performance_exp_acc.csv")
    time_df = pd.DataFrame.from_dict(all_times, orient='index')
    time_df.to_csv("results/performance_exp_time.csv")
    print("\nResults saved to results/performance_exp_acc.csv and results/performance_exp_time.csv")


# =============================================================================
# USAGE - Run with your own single dataset
# =============================================================================
if __name__ == "__main__":
    # RECOMMENDED: Use run_single_dataset.py for a simpler interface
    # Or uncomment the code below to use this script directly
    
    """
    # Load your data
    # X: Label matrix, shape (n_samples, n_labeling_functions)
    #    Values: -1 (abstain), 0, 1, 2, ... (class labels)
    # y: Ground truth, shape (n_samples,)
    #    Values: 0, 1, 2, ... (class labels)
    
    import numpy as np
    
    # Replace with your data loading code:
    X = np.load('your_label_matrix.npy')
    y = np.load('your_ground_truth.npy')
    
    # Run all methods
    results, times = run_label_aggregation_methods(
        X=X,
        y=y,
        dataset_name="my_dataset",
        use_f1=False,  # Set True for F1 score (good for imbalanced data)
        lela_checkpoint="./lela_checkpoint.pt"  # Set None to skip LELA
    )
    
    # Display results table
    print_results_table({"my_dataset": results}, {"my_dataset": times})
    
    # Optional: Save results
    import pandas as pd
    pd.DataFrame([results]).to_csv("my_results.csv", index=False)
    """
    
    # To run original WRENCH experiments, set run_experiments=True
    run_wrench_experiments(run_experiments=False)
