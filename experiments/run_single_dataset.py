"""
Simplified script for running label aggregation methods on a single custom dataset.
Just provide your labeling function results (X) and ground truth (y).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import *
from model import LELAWrapper


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


def run_methods_and_get_results(
    X: np.ndarray,
    y: np.ndarray,
    use_lela: bool = True,
    lela_checkpoint: str = "./lela_checkpoint.pt"
):
    """
    Run all label aggregation methods on your data and return results table.
    
    Parameters:
    -----------
    X : np.ndarray
        Label matrix of shape (n_samples, n_labeling_functions)
        Values: -1 (abstain), 0, 1, 2, ... (class labels)
        
    y : np.ndarray
        Ground truth labels of shape (n_samples,)
        Values: 0, 1, 2, ... (class labels)
        
    use_lela : bool, default=True
        Whether to include LELA method
        
    lela_checkpoint : str
        Path to LELA checkpoint file
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with methods as rows and metrics (Accuracy, F1, Precision, Recall, NPV) as columns
    """
    
    print("="*80)
    print("Running Label Aggregation Methods")
    print("="*80)
    print(f"\nInput data shape: {X.shape}")
    print(f"Ground truth shape: {y.shape}")
    print(f"Number of labeling functions: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    num_classes = len(np.unique(y[y >= 0]))
    print(f"Number of classes: {num_classes}")
    print(f"Abstention rate: {np.mean(X == -1):.2%}")
    print(f"Metrics: Accuracy, F1, Precision, Recall, NPV")
    
    # Remove columns (LFs) with all abstentions
    non_abstain_cols = np.sum(X != -1, axis=0) > 0
    X_filtered = X[:, non_abstain_cols]
    print(f"\nAfter removing all-abstain LFs: {X_filtered.shape[1]} LFs remain")
    
    # Remove rows (samples) with all abstentions
    non_abstain_rows = np.sum(X_filtered != -1, axis=1) > 0
    X_filtered = X_filtered[non_abstain_rows, :]
    y_filtered = y[non_abstain_rows]
    print(f"After removing all-abstain samples: {X_filtered.shape[0]} samples remain")
    
    # Initialize LELA if requested
    lela = None
    if use_lela:
        try:
            lela = LELAWrapper(checkpoint_path=lela_checkpoint)
            print(f"✓ LELA loaded from {lela_checkpoint}")
        except Exception as e:
            print(f"⚠ Could not load LELA: {e}")
            use_lela = False
    
    # Define methods to run
    methods = [
        dawid_skene,
        majority_vote,
        Metal,
        flying_squid,
        data_programming,
        ebcc,
        NPLM
    ]
    
    if use_lela and lela is not None:
        methods.insert(2, lela)  # Insert LELA after majority_vote
    
    # Run methods
    results = {
        'Method': [],
        'Accuracy': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'NPV': [],
        'Time (s)': []
    }
    
    # Determine if binary or multiclass
    is_binary = num_classes == 2
    
    print("\n" + "="*80)
    print("Running Methods...")
    print("="*80)
    
    y_ind = np.arange(0, X_filtered.shape[0])
    
    for method in methods:
        try:
            if isinstance(method, LELAWrapper):
                method_name = "LELA"
            else:
                method_name = method.__name__
            
            print(f"\nRunning {method_name}...", end=" ", flush=True)
            t_start = time.time()
            
            # Get predictions
            if isinstance(method, LELAWrapper):
                pred_labels = method.predict(X_filtered)
            else:
                pred_probs = method(X_filtered)
                pred_labels = np.argmax(pred_probs, axis=1).flatten()
            
            t_elapsed = time.time() - t_start
            
            # Evaluate on samples with ground truth
            gt_labels = np.copy(y_filtered).flatten().astype(int)
            valid_indices = y_ind[gt_labels >= 0]
            gt_labels_valid = gt_labels[gt_labels >= 0]
            pred_labels_valid = pred_labels[valid_indices]
            
            # Calculate all metrics
            acc = accuracy_score(gt_labels_valid, pred_labels_valid)
            
            # For F1, Precision, Recall: use 'binary' for binary classification, 'macro' for multiclass
            avg_type = 'binary' if is_binary else 'macro'
            f1 = f1_score(gt_labels_valid, pred_labels_valid, average=avg_type, zero_division=0)
            prec = precision_score(gt_labels_valid, pred_labels_valid, average=avg_type, zero_division=0)
            rec = recall_score(gt_labels_valid, pred_labels_valid, average=avg_type, zero_division=0)
            npv = calculate_npv(gt_labels_valid, pred_labels_valid)
            
            # Store results
            results['Method'].append(method_name)
            results['Accuracy'].append(acc)
            results['F1'].append(f1)
            results['Precision'].append(prec)
            results['Recall'].append(rec)
            results['NPV'].append(npv)
            results['Time (s)'].append(t_elapsed)
            
            print(f"✓ Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, NPV: {npv:.4f}, Time: {t_elapsed:.3f}s")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            method_name = method.__name__ if hasattr(method, '__name__') else 'Unknown'
            results['Method'].append(method_name)
            results['Accuracy'].append(np.nan)
            results['F1'].append(np.nan)
            results['Precision'].append(np.nan)
            results['Recall'].append(np.nan)
            results['NPV'].append(np.nan)
            results['Time (s)'].append(np.nan)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1 score (descending)
    results_df = results_df.sort_values('F1', ascending=False).reset_index(drop=True)
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("\n" + "="*80)
    
    return results_df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    """
    Replace this section with your actual data loading code.
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Load your data
    # -------------------------------------------------------------------------
    
    # Option A: Load from numpy files
    X = np.load('gully_dev_set.npy')
    y = np.load('gully_gt_set.npy')
    
    # Option B: Load from CSV files
    # import pandas as pd
    # X = pd.read_csv('path/to/your/label_matrix.csv').values
    # y = pd.read_csv('path/to/your/ground_truth.csv').values.flatten()
    
    # Option C: Use your existing variables
    # X = your_label_matrix
    # y = your_ground_truth
    
    # For demonstration, using synthetic data:
    # print("NOTE: Using synthetic data for demonstration.")
    # print("Replace this with your actual data loading code.\n")
    
    # np.random.seed(42)
    # X = np.random.choice([-1, 0, 1], size=(1000, 10), p=[0.2, 0.4, 0.4])
    # y = np.random.choice([0, 1], size=1000)
    
    # -------------------------------------------------------------------------
    # STEP 2: Run methods and get results
    # -------------------------------------------------------------------------
    
    results_df = run_methods_and_get_results(
        X=X,
        y=y,
        use_lela=True,  # Set to False to skip LELA
        lela_checkpoint="../lela_checkpoint.pt"
    )
    
    # -------------------------------------------------------------------------
    # STEP 3: Save results (optional)
    # -------------------------------------------------------------------------
    
    # Save to CSV
    output_file = "label_aggregation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
