"""
Simplified script for running label aggregation methods on a single custom dataset.
Just provide your labeling function results (X) and ground truth (y).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baselines import *
from model import LELAWrapper


def run_methods_and_get_results(
    X: np.ndarray,
    y: np.ndarray,
    use_f1: bool = False,
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
        
    use_f1 : bool, default=False
        If True, use F1 score; otherwise use accuracy
        
    use_lela : bool, default=True
        Whether to include LELA method
        
    lela_checkpoint : str
        Path to LELA checkpoint file
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with methods as columns and metrics as rows
    """
    
    print("="*80)
    print("Running Label Aggregation Methods")
    print("="*80)
    print(f"\nInput data shape: {X.shape}")
    print(f"Ground truth shape: {y.shape}")
    print(f"Number of labeling functions: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of classes: {len(np.unique(y[y >= 0]))}")
    print(f"Abstention rate: {np.mean(X == -1):.2%}")
    print(f"Metric: {'F1 Score' if use_f1 else 'Accuracy'}")
    
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
    results = {}
    times = {}
    
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
            
            # Calculate metric
            if use_f1:
                score = f1_score(gt_labels_valid, pred_labels_valid, average='binary' if len(np.unique(gt_labels_valid)) == 2 else 'macro')
            else:
                score = accuracy_score(gt_labels_valid, pred_labels_valid)
            
            results[method_name] = score
            times[method_name] = t_elapsed
            
            print(f"✓ Score: {score:.4f}, Time: {t_elapsed:.3f}s")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results[method_name] = np.nan
            times[method_name] = np.nan
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Method': list(results.keys()),
        'Score': list(results.values()),
        'Time (s)': list(times.values())
    })
    
    # Sort by score (descending)
    results_df = results_df.sort_values('Score', ascending=False).reset_index(drop=True)
    
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
        use_f1=False,  # Set to True if you want F1 score instead of accuracy
        use_lela=True,  # Set to False to skip LELA
        lela_checkpoint="./lela_checkpoint.pt"
    )
    
    # -------------------------------------------------------------------------
    # STEP 3: Save results (optional)
    # -------------------------------------------------------------------------
    
    # Save to CSV
    output_file = "label_aggregation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
