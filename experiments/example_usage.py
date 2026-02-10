"""
Example script showing how to use the modified gully_performance_exp.py
with your own labeling function results and ground truth.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gully_performance_exp import run_label_aggregation_methods, print_results_table


def example_with_synthetic_data():
    """
    Example using synthetic data to demonstrate the interface.
    """
    print("="*80)
    print("EXAMPLE: Running label aggregation methods with synthetic data")
    print("="*80)
    
    # Create synthetic data
    n_samples = 1000
    n_labeling_functions = 5
    n_classes = 2
    
    # Generate random labeling function outputs
    # -1 = abstain, 0 = class 0, 1 = class 1
    X = np.random.choice([-1, 0, 1], size=(n_samples, n_labeling_functions), p=[0.2, 0.4, 0.4])
    
    # Generate ground truth (with some correlation to labeling functions)
    y = np.random.choice([0, 1], size=n_samples)
    
    print(f"\nData shape: {X.shape}")
    print(f"Ground truth shape: {y.shape}")
    print(f"Abstention rate: {np.mean(X == -1):.2%}")
    
    # Run methods
    results, times = run_label_aggregation_methods(
        X=X,
        y=y,
        dataset_name="synthetic_example",
        use_f1=False,  # Use accuracy
        lela_checkpoint=None  # Skip LELA for this example (set path if available)
    )
    
    # Print results table
    print_results_table({"synthetic_example": results}, {"synthetic_example": times})


def example_with_your_data():
    """
    Template for using your own data.
    Replace the data loading section with your actual data.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Template for your own data")
    print("="*80)
    
    # =========================================================================
    # REPLACE THIS SECTION WITH YOUR DATA LOADING CODE
    # =========================================================================
    
    # Option 1: Load from numpy files
    # X = np.load('path/to/your/label_matrix.npy')
    # y = np.load('path/to/your/ground_truth.npy')
    
    # Option 2: Load from CSV
    # import pandas as pd
    # X = pd.read_csv('path/to/your/label_matrix.csv').values
    # y = pd.read_csv('path/to/your/ground_truth.csv').values.flatten()
    
    # Option 3: Create from your existing variables
    # X = your_label_matrix  # shape: (n_samples, n_labeling_functions)
    # y = your_ground_truth  # shape: (n_samples,)
    
    # For this example, we'll use synthetic data
    X = np.random.choice([-1, 0, 1], size=(500, 10), p=[0.3, 0.35, 0.35])
    y = np.random.choice([0, 1], size=500)
    
    # =========================================================================
    # END OF DATA LOADING SECTION
    # =========================================================================
    
    print(f"\nYour data:")
    print(f"  Label matrix shape: {X.shape}")
    print(f"  Ground truth shape: {y.shape}")
    print(f"  Number of labeling functions: {X.shape[1]}")
    print(f"  Number of samples: {X.shape[0]}")
    print(f"  Number of classes: {len(np.unique(y[y >= 0]))}")
    
    # Run methods
    results, times = run_label_aggregation_methods(
        X=X,
        y=y,
        dataset_name="my_dataset",
        use_f1=False,  # Set to True for F1 score instead of accuracy
        lela_checkpoint="./lela_checkpoint.pt"  # Set to None if LELA not available
    )
    
    # Print results
    print_results_table({"my_dataset": results}, {"my_dataset": times})
    
    # Optionally save results
    import pandas as pd
    results_df = pd.DataFrame([results])
    results_df.to_csv("my_results.csv", index=False)
    print("\nResults saved to my_results.csv")


def example_with_multiple_datasets():
    """
    Example showing how to run on multiple datasets and compare results.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Running on multiple datasets")
    print("="*80)
    
    all_results = {}
    all_times = {}
    
    # Simulate multiple datasets
    datasets = ["dataset_A", "dataset_B", "dataset_C"]
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        
        # Generate synthetic data (replace with your actual data loading)
        X = np.random.choice([-1, 0, 1], size=(200, 8), p=[0.25, 0.375, 0.375])
        y = np.random.choice([0, 1], size=200)
        
        # Run methods
        results, times = run_label_aggregation_methods(
            X=X,
            y=y,
            dataset_name=dataset_name,
            use_f1=False,
            lela_checkpoint=None  # Skip LELA for speed
        )
        
        all_results[dataset_name] = results
        all_times[dataset_name] = times
    
    # Print combined results table
    print("\n" + "="*80)
    print("COMBINED RESULTS")
    print("="*80)
    print_results_table(all_results, all_times)


if __name__ == "__main__":
    # Run examples
    example_with_synthetic_data()
    
    # Uncomment to run other examples:
    # example_with_your_data()
    # example_with_multiple_datasets()
