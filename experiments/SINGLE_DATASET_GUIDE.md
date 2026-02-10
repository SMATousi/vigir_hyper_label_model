# Guide: Running Methods on Your Single Dataset

This guide is for when you have **one dataset** with your own labeling function results and ground truth.

## Quick Start (3 Steps)

### Step 1: Prepare Your Data

You need two arrays:

**X - Label Matrix**
- Shape: `(n_samples, n_labeling_functions)`
- Values: `-1` = abstain, `0, 1, 2, ...` = class labels

**y - Ground Truth**
- Shape: `(n_samples,)`
- Values: `0, 1, 2, ...` = class labels

Example:
```python
import numpy as np

# 5 samples, 3 labeling functions
X = np.array([
    [0, 1, -1],   # Sample 0: LF0 says class 0, LF1 says class 1, LF2 abstains
    [1, 1, 1],    # Sample 1: All LFs say class 1
    [0, 0, 0],    # Sample 2: All LFs say class 0
    [-1, 1, 1],   # Sample 3: LF0 abstains, others say class 1
    [0, -1, 0],   # Sample 4: LF1 abstains, others say class 0
])

y = np.array([1, 1, 0, 1, 0])  # True labels
```

### Step 2: Load Your Data

Choose one method:

```python
# Method A: From numpy files
X = np.load('my_label_matrix.npy')
y = np.load('my_ground_truth.npy')

# Method B: From CSV files
import pandas as pd
X = pd.read_csv('my_labels.csv').values
y = pd.read_csv('my_ground_truth.csv').values.flatten()

# Method C: Already have arrays in memory
X = your_existing_label_matrix
y = your_existing_ground_truth
```

### Step 3: Run Methods

**Option A: Use the simple script (RECOMMENDED)**

```python
# Edit run_single_dataset.py and replace the data loading section
python run_single_dataset.py
```

**Option B: Use the main script**

```python
from gully_performance_exp import run_label_aggregation_methods, print_results_table

results, times = run_label_aggregation_methods(X, y)
print_results_table({"my_dataset": results}, {"my_dataset": times})
```

## Complete Example

```python
import numpy as np
from gully_performance_exp import run_label_aggregation_methods, print_results_table

# 1. Load your data
X = np.load('my_label_matrix.npy')  # Shape: (1000, 10)
y = np.load('my_ground_truth.npy')  # Shape: (1000,)

# 2. Run methods
results, times = run_label_aggregation_methods(
    X=X,
    y=y,
    dataset_name="my_dataset",
    use_f1=False,  # Change to True for F1 score
    lela_checkpoint="./lela_checkpoint.pt"  # or None to skip LELA
)

# 3. View results
print_results_table({"my_dataset": results}, {"my_dataset": times})

# 4. Save results (optional)
import pandas as pd
pd.DataFrame([results]).to_csv("my_results.csv", index=False)
```

## Output Example

```
================================================================================
Running Label Aggregation Methods
================================================================================

Input data shape: (1000, 10)
Ground truth shape: (1000,)
Number of labeling functions: 10
Number of samples: 1000
Number of classes: 2
Abstention rate: 20.00%
Metric: Accuracy

After removing all-abstain LFs: 10 LFs remain
After removing all-abstain samples: 1000 samples remain

================================================================================
Running Methods...
================================================================================

Running dawid_skene... ✓ Score: 0.8234, Time: 0.234s
Running majority_vote... ✓ Score: 0.7891, Time: 0.012s
Running LELA... ✓ Score: 0.8567, Time: 1.234s
Running Metal... ✓ Score: 0.8423, Time: 2.345s
Running flying_squid... ✓ Score: 0.8156, Time: 3.456s
Running data_programming... ✓ Score: 0.8089, Time: 2.123s
Running ebcc... ✓ Score: 0.7945, Time: 0.567s
Running NPLM... ✓ Score: 0.8312, Time: 4.567s

================================================================================
RESULTS
================================================================================
         Method  Score  Time (s)
           LELA 0.8567     1.234
          Metal 0.8423     2.345
          NPLM  0.8312     4.567
   dawid_skene  0.8234     0.234
  flying_squid  0.8156     3.456
data_programming 0.8089     2.123
          ebcc  0.7945     0.567
  majority_vote 0.7891     0.012
================================================================================
```

## Options

### Use F1 Score (for imbalanced data)
```python
results, times = run_label_aggregation_methods(
    X, y, 
    use_f1=True  # Use F1 instead of accuracy
)
```

### Skip LELA (if no checkpoint)
```python
results, times = run_label_aggregation_methods(
    X, y,
    lela_checkpoint=None  # Skip LELA
)
```

### Custom dataset name
```python
results, times = run_label_aggregation_methods(
    X, y,
    dataset_name="Gully Dataset"  # Shows in output
)
```

## Methods Included

The script runs these 8 label aggregation methods:

1. **dawid_skene** - Classic Dawid-Skene algorithm
2. **majority_vote** - Simple majority voting (baseline)
3. **LELA** - Learning to Ensemble (requires checkpoint)
4. **Metal** - Snorkel's MeTaL model
5. **flying_squid** - FlyingSquid algorithm
6. **data_programming** - Data Programming approach
7. **ebcc** - EBCC method
8. **NPLM** - Noisy Partial Label Model

## Tips

✅ **Use F1 score** if your classes are imbalanced (e.g., 90% class 0, 10% class 1)

✅ **Skip LELA** if you don't have the checkpoint file (set `lela_checkpoint=None`)

✅ **Check your data format**:
- X should have -1 for abstentions
- y should have 0, 1, 2, ... for class labels (no -1)
- Both should be numpy arrays

✅ **Save results** to CSV for later analysis

## Files

- **`run_single_dataset.py`** - Simplest script for single dataset (RECOMMENDED)
- **`gully_performance_exp.py`** - Main script with functions you can import
- **`SINGLE_DATASET_GUIDE.md`** - This file
- **`QUICKSTART.md`** - Ultra-quick reference

## Troubleshooting

**Problem**: "Could not load LELA model"
- **Solution**: Set `lela_checkpoint=None` or provide correct path

**Problem**: "No ground truth labels available"
- **Solution**: Check that y contains values >= 0 (not all -1)

**Problem**: Methods give low scores
- **Solution**: Check that X and y are correctly formatted and aligned

**Problem**: Out of memory
- **Solution**: Some methods (FlyingSquid, NPLM) use more memory. Try with fewer samples first.

## Need Help?

See the other documentation files:
- `QUICKSTART.md` - Minimal example
- `README_USAGE.md` - Comprehensive guide
- `example_usage.py` - Working code examples
