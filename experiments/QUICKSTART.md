# Quick Start Guide

## Simplest Usage (3 steps)

### Step 1: Prepare your data
```python
import numpy as np

# X: Label matrix (n_samples, n_labeling_functions)
# Values: -1 (abstain), 0, 1, 2, ... (class labels)
X = your_label_matrix  # e.g., shape (1000, 5)

# y: Ground truth (n_samples,)
# Values: 0, 1, 2, ... (class labels)
y = your_ground_truth  # e.g., shape (1000,)
```

### Step 2: Run methods
```python
from gully_performance_exp import run_label_aggregation_methods

results, times = run_label_aggregation_methods(X, y)
```

### Step 3: View results
```python
from gully_performance_exp import print_results_table

print_results_table({"my_data": results}, {"my_data": times})
```

## Complete Minimal Example

```python
import numpy as np
from gully_performance_exp import run_label_aggregation_methods, print_results_table

# Your data
X = np.array([
    [0, 1, -1, 1, 0],
    [1, 1, 1, -1, 1],
    [-1, 0, 0, 0, -1],
    [1, 1, 1, 1, 1],
    [0, 0, -1, 0, 0],
])  # 5 samples, 5 labeling functions

y = np.array([1, 1, 0, 1, 0])  # Ground truth

# Run and display
results, times = run_label_aggregation_methods(X, y, dataset_name="example")
print_results_table({"example": results}, {"example": times})
```

## Common Options

### Use F1 score instead of accuracy
```python
results, times = run_label_aggregation_methods(X, y, use_f1=True)
```

### Skip LELA (if no checkpoint available)
```python
results, times = run_label_aggregation_methods(X, y, lela_checkpoint=None)
```

### Run on multiple datasets
```python
all_results = {}
all_times = {}

for name, (X, y) in my_datasets.items():
    results, times = run_label_aggregation_methods(X, y, dataset_name=name)
    all_results[name] = results
    all_times[name] = times

print_results_table(all_results, all_times)
```

## That's it!

For more details, see `README_USAGE.md` or `example_usage.py`.
