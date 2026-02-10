# Using gully_performance_exp.py with Your Own Data

## Overview

The `gully_performance_exp.py` script has been modified to allow you to easily run label aggregation methods on your own labeling function results and ground truth data.

## Quick Start

### Basic Usage

```python
from gully_performance_exp import run_label_aggregation_methods, print_results_table
import numpy as np

# Load your data
X = np.load('your_label_matrix.npy')  # Shape: (n_samples, n_labeling_functions)
y = np.load('your_ground_truth.npy')  # Shape: (n_samples,)

# Run methods
results, times = run_label_aggregation_methods(
    X=X,
    y=y,
    dataset_name="my_dataset",
    use_f1=False,  # Set to True for F1 score
    lela_checkpoint="./lela_checkpoint.pt"  # or None to skip LELA
)

# Print results table
print_results_table({"my_dataset": results}, {"my_dataset": times})
```

## Data Format

### Label Matrix (X)
- **Shape**: `(n_samples, n_labeling_functions)`
- **Values**: 
  - `-1`: Abstention (labeling function did not label this sample)
  - `0, 1, 2, ...`: Class labels

**Example**:
```python
X = np.array([
    [0, 1, -1, 1, 0],   # Sample 0: LF0=class0, LF1=class1, LF2=abstain, etc.
    [1, 1, 1, -1, 1],   # Sample 1
    [-1, 0, 0, 0, -1],  # Sample 2
    # ... more samples
])
```

### Ground Truth (y)
- **Shape**: `(n_samples,)`
- **Values**: `0, 1, 2, ...` (class labels)
- Can include `-1` for samples without ground truth (they will be filtered out during evaluation)

**Example**:
```python
y = np.array([1, 1, 0, 1, 0, ...])  # True class labels
```

## Available Methods

The script runs the following label aggregation methods:
1. **dawid_skene**: Dawid-Skene algorithm
2. **majority_vote**: Simple majority voting
3. **LELA**: Learning to Ensemble Labeling Functions (requires checkpoint)
4. **Metal**: Snorkel's MeTaL model
5. **flying_squid**: FlyingSquid algorithm
6. **data_programming**: Data Programming approach
7. **ebcc**: EBCC method
8. **NPLM**: Noisy Partial Label Model

## Function Reference

### `run_label_aggregation_methods()`

Runs label aggregation methods on provided data.

**Parameters**:
- `X` (np.ndarray): Label matrix of shape (n_samples, n_labeling_functions)
- `y` (np.ndarray): Ground truth labels of shape (n_samples,)
- `dataset_name` (str, optional): Name for display. Default: "custom"
- `use_f1` (bool, optional): Use F1 score instead of accuracy. Default: False
- `lela_checkpoint` (str or None, optional): Path to LELA checkpoint. Default: "./lela_checkpoint.pt"
- `methods_to_run` (list or None, optional): Specific methods to run. Default: None (runs all)

**Returns**:
- `results` (dict): Method names → scores
- `times` (dict): Method names → running times

### `print_results_table()`

Prints formatted results table.

**Parameters**:
- `results_dict` (dict): Dataset names → method results
- `times_dict` (dict or None, optional): Dataset names → method times

**Returns**:
- `df` (pd.DataFrame): Results as DataFrame

## Examples

### Example 1: Single Dataset

```python
from gully_performance_exp import run_label_aggregation_methods, print_results_table
import numpy as np

# Your data
X = np.array([...])  # (1000, 5) - 1000 samples, 5 labeling functions
y = np.array([...])  # (1000,) - ground truth

# Run
results, times = run_label_aggregation_methods(X, y, dataset_name="my_data")
print_results_table({"my_data": results}, {"my_data": times})
```

### Example 2: Multiple Datasets

```python
all_results = {}
all_times = {}

for dataset_name in ["dataset_A", "dataset_B", "dataset_C"]:
    X, y = load_my_data(dataset_name)  # Your loading function
    results, times = run_label_aggregation_methods(X, y, dataset_name=dataset_name)
    all_results[dataset_name] = results
    all_times[dataset_name] = times

print_results_table(all_results, all_times)
```

### Example 3: Using F1 Score

```python
# For imbalanced datasets or binary classification
results, times = run_label_aggregation_methods(
    X, y, 
    dataset_name="imbalanced_data",
    use_f1=True  # Use F1 instead of accuracy
)
```

### Example 4: Skip LELA

```python
# If you don't have LELA checkpoint
results, times = run_label_aggregation_methods(
    X, y,
    lela_checkpoint=None  # Skip LELA
)
```

### Example 5: Run Specific Methods Only

```python
from baselines.baselines import majority_vote, dawid_skene, Metal

results, times = run_label_aggregation_methods(
    X, y,
    methods_to_run=[majority_vote, dawid_skene, Metal]  # Only these methods
)
```

## Loading Data from Different Sources

### From NumPy files
```python
X = np.load('label_matrix.npy')
y = np.load('ground_truth.npy')
```

### From CSV files
```python
import pandas as pd
X = pd.read_csv('label_matrix.csv').values
y = pd.read_csv('ground_truth.csv').values.flatten()
```

### From Pandas DataFrame
```python
import pandas as pd
df = pd.read_csv('data.csv')
X = df[['LF1', 'LF2', 'LF3', 'LF4', 'LF5']].values
y = df['ground_truth'].values
```

## Output

The script produces:
1. **Console output**: Real-time progress and results for each method
2. **Results table**: Formatted table showing all methods' performance
3. **CSV files** (optional): Can save results to CSV for further analysis

### Sample Output
```
================================================================================
RESULTS TABLE (Accuracy/F1 Score)
================================================================================
              dawid_skene  majority_vote      LELA     Metal  flying_squid  ...
my_dataset         0.8234         0.7891    0.8567    0.8423        0.8156  ...

Mean Performance:
dawid_skene      0.8234
majority_vote    0.7891
LELA            0.8567
...

================================================================================
RUNNING TIMES (seconds)
================================================================================
              dawid_skene  majority_vote    LELA   Metal  flying_squid  ...
my_dataset         0.234          0.012   1.234   2.345         3.456  ...
```

## Tips

1. **Data Preprocessing**: The script automatically removes:
   - Columns (labeling functions) with all abstentions
   - Rows (samples) with all abstentions
   - Samples without ground truth during evaluation

2. **Memory**: For large datasets, methods like FlyingSquid and NPLM may require significant memory

3. **Speed**: Majority vote is fastest; LELA and NPLM are typically slower

4. **Metrics**: Use `use_f1=True` for:
   - Binary classification with class imbalance
   - When you care more about positive class performance

5. **LELA**: Requires a pre-trained checkpoint. Set to `None` if unavailable.

## Running the Original WRENCH Experiments

To run the original experiments on WRENCH datasets:

```python
from gully_performance_exp import run_wrench_experiments

run_wrench_experiments(run_experiments=True)
```

## See Also

- `example_usage.py`: Complete working examples
- Original paper/documentation for each method's details
