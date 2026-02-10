# Label Aggregation Methods - Usage Guide

## Overview

This directory contains scripts for running label aggregation methods on weak supervision data. The code has been modified to easily work with **your own single dataset** where you provide the labeling function results and ground truth.

## For Single Dataset Users (Your Use Case)

### 🎯 Quick Start

**You have**: One dataset with labeling function results (X) and ground truth (y)

**You want**: Run all methods and get a results table

**Use this**: `run_single_dataset.py`

```python
# 1. Edit run_single_dataset.py - replace the data loading section with:
X = np.load('your_label_matrix.npy')
y = np.load('your_ground_truth.npy')

# 2. Run the script
python run_single_dataset.py

# 3. Get results table automatically!
```

### 📖 Documentation

- **`SINGLE_DATASET_GUIDE.md`** ⭐ START HERE - Complete guide for single dataset
- **`QUICKSTART.md`** - 3-step minimal example
- **`run_single_dataset.py`** - Ready-to-use script (just add your data)

## Data Format

### Input: Label Matrix (X)
```python
# Shape: (n_samples, n_labeling_functions)
# Values: -1 (abstain), 0, 1, 2, ... (class labels)

X = np.array([
    [0, 1, -1, 1],   # Sample 0
    [1, 1, 1, -1],   # Sample 1
    [0, 0, 0, 0],    # Sample 2
    # ...
])
```

### Input: Ground Truth (y)
```python
# Shape: (n_samples,)
# Values: 0, 1, 2, ... (class labels)

y = np.array([1, 1, 0, ...])
```

### Output: Results Table
```
Method            Score   Time (s)
LELA             0.8567   1.234
Metal            0.8423   2.345
NPLM             0.8312   4.567
dawid_skene      0.8234   0.234
...
```

## Methods Included

8 label aggregation methods are automatically run:

1. **Dawid-Skene** - Classic probabilistic model
2. **Majority Vote** - Simple baseline
3. **LELA** - Learning to Ensemble (requires checkpoint)
4. **Metal** - Snorkel's MeTaL model
5. **FlyingSquid** - Triplet-based method
6. **Data Programming** - Generative model
7. **EBCC** - Expectation-based method
8. **NPLM** - Noisy Partial Label Model

## File Guide

### For Single Dataset (Your Use Case)
```
run_single_dataset.py          ⭐ Main script - use this!
SINGLE_DATASET_GUIDE.md        📖 Complete guide
QUICKSTART.md                  📖 Minimal example
```

### For Multiple Datasets (Original Use Case)
```
gully_performance_exp.py       Main script with functions
README_USAGE.md                Comprehensive documentation
example_usage.py               Working examples
```

### Other Files
```
test_modifications.py          Test suite
CHANGES.md                     Detailed changelog
baselines/                     Baseline method implementations
```

## Three Ways to Use

### 1. Simplest: Use run_single_dataset.py (RECOMMENDED)

```python
# Edit the script, replace data loading section
X = np.load('your_data.npy')
y = np.load('your_labels.npy')

# Run
python run_single_dataset.py
```

### 2. Import as Functions

```python
from gully_performance_exp import run_label_aggregation_methods, print_results_table

results, times = run_label_aggregation_methods(X, y)
print_results_table({"my_data": results}, {"my_data": times})
```

### 3. Interactive (Jupyter/IPython)

```python
import numpy as np
from gully_performance_exp import run_label_aggregation_methods, print_results_table

# Load data
X = np.load('data.npy')
y = np.load('labels.npy')

# Run and display
results, times = run_label_aggregation_methods(X, y, dataset_name="My Dataset")
print_results_table({"My Dataset": results}, {"My Dataset": times})
```

## Common Options

### Use F1 Score (for imbalanced data)
```python
results, times = run_label_aggregation_methods(X, y, use_f1=True)
```

### Skip LELA (if no checkpoint)
```python
results, times = run_label_aggregation_methods(X, y, lela_checkpoint=None)
```

### Save Results
```python
import pandas as pd
pd.DataFrame([results]).to_csv("my_results.csv", index=False)
```

## Example Workflow

```python
import numpy as np
from gully_performance_exp import run_label_aggregation_methods, print_results_table

# 1. Load your data
X = np.load('gully_labels.npy')      # (1000, 5) - 1000 samples, 5 LFs
y = np.load('gully_ground_truth.npy')  # (1000,) - true labels

# 2. Run methods
results, times = run_label_aggregation_methods(
    X=X,
    y=y,
    dataset_name="Gully Dataset",
    use_f1=False,  # or True for F1 score
    lela_checkpoint="./lela_checkpoint.pt"  # or None
)

# 3. Display results
print_results_table({"Gully Dataset": results}, {"Gully Dataset": times})

# 4. Save results
import pandas as pd
pd.DataFrame([results]).to_csv("gully_results.csv", index=False)
print("✓ Results saved!")
```

## Requirements

- Python 3.6+
- NumPy
- Pandas
- scikit-learn
- PyTorch (for LELA, Metal, NPLM)
- Snorkel
- FlyingSquid
- Other dependencies in the baselines

## Getting Started

1. **Read**: `SINGLE_DATASET_GUIDE.md`
2. **Edit**: `run_single_dataset.py` with your data
3. **Run**: `python run_single_dataset.py`
4. **Done**: Get your results table!

## Support

- **Single dataset**: See `SINGLE_DATASET_GUIDE.md`
- **Quick reference**: See `QUICKSTART.md`
- **Multiple datasets**: See `README_USAGE.md`
- **Examples**: See `example_usage.py`
- **Testing**: Run `python test_modifications.py`

## Original Functionality

The original WRENCH dataset experiments are preserved:

```python
from gully_performance_exp import run_wrench_experiments
run_wrench_experiments(run_experiments=True)
```

---

**TL;DR**: Use `run_single_dataset.py` - just add your data and run!
