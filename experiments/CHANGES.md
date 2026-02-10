# Changes to gully_performance_exp.py

## Summary

The script has been refactored to accept pre-computed labeling function results and ground truth as inputs, making it easy to run label aggregation methods on your own data and produce formatted results tables.

## Key Changes

### 1. New Function: `run_label_aggregation_methods()`

**Purpose**: Core function that runs all label aggregation methods on provided data.

**Signature**:
```python
def run_label_aggregation_methods(
    X: np.ndarray, 
    y: np.ndarray, 
    dataset_name: str = "custom",
    use_f1: bool = False,
    lela_checkpoint: Optional[str] = "./lela_checkpoint.pt",
    methods_to_run: Optional[List] = None
) -> Tuple[Dict[str, float], Dict[str, float]]
```

**Features**:
- Accepts label matrix (X) and ground truth (y) as inputs
- Automatically filters out rows/columns with all abstentions
- Handles LELA initialization with error handling
- Supports both accuracy and F1 score metrics
- Returns dictionaries of results and running times
- Includes detailed progress output

### 2. New Function: `print_results_table()`

**Purpose**: Formats and displays results in a clean table format.

**Signature**:
```python
def print_results_table(
    results_dict: Dict[str, Dict[str, float]], 
    times_dict: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame
```

**Features**:
- Creates formatted tables for results and times
- Calculates and displays mean performance across datasets
- Returns DataFrame for further processing
- Clean, readable output with separators

### 3. Refactored: `run_wrench_experiments()`

**Purpose**: Wrapped original experiment loop in a function.

**Changes**:
- Original code now in a function with `run_experiments` flag
- Uses new `run_label_aggregation_methods()` internally
- Cleaner structure and better organization
- Maintains all original functionality

### 4. Added: Example Usage Section

**Location**: Bottom of script in `if __name__ == "__main__"` block

**Includes**:
- Template for using your own data
- Instructions for running WRENCH experiments
- Clear comments and documentation

## Backward Compatibility

✅ **Fully backward compatible**: Original functionality preserved
- Set `run_wrench_experiments(run_experiments=True)` to run original experiments
- All original methods and logic intact
- Same output format and CSV files

## New Capabilities

### Before (Original)
```python
# Could only run on WRENCH datasets
# Had to modify code to use custom data
# Results printed to console only
```

### After (Modified)
```python
# Can easily use your own data
from gully_performance_exp import run_label_aggregation_methods, print_results_table

X = your_label_matrix
y = your_ground_truth

results, times = run_label_aggregation_methods(X, y)
print_results_table({"my_data": results}, {"my_data": times})
```

## File Structure

```
experiments/
├── gully_performance_exp.py      # Modified main script
├── example_usage.py               # NEW: Working examples
├── README_USAGE.md                # NEW: Detailed documentation
├── QUICKSTART.md                  # NEW: Quick reference
└── CHANGES.md                     # NEW: This file
```

## Methods Included

All original methods are included:
1. Dawid-Skene
2. Majority Vote
3. LELA (if checkpoint available)
4. Metal (Snorkel)
5. FlyingSquid
6. Data Programming
7. EBCC
8. NPLM

## Usage Patterns

### Pattern 1: Single Dataset
```python
results, times = run_label_aggregation_methods(X, y)
print_results_table({"dataset": results}, {"dataset": times})
```

### Pattern 2: Multiple Datasets
```python
all_results = {}
all_times = {}
for name, (X, y) in datasets.items():
    results, times = run_label_aggregation_methods(X, y, dataset_name=name)
    all_results[name] = results
    all_times[name] = times
print_results_table(all_results, all_times)
```

### Pattern 3: Custom Configuration
```python
results, times = run_label_aggregation_methods(
    X, y,
    dataset_name="custom",
    use_f1=True,
    lela_checkpoint=None,
    methods_to_run=[majority_vote, dawid_skene]
)
```

## Benefits

1. **Ease of Use**: Simple API for running methods on custom data
2. **Flexibility**: Control which methods to run, metrics to use
3. **Clean Output**: Formatted tables instead of raw prints
4. **Reusability**: Functions can be imported and used in other scripts
5. **Documentation**: Comprehensive docs and examples
6. **Type Hints**: Better IDE support and code clarity
7. **Error Handling**: Graceful handling of missing checkpoints/errors

## Testing

To test the changes:

1. **Test with synthetic data**:
   ```bash
   python example_usage.py
   ```

2. **Test with your data**:
   ```python
   from gully_performance_exp import run_label_aggregation_methods
   results, times = run_label_aggregation_methods(your_X, your_y)
   ```

3. **Test original functionality**:
   ```python
   from gully_performance_exp import run_wrench_experiments
   run_wrench_experiments(run_experiments=True)
   ```

## Migration Guide

### If you were using the original script:

**Before**:
```python
# Run the script directly
python gully_performance_exp.py
```

**After**:
```python
# Option 1: Run as before (set flag to True in script)
python gully_performance_exp.py

# Option 2: Import and use programmatically
from gully_performance_exp import run_wrench_experiments
run_wrench_experiments(run_experiments=True)
```

### If you want to use your own data:

**New capability**:
```python
from gully_performance_exp import run_label_aggregation_methods, print_results_table

results, times = run_label_aggregation_methods(X, y)
print_results_table({"my_data": results}, {"my_data": times})
```

## Notes

- All original dependencies remain the same
- No breaking changes to existing functionality
- New type hints require Python 3.5+ (already required by other dependencies)
- LELA checkpoint handling is more robust (won't crash if missing)
