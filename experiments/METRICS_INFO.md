# Metrics Calculated

The scripts now calculate **5 comprehensive metrics** for each label aggregation method:

## Metrics

### 1. **Accuracy**
- Overall correctness: (TP + TN) / (TP + TN + FP + FN)
- Range: 0.0 to 1.0 (higher is better)
- Best for: Balanced datasets

### 2. **F1 Score**
- Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall)
- Range: 0.0 to 1.0 (higher is better)
- Best for: Imbalanced datasets, overall performance

### 3. **Precision** (Positive Predictive Value)
- How many predicted positives are actually positive: TP / (TP + FP)
- Range: 0.0 to 1.0 (higher is better)
- Best for: When false positives are costly

### 4. **Recall** (Sensitivity, True Positive Rate)
- How many actual positives were correctly identified: TP / (TP + FN)
- Range: 0.0 to 1.0 (higher is better)
- Best for: When false negatives are costly

### 5. **NPV** (Negative Predictive Value)
- How many predicted negatives are actually negative: TN / (TN + FN)
- Range: 0.0 to 1.0 (higher is better)
- Best for: Assessing negative class prediction quality

## Output Format

### For Binary Classification
All metrics use standard binary classification formulas.

### For Multiclass Classification
- **Accuracy**: Standard multiclass accuracy
- **F1, Precision, Recall**: Macro-averaged (average across all classes)
- **NPV**: Averaged across all classes

## Example Output

```
================================================================================
Running Methods...
================================================================================

Running dawid_skene... ✓ Acc: 0.8234, F1: 0.8156, Prec: 0.8345, Rec: 0.7989, NPV: 0.8567, Time: 0.234s
Running majority_vote... ✓ Acc: 0.7891, F1: 0.7723, Prec: 0.7956, Rec: 0.7512, NPV: 0.8234, Time: 0.012s
Running LELA... ✓ Acc: 0.8567, F1: 0.8489, Prec: 0.8623, Rec: 0.8367, NPV: 0.8789, Time: 1.234s
Running Metal... ✓ Acc: 0.8423, F1: 0.8334, Prec: 0.8512, Rec: 0.8167, NPV: 0.8656, Time: 2.345s
...

================================================================================
RESULTS
================================================================================
         Method  Accuracy     F1  Precision  Recall    NPV  Time (s)
           LELA    0.8567 0.8489     0.8623  0.8367 0.8789     1.234
          Metal    0.8423 0.8334     0.8512  0.8167 0.8656     2.345
    dawid_skene    0.8234 0.8156     0.8345  0.7989 0.8567     0.234
  majority_vote    0.7891 0.7723     0.7956  0.7512 0.8234     0.012
...
```

## CSV Output

The results are saved with all metrics:

```csv
Method,Accuracy,F1,Precision,Recall,NPV,Time (s)
LELA,0.8567,0.8489,0.8623,0.8367,0.8789,1.234
Metal,0.8423,0.8334,0.8512,0.8167,0.8656,2.345
dawid_skene,0.8234,0.8156,0.8345,0.7989,0.8567,0.234
majority_vote,0.7891,0.7723,0.7956,0.7512,0.8234,0.012
...
```

## Interpreting Results

### High Accuracy + High F1
✅ Model performs well overall

### High Precision, Low Recall
⚠️ Model is conservative (few false positives, but misses many true positives)

### Low Precision, High Recall
⚠️ Model is aggressive (catches most positives, but many false alarms)

### High NPV
✅ When model predicts negative, it's usually correct

### Balanced Metrics
✅ All metrics similar → robust performance across all aspects

## Confusion Matrix Reference

```
                Predicted
                Neg    Pos
Actual  Neg     TN     FP
        Pos     FN     TP
```

- **TP** (True Positive): Correctly predicted positive
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Incorrectly predicted positive
- **FN** (False Negative): Incorrectly predicted negative

## Which Metric to Focus On?

- **Balanced dataset**: Accuracy or F1
- **Imbalanced dataset**: F1 Score
- **Cost of false positives high** (e.g., spam detection): Precision
- **Cost of false negatives high** (e.g., disease detection): Recall
- **Need to trust negative predictions**: NPV
- **Comprehensive evaluation**: Look at all metrics together
