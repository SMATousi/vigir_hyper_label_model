import pandas as pd
import numpy as np
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset_wrench

# Same datasets as in avg_noise_performance_exp.py
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

def calculate_lf_statistics(X, y, dataset_name):
    # print(X)
    # print(y)
    """
    Calculate comprehensive statistics for each labeling function using sklearn metrics
    
    Args:
        X: LF matrix (n_samples x n_lfs) with values {-1, 0, 1}
        y: Ground truth labels (n_samples,)
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with statistics for each LF
    """
    n_samples, n_lfs = X.shape
    stats = {}
    
    # Filter out samples with abstaining ground truth (y < 0)
    valid_mask = y >= 0
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Total samples: {n_samples}")
    print(f"Valid samples (non-abstaining GT): {np.sum(valid_mask)}")
    print(f"Number of labeling functions: {n_lfs}")
    
    # Get unique labels for proper sklearn handling
    unique_labels = np.unique(y_valid)
    print(f"Unique ground truth labels: {unique_labels}")
    
    for lf_idx in range(n_lfs):
        lf_outputs = X_valid[:, lf_idx]
        
        # Basic counts
        # In the wrench format: -1 = abstention, 0 = negative class, 1 = positive class
        n_abstain = np.sum(lf_outputs == -1)  # -1 is abstention
        n_positive = np.sum(lf_outputs == 1)   # 1 is positive class
        n_negative = np.sum(lf_outputs == 0)   # 0 is negative class
        n_labeled = n_positive + n_negative
        
        # Coverage: fraction of non-abstaining predictions
        coverage = n_labeled / len(lf_outputs) if len(lf_outputs) > 0 else 0
        
        # Initialize metrics with default values
        accuracy = 0.0
        precision_pos = 0.0
        recall_pos = 0.0
        f1_pos = 0.0
        precision_neg = 0.0
        recall_neg = 0.0
        f1_neg = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
        
        # Calculate metrics only for non-abstaining predictions
        if n_labeled > 0:
            non_abstain_mask = lf_outputs != -1  # -1 is abstention
            lf_non_abstain = lf_outputs[non_abstain_mask]
            y_non_abstain = y_valid[non_abstain_mask]
            
            # Use sklearn metrics
            try:
                # Overall accuracy
                accuracy = accuracy_score(y_non_abstain, lf_non_abstain)
                
                # Get unique labels in the non-abstaining subset
                unique_pred_labels = np.unique(lf_non_abstain)
                unique_true_labels = np.unique(y_non_abstain)
                all_labels = np.unique(np.concatenate([unique_pred_labels, unique_true_labels]))
                
                # Macro-averaged metrics (average across classes)
                if len(all_labels) > 1:
                    macro_precision = precision_score(y_non_abstain, lf_non_abstain, 
                                                    labels=all_labels, average='macro', zero_division=0)
                    macro_recall = recall_score(y_non_abstain, lf_non_abstain, 
                                              labels=all_labels, average='macro', zero_division=0)
                    macro_f1 = f1_score(y_non_abstain, lf_non_abstain, 
                                       labels=all_labels, average='macro', zero_division=0)
                    
                    # Weighted-averaged metrics (weighted by support)
                    weighted_precision = precision_score(y_non_abstain, lf_non_abstain, 
                                                       labels=all_labels, average='weighted', zero_division=0)
                    weighted_recall = recall_score(y_non_abstain, lf_non_abstain, 
                                                 labels=all_labels, average='weighted', zero_division=0)
                    weighted_f1 = f1_score(y_non_abstain, lf_non_abstain, 
                                          labels=all_labels, average='weighted', zero_division=0)
                
                # Per-class metrics
                if len(all_labels) > 0:
                    per_class_precision = precision_score(y_non_abstain, lf_non_abstain, 
                                                        labels=all_labels, average=None, zero_division=0)
                    per_class_recall = recall_score(y_non_abstain, lf_non_abstain, 
                                                  labels=all_labels, average=None, zero_division=0)
                    per_class_f1 = f1_score(y_non_abstain, lf_non_abstain, 
                                           labels=all_labels, average=None, zero_division=0)
                    
                    # Map to specific classes (1 and 0)
                    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
                    
                    if 1 in label_to_idx:
                        idx_pos = label_to_idx[1]
                        precision_pos = per_class_precision[idx_pos]
                        recall_pos = per_class_recall[idx_pos]
                        f1_pos = per_class_f1[idx_pos]
                    
                    if 0 in label_to_idx:
                        idx_neg = label_to_idx[0]
                        precision_neg = per_class_precision[idx_neg]
                        recall_neg = per_class_recall[idx_neg]
                        f1_neg = per_class_f1[idx_neg]
                
            except Exception as e:
                print(f"Warning: Error calculating sklearn metrics for LF {lf_idx}: {e}")
        
        # Conflict analysis: how often does this LF disagree with other LFs
        conflicts_with_others = 0
        agreements_with_others = 0
        
        for other_lf_idx in range(n_lfs):
            if other_lf_idx == lf_idx:
                continue
            other_lf_outputs = X_valid[:, other_lf_idx]
            
            # Only consider cases where both LFs make non-abstaining predictions
            both_labeled_mask = (lf_outputs != -1) & (other_lf_outputs != -1)
            if np.sum(both_labeled_mask) > 0:
                lf_both = lf_outputs[both_labeled_mask]
                other_both = other_lf_outputs[both_labeled_mask]
                
                conflicts_with_others += np.sum(lf_both != other_both)
                agreements_with_others += np.sum(lf_both == other_both)
        
        total_comparisons = conflicts_with_others + agreements_with_others
        conflict_rate = conflicts_with_others / total_comparisons if total_comparisons > 0 else 0
        
        # Bias: tendency to predict positive vs negative
        if n_labeled > 0:
            positive_bias = n_positive / n_labeled
        else:
            positive_bias = 0.5  # neutral if no predictions
        
        # Confusion matrix analysis (if we have predictions)
        confusion_matrix_stats = {}
        if n_labeled > 0:
            try:
                non_abstain_mask = lf_outputs != -1  # -1 is abstention
                lf_non_abstain = lf_outputs[non_abstain_mask]
                y_non_abstain = y_valid[non_abstain_mask]
                
                if len(np.unique(y_non_abstain)) > 1 and len(np.unique(lf_non_abstain)) > 1:
                    cm = confusion_matrix(y_non_abstain, lf_non_abstain, labels=[0, 1])  # 0 is negative, 1 is positive
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        confusion_matrix_stats = {
                            'true_negatives': int(tn),
                            'false_positives': int(fp),
                            'false_negatives': int(fn),
                            'true_positives': int(tp)
                        }
            except Exception as e:
                print(f"Warning: Error calculating confusion matrix for LF {lf_idx}: {e}")
        
        stats[f'LF_{lf_idx}'] = {
            'dataset': dataset_name,
            'lf_index': lf_idx,
            'coverage': coverage,
            'accuracy': accuracy,
            'precision_positive': precision_pos,
            'recall_positive': recall_pos,
            'f1_positive': f1_pos,
            'precision_negative': precision_neg,
            'recall_negative': recall_neg,
            'f1_negative': f1_neg,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'conflict_rate': conflict_rate,
            'positive_bias': positive_bias,
            'n_abstain': n_abstain,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_labeled': n_labeled,
            'total_samples': len(lf_outputs),
            **confusion_matrix_stats
        }
    
    return stats

def analyze_dataset_level_statistics(X, y, dataset_name):
    """
    Calculate dataset-level statistics
    """
    # Filter out samples with abstaining ground truth
    valid_mask = y >= 0
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    n_samples, n_lfs = X_valid.shape
    
    # Overall coverage statistics
    labeled_per_sample = np.sum(X_valid != -1, axis=1)  # -1 is abstention
    avg_lfs_per_sample = np.mean(labeled_per_sample)
    min_lfs_per_sample = np.min(labeled_per_sample)
    max_lfs_per_sample = np.max(labeled_per_sample)
    
    # Samples with no LF coverage
    no_coverage_samples = np.sum(labeled_per_sample == 0)
    
    # Class distribution in ground truth
    unique_labels, label_counts = np.unique(y_valid, return_counts=True)
    class_distribution = dict(zip(unique_labels, label_counts))
    
    # Agreement statistics
    # For each sample, calculate how many LFs agree with each other
    agreement_scores = []
    for sample_idx in range(n_samples):
        sample_lfs = X_valid[sample_idx]
        non_abstain_lfs = sample_lfs[sample_lfs != -1]  # -1 is abstention
        
        if len(non_abstain_lfs) > 1:
            # Calculate pairwise agreement
            agreements = 0
            total_pairs = 0
            for i in range(len(non_abstain_lfs)):
                for j in range(i+1, len(non_abstain_lfs)):
                    total_pairs += 1
                    if non_abstain_lfs[i] == non_abstain_lfs[j]:
                        agreements += 1
            
            if total_pairs > 0:
                agreement_scores.append(agreements / total_pairs)
    
    avg_agreement = np.mean(agreement_scores) if agreement_scores else 0
    
    dataset_stats = {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'n_lfs': n_lfs,
        'avg_lfs_per_sample': avg_lfs_per_sample,
        'min_lfs_per_sample': min_lfs_per_sample,
        'max_lfs_per_sample': max_lfs_per_sample,
        'no_coverage_samples': no_coverage_samples,
        'no_coverage_rate': no_coverage_samples / n_samples,
        'avg_lf_agreement': avg_agreement,
        'class_distribution': class_distribution
    }
    
    return dataset_stats

def main():
    print("Analyzing Labeling Function Statistics")
    print("=" * 50)
    
    all_lf_stats = []
    all_dataset_stats = []
    
    for dataset_name in datasets:
        try:
            print(f"\nProcessing dataset: {dataset_name}")
            X, y = load_dataset_wrench(f"datasets/{dataset_name}")
            
            # Calculate LF-level statistics
            lf_stats = calculate_lf_statistics(X, y, dataset_name)
            for lf_name, stats in lf_stats.items():
                all_lf_stats.append(stats)
            
            # Calculate dataset-level statistics
            dataset_stats = analyze_dataset_level_statistics(X, y, dataset_name)
            all_dataset_stats.append(dataset_stats)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    # Convert to DataFrames
    lf_stats_df = pd.DataFrame(all_lf_stats)
    dataset_stats_df = pd.DataFrame(all_dataset_stats)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    lf_stats_df.to_csv("results/labeling_function_statistics.csv", index=False)
    dataset_stats_df.to_csv("results/dataset_level_statistics.csv", index=False)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nDataset-Level Statistics:")
    print(dataset_stats_df[['dataset', 'n_samples', 'n_lfs', 'avg_lfs_per_sample', 
                           'no_coverage_rate', 'avg_lf_agreement']].to_string(index=False))
    
    print("\nLabeling Function Statistics (Aggregated):")
    print("Average Coverage by Dataset:")
    coverage_by_dataset = lf_stats_df.groupby('dataset')['coverage'].agg(['mean', 'std', 'min', 'max'])
    print(coverage_by_dataset.round(4))
    
    print("\nAverage Accuracy by Dataset:")
    accuracy_by_dataset = lf_stats_df.groupby('dataset')['accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(accuracy_by_dataset.round(4))
    
    print("\nAverage Macro F1 Score by Dataset:")
    macro_f1_by_dataset = lf_stats_df.groupby('dataset')['macro_f1'].agg(['mean', 'std', 'min', 'max'])
    print(macro_f1_by_dataset.round(4))
    
    print("\nAverage Weighted F1 Score by Dataset:")
    weighted_f1_by_dataset = lf_stats_df.groupby('dataset')['weighted_f1'].agg(['mean', 'std', 'min', 'max'])
    print(weighted_f1_by_dataset.round(4))
    
    print("\nAverage Conflict Rate by Dataset:")
    conflict_by_dataset = lf_stats_df.groupby('dataset')['conflict_rate'].agg(['mean', 'std', 'min', 'max'])
    print(conflict_by_dataset.round(4))
    
    # Overall statistics across all datasets
    print("\nOverall Statistics Across All Datasets:")
    print(f"Total LFs analyzed: {len(lf_stats_df)}")
    print(f"Average LF coverage: {lf_stats_df['coverage'].mean():.4f} ± {lf_stats_df['coverage'].std():.4f}")
    print(f"Average LF accuracy: {lf_stats_df['accuracy'].mean():.4f} ± {lf_stats_df['accuracy'].std():.4f}")
    print(f"Average macro F1: {lf_stats_df['macro_f1'].mean():.4f} ± {lf_stats_df['macro_f1'].std():.4f}")
    print(f"Average weighted F1: {lf_stats_df['weighted_f1'].mean():.4f} ± {lf_stats_df['weighted_f1'].std():.4f}")
    print(f"Average macro precision: {lf_stats_df['macro_precision'].mean():.4f} ± {lf_stats_df['macro_precision'].std():.4f}")
    print(f"Average macro recall: {lf_stats_df['macro_recall'].mean():.4f} ± {lf_stats_df['macro_recall'].std():.4f}")
    print(f"Average conflict rate: {lf_stats_df['conflict_rate'].mean():.4f} ± {lf_stats_df['conflict_rate'].std():.4f}")
    print(f"Average positive bias: {lf_stats_df['positive_bias'].mean():.4f} ± {lf_stats_df['positive_bias'].std():.4f}")
    
    # Identify best and worst performing LFs
    print("\nTop 10 Best Performing LFs (by macro F1 score):")
    best_lfs = lf_stats_df.nlargest(10, 'macro_f1')[['dataset', 'lf_index', 'macro_f1', 'accuracy', 'coverage', 'conflict_rate']]
    print(best_lfs.to_string(index=False))
    
    print("\nTop 10 Best Performing LFs (by accuracy):")
    best_acc_lfs = lf_stats_df.nlargest(10, 'accuracy')[['dataset', 'lf_index', 'accuracy', 'macro_f1', 'coverage', 'conflict_rate']]
    print(best_acc_lfs.to_string(index=False))
    
    print("\nTop 10 Worst Performing LFs (by macro F1, with coverage > 0.1):")
    worst_lfs = lf_stats_df[lf_stats_df['coverage'] > 0.1].nsmallest(10, 'macro_f1')[['dataset', 'lf_index', 'macro_f1', 'accuracy', 'coverage', 'conflict_rate']]
    print(worst_lfs.to_string(index=False))
    
    print("\nTop 10 Highest Coverage LFs:")
    high_coverage = lf_stats_df.nlargest(10, 'coverage')[['dataset', 'lf_index', 'coverage', 'accuracy', 'macro_f1', 'conflict_rate']]
    print(high_coverage.to_string(index=False))
    
    print("\nTop 10 Most Conflicting LFs:")
    high_conflict = lf_stats_df.nlargest(10, 'conflict_rate')[['dataset', 'lf_index', 'conflict_rate', 'accuracy', 'macro_f1', 'coverage']]
    print(high_conflict.to_string(index=False))
    
    print("\nTop 10 Most Balanced LFs (by positive bias closest to 0.5):")
    lf_stats_df['bias_balance'] = abs(lf_stats_df['positive_bias'] - 0.5)
    balanced_lfs = lf_stats_df[lf_stats_df['coverage'] > 0.1].nsmallest(10, 'bias_balance')[['dataset', 'lf_index', 'positive_bias', 'accuracy', 'macro_f1', 'coverage']]
    print(balanced_lfs.to_string(index=False))
    
    print(f"\nResults saved to:")
    print(f"- results/labeling_function_statistics.csv")
    print(f"- results/dataset_level_statistics.csv")

if __name__ == "__main__":
    main()
