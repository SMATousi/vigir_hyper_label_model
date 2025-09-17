import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
import sys
import os
from collections import defaultdict

from baselines.baselines import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LELAWrapper
from data import load_dataset_wrench


lela = LELAWrapper(checkpoint_path="./lela_checkpoint.pt") #load pretrained LELA model

# Define noise power levels from 0.05 to 0.5 with increment of 0.05
noise_levels = np.arange(0.00, 0.55, 0.05)  # [0.05, 0.10, 0.15, ..., 0.50]

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
    ] # name of the 14 datasets

# Dictionary to store results for each noise level
all_noise_results = {}
all_noise_times = {}

# Run experiments for each noise level
for noise_power in noise_levels:
    print(f"\n{'='*60}")
    print(f"Running experiments with NOISE_POWER = {noise_power:.2f}")
    print(f"{'='*60}")
    
    # dicts to save performance scores and running times for this noise level
    rsts = defaultdict(list)
    running_times = defaultdict(list)
    
    for i in range(len(datasets)):
        rsts['dataset'].append(datasets[i])
        print(f"\nProcessing dataset: {datasets[i]}")
        X, y = load_dataset_wrench("datasets/"+datasets[i]) # load dataset
        
        # Apply label flipping noise: flip -1 to 1 and 1 to -1 with probability noise_power
        print(f"Applying label flipping noise with probability {noise_power:.2f} to non-zero values")
        X_noisy = X.copy()
        
        # Create mask for non-zero values (where X is -1 or 1)
        non_zero_mask = X != 0
        
        # Generate random probabilities for each position
        flip_probs = np.random.random(X.shape)
        
        # Create flip mask: flip where probability < noise_power and value is non-zero
        flip_mask = (flip_probs < noise_power) & non_zero_mask
        
        # Flip the labels: -1 becomes 1, 1 becomes -1, 0 stays 0
        X_noisy[flip_mask] = -X_noisy[flip_mask]
        
        # Count flipped labels for reporting
        num_flipped = np.sum(flip_mask)
        total_non_zero = np.sum(non_zero_mask)
        print(f"Flipped {num_flipped} out of {total_non_zero} non-zero labels ({num_flipped/total_non_zero*100:.1f}%)")
        
        X = X_noisy
        # remove cols and rows with all abstentions
        non_zero_cols = np.sum(X >= 0, axis=0) != 0
        X = X[:, non_zero_cols]
        non_zero = np.sum(X >= 0, axis=1) != 0
        X = X[non_zero, :]
        y = y[non_zero]

        y_ind = np.arange(0, X.shape[0])
        for method in [dawid_skene, majority_vote, lela, Metal, flying_squid, data_programming, ebcc,NPLM]:
            t_s = time.time()

            if isinstance(method, LELAWrapper):
                method_name = "LELA"
                pred_round = method.predict(X)
            else:
                method_name = method.__name__
                pred = method(X)
                pred_round = np.argmax(pred, axis=1).flatten()
            delta_t = time.time() - t_s
            gt_labels = np.copy(y).flatten().astype(int)

            y_ind_ = y_ind[gt_labels >= 0] # only use data points with non-abstension gt labels to evalute 
            gt_labels = gt_labels[gt_labels >= 0]
            pred_round = pred_round[y_ind_]
            if datasets[i] in ["sms", "spouse", 'cdr',  'commercial', 'tennis', 'basketball',
                            'census']:
                acc = f1_score(gt_labels, pred_round)
            else:
                acc = accuracy_score(gt_labels, pred_round)
            print(f"{method_name}: {acc:.4f} (time: {delta_t:.2f}s)")
            rsts[method_name].append(acc)
            running_times[method_name].append(delta_t)

            # break
        # break
    
    # Store results for this noise level
    rst_df = pd.DataFrame.from_dict(rsts)
    run_time_df = pd.DataFrame.from_dict(running_times)
    
    # Calculate and store average results for this noise level
    avg_results = rst_df.mean(numeric_only=True)
    avg_times = run_time_df.mean(numeric_only=True)
    
    all_noise_results[f'noise_{noise_power:.2f}'] = avg_results
    all_noise_times[f'noise_{noise_power:.2f}'] = avg_times
    
    print(f"\nAverage results for noise power {noise_power:.2f}:")
    print(avg_results)
    print(f"\nAverage times for noise power {noise_power:.2f}:")
    print(avg_times)

    # break

# Create final DataFrames with average results across all noise levels
final_avg_results_df = pd.DataFrame.from_dict(all_noise_results, orient='index')
final_avg_times_df = pd.DataFrame.from_dict(all_noise_times, orient='index')

# Add noise power as a column
final_avg_results_df['noise_power'] = [float(idx.split('_')[1]) for idx in final_avg_results_df.index]
final_avg_times_df['noise_power'] = [float(idx.split('_')[1]) for idx in final_avg_times_df.index]

# Reset index to make noise_power a regular column
final_avg_results_df = final_avg_results_df.reset_index(drop=True)
final_avg_times_df = final_avg_times_df.reset_index(drop=True)

# Reorder columns to put noise_power first
cols_acc = ['noise_power'] + [col for col in final_avg_results_df.columns if col != 'noise_power']
cols_time = ['noise_power'] + [col for col in final_avg_times_df.columns if col != 'noise_power']
final_avg_results_df = final_avg_results_df[cols_acc]
final_avg_times_df = final_avg_times_df[cols_time]

print("\n" + "="*80)
print("FINAL AVERAGE RESULTS ACROSS ALL NOISE LEVELS")
print("="*80)
print("\nAccuracy Results:")
print(final_avg_results_df)
print("\nTiming Results:")
print(final_avg_times_df)

# Save the final average results
final_avg_results_df.to_csv("results/avg_noise_performance_exp_acc.csv", index=False)
final_avg_times_df.to_csv("results/avg_noise_performance_exp_time.csv", index=False)

print("\nResults saved to:")
print("- results/avg_noise_performance_exp_acc.csv")
print("- results/avg_noise_performance_exp_time.csv")
