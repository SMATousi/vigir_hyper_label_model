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

NOISE_POWER = 0.0  # 10% chance of flipping each non-zero label
# datasets = ["semeval", "agnews", "trec", "spouse", "chemprot", "sms",  'census', 'commercial', 'youtube',
#             "yelp", 'imdb', 'cdr', 'tennis', 'basketball'] # name of the 14 datasets

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

# dicts to save performance scores and runing times 
rsts = defaultdict(list)
running_times = defaultdict(list)
for i in range(len(datasets)):
    rsts['dataset'].append(datasets[i])
    print(datasets[i])
    X, y = load_dataset_wrench("datasets/"+datasets[i]) # load dataset
    # print(y[:10])

    # mallicious_labeler = (1 - y).reshape(-1, 1)
    # print(X.shape)
    # print(mallicious_labeler.shape)
    # X = np.concatenate([X, mallicious_labeler], axis=1)
    # Apply label flipping noise: flip 0 to 1 and 1 to 0 with probability NOISE_POWER
    # Note: -1 is abstention, 0 is negative class, 1 is positive class
    print(f"Applying label flipping noise with probability {NOISE_POWER} to non-abstaining values")
    X_noisy = X.copy()
    
    # Create mask for non-abstaining values (where X is 0 or 1, not -1)
    non_abstain_mask = X != -1
    
    # Generate random probabilities for each position
    flip_probs = np.random.random(X.shape)
    
    # Create flip mask: flip where probability < NOISE_POWER and value is non-abstaining
    flip_mask = (flip_probs < NOISE_POWER) & non_abstain_mask
    
    # Flip the labels: 0 becomes 1, 1 becomes 0, -1 stays -1
    X_noisy[flip_mask] = 1 - X_noisy[flip_mask]
    
    # Count flipped labels for reporting
    num_flipped = np.sum(flip_mask)
    total_non_abstain = np.sum(non_abstain_mask)
    print(f"Flipped {num_flipped} out of {total_non_abstain} non-abstaining labels ({num_flipped/total_non_abstain*100:.1f}%)")
    
    X = X_noisy
    # remove cols and rows with all abstentions (-1)
    non_abstain_cols = np.sum(X != -1, axis=0) > 0
    X = X[:, non_abstain_cols]
    non_abstain_rows = np.sum(X != -1, axis=1) > 0
    X = X[non_abstain_rows, :]
    y = y[non_abstain_rows]

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
        print(method_name, acc, delta_t)
        rsts[method_name].append(acc)
        running_times[method_name].append(delta_t)
        print("----")

rst_df = pd.DataFrame.from_dict(rsts)
run_time_df = pd.DataFrame.from_dict(running_times)
print(rst_df.mean())
print(run_time_df.mean())
rst_df.to_csv("results/performance_exp_acc.csv", index=False)
run_time_df.to_csv("results/performance_exp_time.csv", index=False)
