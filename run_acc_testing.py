
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from data import DatasetOnlineGen
from transformer_models import LELATransformer
from bag_attention_model import LELATransformerBag
from loss import  BCEMask
import numpy as np
from data import SytheticValidation
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

NUM_RUNS = 2



#select the best run
val_accs = []
for i in range(NUM_RUNS): 
    checkpoint = torch.load("model_checkpoints/model_transformer_"+str(i)+".pt", map_location="cpu")
    val_accs.append(checkpoint['val_acc_avg'])
best_run = np.argmax(val_accs)
print('Please use the checkpoint:', "model_transformer_" + str(best_run) + ".pt")
