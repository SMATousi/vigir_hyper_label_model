import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from data import DatasetOnlineGen
from transformer_models import LELATransformer
from bag_attention_model import LELATransformerBag, StackedLELATransformerBag
from loss import  BCEMask
import numpy as np
from data import SytheticValidation
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

NUM_RUNS = 10
for i_run in range(NUM_RUNS): #train LELA model 10 runs
    device = "cuda:0"
    # Use sequence length limit to prevent memory explosion
    max_seq_len = 800000  # Adjust based on available GPU memory
    # net = LELATransformer(max_seq_len=max_seq_len)
    # net = LELATransformerBag()
    net = StackedLELATransformerBag()

    optimizer = optim.Adam(
        net.parameters(),
        amsgrad=True,
    )
    net.to(device)

    criterion = BCEMask()

    dataset = DatasetOnlineGen(
        size=100, # this is just to trick data loader to work, the dataset is generated on the fly so there is no dataset size
        max_n_lfs=60,
        max_example=2000,
    )


    valid = SytheticValidation()# the sythetic validation set


    collate_fn = dataset.collate

    dataloader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    writer = SummaryWriter(comment="run_"+str(i_run))
    n_iter = 0
    optimizer.zero_grad()
    n_not_improved = 0
    min_loss = np.inf
    losses = []
    val_accs = []

    torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg':np.mean(val_accs),
            },
            "model_checkpoints/stacked_bag_model_transformer_random.pt"
            )
    
    for _ in tqdm(range(100)):  
        if n_not_improved > 10**4:
            break
        for i, (index, value, labels) in enumerate(dataloader):

            try:
                # Place tensors on GPU
                index = index.int()
                index, value, labels = index.squeeze().to(
                    device), value.squeeze().to(device), labels.squeeze().to(device)

                outputs, _, aux = net(index, value)     # outputs: (B, E_max)
                B, E_max = outputs.shape
                E = labels.shape[1]
                if E < E_max:
                    pad = torch.full((B, E_max - E), -1, dtype=labels.dtype, device=labels.device)
                    labels = torch.cat([labels, pad], dim=1)

                # print(outputs.shape, (labels != -1).shape, aux["example_mask"].shape)
                mask = aux["example_mask"] & (labels != -1)
                loss = criterion(outputs, labels.float(), mask)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                l_value = loss.item()

                n_iter += 1

                eval_fre = 100
                losses.append(l_value)
                if n_iter%eval_fre==0:
                    net.eval()
                    with torch.no_grad():
                        test_score_sythetic_ind = valid.get_avg_score_sythetic(
                            net)
                    writer.add_scalar('score/test_syth_acc_ind',
                                        test_score_sythetic_ind, n_iter)
                    val_accs.append(test_score_sythetic_ind)
                    l_avg = np.mean(losses[-1000:])
                    writer.add_scalar('score/train_loss_iter',l_avg, n_iter)
                    if l_avg< min_loss:
                        min_loss = l_avg
                        n_not_improved = 0
                    else:
                        n_not_improved+=eval_fre
                    net.train()
            except Exception as e:
                # print(e)
                continue
        if not os.path.exists("model_checkpoints"): 
            os.mkdir("model_checkpoints")

        torch.save(
            {
                'n_iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_avg':np.mean(val_accs),
            },
            "model_checkpoints/stacked_bag_model_transformer_" + str(i_run) + ".pt"
            )

#select the best run
val_accs = []
for i in range(NUM_RUNS): 
    checkpoint = torch.load("model_checkpoints/stacked_bag_model_transformer_"+str(i)+".pt", map_location="cpu")
    val_accs.append(checkpoint['val_acc_avg'])
best_run = np.argmax(val_accs)
print('Please use the checkpoint:', "stacked_bag_model_transformer_" + str(best_run) + ".pt")
