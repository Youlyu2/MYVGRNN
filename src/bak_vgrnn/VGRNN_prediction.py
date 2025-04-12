import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from model import VGRNN, graph_gru_sage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, SAGEConv

# from torch_geometric import nn as tgnn
from utils import (
    get_roc_scores,
    mask_edges_det,
    mask_edges_prd,
    mask_edges_prd_new,
)

seed = 123
np.random.seed(seed)

# loading data
topic = 'edca/cumulative'
adj_time_list = torch.load(f'../data/{topic}/adj_time_list.pt', weights_only=False)
adj_orig_dense_list = torch.load(f'../data/{topic}/adj_orig_dense_list.pt')

outs = mask_edges_det(adj_time_list)
train_edges_l = outs[1]

pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list)
pos_edges_l_n, false_edges_l_n = mask_edges_prd_new(adj_time_list, adj_orig_dense_list)


# creating edge list

edge_idx_list = []

for i in range(len(train_edges_l)):
    edge_idx_list.append(torch.tensor(np.transpose(train_edges_l[i]), dtype=torch.long))


h_dim = 32
z_dim = 16
n_layers =  1
clip = 10
learning_rate = 1e-2
seq_len = len(train_edges_l)
num_nodes = adj_orig_dense_list[seq_len-1].shape[0]
x_dim = num_nodes
eps = 1e-10
conv_type='GCN'


# creating input tensors

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32))
    x_in_list.append(torch.tensor(x_temp))

x_in = Variable(torch.stack(x_in_list))




model = VGRNN(x_dim, h_dim, z_dim, n_layers, eps, conv=conv_type, bias=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training

seq_start = 0
seq_end = seq_len - 3
tst_after = 0

# plot training curves and test results
losses_rec = []
auc_scores_prd_rec = []
ap_scores_prd_rec = []
auc_scores_prd_new_rec = []
ap_scores_prd_new_rec= []

for k in range(1000):
    optimizer.zero_grad()
    start_time = time.time()
    kld_loss, nll_loss, _, _, hidden_st, z_t = model(x_in[seq_start:seq_end]
                                                , edge_idx_list[seq_start:seq_end]
                                                , adj_orig_dense_list[seq_start:seq_end])
    loss = kld_loss + nll_loss
    loss.backward()
    optimizer.step()
    
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    if k>=tst_after:
        _, _, enc_means, pri_means, _, z_t = model(x_in[seq_end:seq_len]
                                              , edge_idx_list[seq_end:seq_len]
                                              , adj_orig_dense_list[seq_end:seq_len]
                                              , hidden_st)
        
        auc_scores_prd, ap_scores_prd = get_roc_scores(pos_edges_l[seq_end:seq_len]
                                                        , false_edges_l[seq_end:seq_len]
                                                        , adj_orig_dense_list[seq_end:seq_len]
                                                        , pri_means)
        
        auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(pos_edges_l_n[seq_end:seq_len]
                                                                , false_edges_l_n[seq_end:seq_len]
                                                                , adj_orig_dense_list[seq_end:seq_len]
                                                                , pri_means)
        
    
    print('epoch: ', k)
    print('kld_loss =', kld_loss.mean().item())
    print('nll_loss =', nll_loss.mean().item())
    print('loss =', loss.mean().item())
    if k>=tst_after:
        print('----------------------------------')
        print('Link Prediction')
        print('link_prd_auc_mean', np.mean(np.array(auc_scores_prd)))
        print('link_prd_ap_mean', np.mean(np.array(ap_scores_prd)))
        print('----------------------------------')
        print('New Link Prediction')
        print('new_link_prd_auc_mean', np.mean(np.array(auc_scores_prd_new)))
        print('new_link_prd_ap_mean', np.mean(np.array(ap_scores_prd_new)))
        print('----------------------------------')

    losses_rec.append(loss.mean().item())
    auc_scores_prd_rec.append(np.mean(np.array(auc_scores_prd)))
    ap_scores_prd_rec.append(np.mean(np.array(ap_scores_prd)))
    auc_scores_prd_new_rec.append(np.mean(np.array(auc_scores_prd_new)))
    ap_scores_prd_new_rec.append(np.mean(np.array(ap_scores_prd_new)))

    

    
    print('----------------------------------')


import os

import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

# Create figure with constrained layout
fig, ax1 = plt.subplots(figsize=(10, 6), layout='constrained')

# Plot loss on primary y-axis
color = 'tab:red'
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', color=color, fontsize=12)
loss_line = ax1.plot(losses_rec, color=color, label='Loss', linewidth=2, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color)

# Create secondary y-axis for metrics
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('AUC / AP Score', color=color, fontsize=12)

# Plot metrics with different styles
val_auc_line = ax2.plot(auc_scores_prd_rec, color='tab:blue', linestyle='--', 
                       label='Val AUC', linewidth=2, alpha=0.8)
val_ap_line = ax2.plot(ap_scores_prd_rec, color='tab:green', linestyle='--', 
                      label='Val AP', linewidth=2, alpha=0.8)
test_auc_line = ax2.plot(auc_scores_prd_new_rec, color='tab:blue', 
                        label='Test AUC', linewidth=2, alpha=0.8)
test_ap_line = ax2.plot(ap_scores_prd_new_rec, color='tab:green', 
                       label='Test AP', linewidth=2, alpha=0.8)

# Combine legends from both axes
lines = loss_line + val_auc_line + val_ap_line + test_auc_line + test_ap_line
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, fontsize=10)

# Add title and adjust layout
plt.title(f'Training Curves - {topic}', fontsize=14, pad=20)

# Save figure with high DPI
os.makedirs(f'./results/{topic}', exist_ok=True)
plt.savefig(f'./results/{topic}training_curves_pred.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


