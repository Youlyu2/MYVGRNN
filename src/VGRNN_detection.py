import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from model import VGRNN
from torch.autograd import Variable

# from torch_geometric import nn as tgnn
from utils import (
    get_roc_scores,
    mask_edges_det,
)

topic = 'edca/cumulative'

seed = 3
np.random.seed(seed)

# utility functions

adj_time_list = torch.load(f'../data/{topic}/adj_time_list.pt', weights_only=False)

# with open('data/fb/adj_orig_dense_list_torch.pickle', 'rb') as handle:
#     adj_orig_dense_list = pickle.load(handle)
adj_orig_dense_list = torch.load(f'../data/{topic}/adj_orig_dense_list.pt')


# masking edges

outs = mask_edges_det(adj_time_list)

adj_train_l = outs[0]
train_edges_l = outs[1]
val_edges_l = outs[2]
val_edges_false_l = outs[3]
test_edges_l = outs[4]
test_edges_false_l = outs[5]

# creating edge list

edge_idx_list = []

for i in range(len(train_edges_l)):
    edge_idx_list.append(torch.tensor(np.transpose(train_edges_l[i]), dtype=torch.long))


# hyperparameters

h_dim = 32
z_dim = 16
n_layers =  1
clip = 10
learning_rate = 1e-2
seq_len = len(train_edges_l)
num_nodes = adj_orig_dense_list[seq_len-1].shape[0]
print("num_nodes", num_nodes)
x_dim = num_nodes
eps = 1e-10
conv_type='GCN'


# creating input tensors

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32))
    x_in_list.append(torch.tensor(x_temp))

x_in = Variable(torch.stack(x_in_list))

adj_label_l = []
for i in range(len(adj_train_l)):
    temp_matrix = adj_train_l[i] 
    adj_label_l.append(torch.tensor(temp_matrix.toarray().astype(np.float32)))


# building model

model = VGRNN(x_dim, h_dim, z_dim, n_layers, eps, conv=conv_type, bias=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training

seq_start = 0
seq_end = seq_len - 3
tst_after = 0

# plot training curves and test results
losses_rec = []
auc_scores_det_val_rec = []
ap_scores_det_val_rec = []
auc_scores_det_test_rec = []
ap_scores_det_test_rec= []

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
        _, _, enc_means, _, _, z_t = model(x_in[seq_end:seq_len]
                                      , edge_idx_list[seq_end:seq_len]
                                      , adj_label_l[seq_end:seq_len]
                                      , hidden_st)
        
        auc_scores_det_val, ap_scores_det_val = get_roc_scores(val_edges_l[seq_end:seq_len]
                                                                , val_edges_false_l[seq_end:seq_len]
                                                                , adj_orig_dense_list[seq_end:seq_len]
                                                                , enc_means)
        
        auc_scores_det_test, ap_scores_det_tes = get_roc_scores(test_edges_l[seq_end:seq_len]
                                                                , test_edges_false_l[seq_end:seq_len]
                                                                , adj_orig_dense_list[seq_end:seq_len]
                                                                , enc_means)
        
    
    print('epoch: ', k)
    print('kld_loss =', kld_loss.mean().item())
    print('nll_loss =', nll_loss.mean().item())
    print('loss =', loss.mean().item())
    if k>=tst_after:
        print('----------------------------------')
        print('Link Detection')
        print('val_link_det_auc_mean', np.mean(np.array(auc_scores_det_val)))
        print('val_link_det_ap_mean', np.mean(np.array(ap_scores_det_val)))
        print('test_link_det_auc_mean', np.mean(np.array(auc_scores_det_test)))
        print('test_link_det_ap_mean', np.mean(np.array(ap_scores_det_tes)))
        print('----------------------------------')

    losses_rec.append(loss.mean().item())
    auc_scores_det_test_rec.append(np.mean(np.array(auc_scores_det_test)))
    ap_scores_det_test_rec.append(np.mean(np.array(ap_scores_det_tes)))
    auc_scores_det_val_rec.append(np.mean(np.array(auc_scores_det_val)))
    ap_scores_det_val_rec.append(np.mean(np.array(ap_scores_det_val)))

    

    
    print('----------------------------------')



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
val_auc_line = ax2.plot(auc_scores_det_val_rec, color='tab:blue', linestyle='--', 
                       label='Val AUC', linewidth=2, alpha=0.8)
val_ap_line = ax2.plot(ap_scores_det_val_rec, color='tab:green', linestyle='--', 
                      label='Val AP', linewidth=2, alpha=0.8)
test_auc_line = ax2.plot(auc_scores_det_test_rec, color='tab:blue', 
                        label='Test AUC', linewidth=2, alpha=0.8)
test_ap_line = ax2.plot(ap_scores_det_test_rec, color='tab:green', 
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
plt.savefig(f'./results/{topic}training_curves.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


