# %%
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from model import VGRNN
from torch.autograd import Variable
from torch_geometric.utils import (
    negative_sampling,
    remove_isolated_nodes,
    subgraph,
    to_networkx,
)

# from torch_geometric import nn as tgnn
from utils import (
    get_labeled_nodes,
    get_node_classification_scores,
    get_roc_scores,
    mask_edges_det,
)

# %%
torch.autograd.set_detect_anomaly(True)
topic = "edca/cumulative"

seed = 3
np.random.seed(seed)

# utility functions

# adj_time_list = torch.load(f'../data/{topic}/adj_time_list.pt', weights_only=False)
# adj_orig_dense_list = torch.load(f'../data/{topic}/adj_orig_dense_list.pt')
adj_time_list = []
adj_orig_dense_list = []
gdatas = torch.load(f"../data/{topic}/gdatas.pt", weights_only=False)
for i in range(len(gdatas)):
    gdata = gdatas[i]
    G = to_networkx(gdata, to_undirected=True, remove_self_loops=True)
    sp_adj = nx.adjacency_matrix(G)
    dense_adj = sp_adj.todense()
    adj_time_list.append(sp_adj)
    adj_orig_dense_list.append(torch.tensor(dense_adj, dtype=torch.float32))

# %%
# masking edges

adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l = mask_edges_det(
    adj_time_list
)

# creating edge list

edge_idx_list = []

for i in range(len(train_edges_l)):
    edge_idx_list.append(torch.tensor(np.transpose(train_edges_l[i]), dtype=torch.long))

# %%
# hyperparameters

h_dim = 32
z_dim = 2
n_layers = 2
clip = 10
learning_rate = 1e-2
seq_len = len(train_edges_l)
num_nodes = adj_orig_dense_list[seq_len - 1].shape[0]
x_dim = num_nodes
eps = 1e-10
conv_type = "GCN"


# creating input tensors

x_in = torch.stack([torch.eye(num_nodes, dtype=torch.float32) for _ in range(seq_len)])


adj_label_l = []
for i in range(len(adj_train_l)):
    temp_matrix = adj_train_l[i]
    adj_label_l.append(torch.tensor(temp_matrix.toarray().astype(np.float32)))

# %%
# building model

model = VGRNN(x_dim, h_dim, z_dim, n_layers, eps, conv=conv_type, bias=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training

seq_start = 0
seq_end = seq_len - 3
tst_after = 0

# plot training curves and test results
losses_rec = []
f1_scores_node_cls_rec = []
auc_scores_det_val_rec = []
ap_scores_det_val_rec = []
auc_scores_det_test_rec = []
ap_scores_det_test_rec = []

for k in range(1000):
    # training

    optimizer.zero_grad()
    start_time = time.time()
    # %%
    kld_loss, nll_loss, cls_loss, _, _, hidden_st, all_z_t = model(
        x_in[seq_start:seq_end],
        edge_idx_list[seq_start:seq_end],
        adj_orig_dense_list[seq_start:seq_end],
        [gdata.train_asser_nodes for gdata in gdatas[seq_start:seq_end]],
        [gdata.y for gdata in gdatas[seq_start:seq_end]],
    )

    loss = kld_loss + nll_loss
    loss.backward()
    optimizer.step()

    nn.utils.clip_grad_norm_(model.parameters(), clip)

    if k >= tst_after:
        _, _, _, enc_means, _, _, z_t = model(
            x_in[seq_end:seq_len],
            edge_idx_list[seq_end:seq_len],
            adj_label_l[seq_end:seq_len],
            [gdata.train_asser_nodes for gdata in gdatas[seq_end:seq_len]],
            [gdata.y for gdata in gdatas[seq_end:seq_len]],
            hidden_st,
        )

        # get test nodes
       

        # auc_scores_det_val, ap_scores_det_val = get_roc_scores(
        #     val_edges_l[seq_end:seq_len],
        #     val_edges_false_l[seq_end:seq_len],
        #     adj_orig_dense_list[seq_end:seq_len],
        #     enc_means,
        # )

        auc_scores_det_test, ap_scores_det_tes = get_roc_scores(
            test_edges_l[seq_end:seq_len],
            test_edges_false_l[seq_end:seq_len],
            adj_orig_dense_list[seq_end:seq_len],
            enc_means,
        )

        # node classification
        cls_metrics = [get_node_classification_scores(all_z_t[i][gdata.test_asser_nodes], gdata.y[gdata.test_asser_nodes]-1)[0] for i, gdata in enumerate(gdatas[seq_end:seq_len])]


    print("epoch: ", k)
    print("kld_loss =", kld_loss.mean().item())
    print("nll_loss =", nll_loss.mean().item())
    print("loss =", loss.mean().item())
    print("cls_loss =", cls_loss.mean().item())
    if k >= tst_after:
        print("----------------------------------")
        print("Link Detection")
        print("f1_score_node_cls =", np.mean(np.array([metric['F1(micro)'] for metric in cls_metrics])))
        # print("val_link_det_auc_mean", np.mean(np.array(auc_scores_det_val)))
        # print("val_link_det_ap_mean", np.mean(np.array(ap_scores_det_val)))
        print("test_link_det_auc_mean", np.mean(np.array(auc_scores_det_test)))
        print("test_link_det_ap_mean", np.mean(np.array(ap_scores_det_tes)))
        print("----------------------------------")

    losses_rec.append(loss.mean().item())
    f1_scores_node_cls_rec.append(np.mean(np.array([metric['F1(micro)'] for metric in cls_metrics])))
    auc_scores_det_test_rec.append(np.mean(np.array(auc_scores_det_test)))
    ap_scores_det_test_rec.append(np.mean(np.array(ap_scores_det_tes)))
    # auc_scores_det_val_rec.append(np.mean(np.array(auc_scores_det_val)))
    # ap_scores_det_val_rec.append(np.mean(np.array(ap_scores_det_val)))

    print("----------------------------------")


import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8")

# Create figure with constrained layout
fig, ax1 = plt.subplots(figsize=(10, 6), layout="constrained")

# Plot loss on primary y-axis
color = "tab:red"
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Loss", color=color, fontsize=12)
loss_line = ax1.plot(losses_rec, color=color, label="Loss", linewidth=2, alpha=0.8)
ax1.tick_params(axis="y", labelcolor=color)

# Create secondary y-axis for metrics
ax2 = ax1.twinx()
color = "tab:blue"
ax2.set_ylabel("AUC / AP Score", color=color, fontsize=12)

# Plot metrics with different styles
f1_cls_line = ax2.plot(
    f1_scores_node_cls_rec, color="tab:red", linestyle="--", label="F1 Score", linewidth=2, alpha=0.8
)
val_auc_line = ax2.plot(
    auc_scores_det_val_rec, color="tab:blue", linestyle="--", label="Val AUC", linewidth=2, alpha=0.8
)
val_ap_line = ax2.plot(ap_scores_det_val_rec, color="tab:green", linestyle="--", label="Val AP", linewidth=2, alpha=0.8)
test_auc_line = ax2.plot(auc_scores_det_test_rec, color="tab:blue", label="Test AUC", linewidth=2, alpha=0.8)
test_ap_line = ax2.plot(ap_scores_det_test_rec, color="tab:green", label="Test AP", linewidth=2, alpha=0.8)

# Combine legends from both axes
lines = loss_line + f1_cls_line + val_auc_line + val_ap_line + test_auc_line + test_ap_line
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

# Add title and adjust layout
plt.title(f"Training Curves - {topic}", fontsize=14, pad=20)

# Save figure with high DPI
os.makedirs(f"./results/{topic}", exist_ok=True)
plt.savefig(f"./results/{topic}training_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# %%
