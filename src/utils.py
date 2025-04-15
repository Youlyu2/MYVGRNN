import math
import pickle as pkl

import networkx as nx
import numpy as np

# from torch_geometric import nn as tgnn
import scipy.sparse as sp
import torch
import torch.utils
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torch_geometric.data import Data
from torch_geometric.utils import (
    train_test_split_edges,
)


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    
    return out


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


# masking functions

def get_labeled_nodes(metadata, device='cpu'):
    label_tensor = torch.tensor(metadata['id2label'], dtype=torch.long, device=device)
    is_gt_tensor = torch.tensor(metadata['id2is_gt'], dtype=torch.long, device=device)
    
    # 条件：标签是 0（opposing）或 2（supportive），并且是 ground truth
    mask = ((label_tensor == 0) | (label_tensor == 2)) & (is_gt_tensor == 1)
    labeled_nodes = torch.where(mask)[0]

    # 将标签从 [0, 2] 转成 [0, 1] （即 0: opposing, 1: supportive）
    labels = label_tensor[labeled_nodes]
    labels = (labels == 2).long()  # 2 → 1, 0 → 0

    return labeled_nodes, labels

def mask_edges_det(adjs_list):
    adj_train_l, train_edges_l, val_edges_l = [], [], []
    val_edges_false_l, test_edges_l, test_edges_false_l = [], [], []
    edges_list = []
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        
        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0
        
        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        tmp_data = Data(edge_index=torch.LongTensor(edges_all.T), num_nodes=adj.shape[0])
        split_data = train_test_split_edges(tmp_data, val_ratio=0.05, test_ratio=0.1)

        # num_test = int(np.floor(edges.shape[0] / 10.))
        # num_val = int(np.floor(edges.shape[0] / 20.))
        
        # all_edge_idx = list(range(edges.shape[0]))
        # np.random.shuffle(all_edge_idx)
        # val_edge_idx = all_edge_idx[:num_val]
        # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        # test_edges = edges[test_edge_idx]
        # val_edges = edges[val_edge_idx]
        # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        train_edges = np.array(split_data.train_pos_edge_index.T.tolist())
        val_edges = np.array(split_data.val_pos_edge_index.T.tolist())
        test_edges = np.array(split_data.test_pos_edge_index.T.tolist())
        test_edges_false = np.array(split_data.test_neg_edge_index.T.tolist())
        val_edges_false = np.array(split_data.val_neg_edge_index.T.tolist())
        edges_list.append(edges)
        
        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

    

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        adj_train_l.append(adj_train)
        train_edges_l.append(train_edges)
        val_edges_l.append(val_edges)
        val_edges_false_l.append(val_edges_false)
        test_edges_l.append(test_edges)
        test_edges_false_l.append(test_edges_false)

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l


def mask_edges_prd(adjs_list):
    pos_edges_l , false_edges_l = [], []
    edges_list = []
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        
        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0
        
        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_false = int(edges.shape[0])
        
        pos_edges_l.append(edges)
        
        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)
        
        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)
        
        false_edges_l.append(edges_false)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l

def mask_edges_prd_new(adjs_list, adj_orig_dense_list):
    pos_edges_l , false_edges_l = [], []
    edges_list = []
    
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    adj = adjs_list[0]
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_false = int(edges.shape[0])

    pos_edges_l.append(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edges_false = []
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    assert ~ismember(edges_false, edges_all)    
    false_edges_l.append(np.asarray(edges_false))
    
    for i in range(1, len(adjs_list)):
        edges_pos = np.transpose(np.asarray(np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i-1])>0)))
        num_false = int(edges_pos.shape[0])
        
        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0
        
        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        
        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])
        
        assert ~ismember(edges_false, edges_all)
        
        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)
    
    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


# evaluation function

def get_roc_scores(edges_pos, edges_neg, adj_orig_dense_list, embs):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    auc_scores = []
    ap_scores = []
    
    for i in range(len(edges_pos)):
        # Predict on test set of edges
        emb = embs[i].detach().numpy()
        adj_rec = np.dot(emb, emb.T)
        adj_orig_t = adj_orig_dense_list[i]
        preds = []
        pos = []
        for e in edges_pos[i]:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig_t[e[0], e[1]])
            
        preds_neg = []
        neg = []
        for e in edges_neg[i]:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig_t[e[0], e[1]])
        
        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        auc_scores.append(roc_auc_score(labels_all, preds_all))
        ap_scores.append(average_precision_score(labels_all, preds_all))

    return auc_scores, ap_scores



def get_node_classification_scores(logits, labels, labeled_nodes):
    """
    logits: [N, num_classes] — raw classifier output
    labels: [num_labeled_nodes] — ground truth labels
    labeled_nodes: [num_labeled_nodes] — indices of nodes to evaluate
    """
    preds = torch.argmax(logits[labeled_nodes], dim=1).cpu().numpy()
    true = labels.cpu().numpy()

    original_preds = preds.copy()
    reversed_preds = torch.tensor([1 - p for p in preds])

    origin_res = {
        'accuracy': accuracy_score(true, original_preds),
        'precision': precision_score(true, original_preds, average='binary'),
        'recall': recall_score(true, original_preds, average='binary'),
        'f1': f1_score(true, original_preds, average='binary')
    }

    reversed_res = {
        'accuracy': accuracy_score(true, reversed_preds),
        'precision': precision_score(true, reversed_preds, average='binary'),
        'recall': recall_score(true, reversed_preds, average='binary'),
        'f1': f1_score(true, reversed_preds, average='binary')
    }
    if origin_res['f1'] > reversed_res['f1']:
        return origin_res
    else:
        return reversed_res




def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false