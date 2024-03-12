import copy
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.utils import (
    k_hop_subgraph,
    subgraph,
    degree,
)
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def create_hypersubgraph(data, args):
    sub_size = args.sub_size
    node_size = int(data.n_x[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        sample_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source"
    )
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    x = data.x[sample_nodes]
    data_sub = Data(x=x, edge_index=sub_edge_index)
    data_sub.n_x = torch.tensor([sub_size])
    data_sub.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size])
    data_sub.norm = 0
    data_sub.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data_sub.num_ori_edge = sub_edge_index.shape[1] - sub_size
    return data_sub


def permute_edges(data, aug_ratio, permute_self_edge, args):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    # if not permute_self_edge:
    permute_num = int((edge_num - node_num) * aug_ratio)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()

    if args.add_e:
        idx_add_1 = np.random.choice(node_num, permute_num)
        idx_add_2 = np.random.choice(int(data.num_hyperedges), permute_num)
        idx_add = np.stack((idx_add_1, idx_add_2), axis=0)
    edge2remove_index = np.where(edge_index[1] < data.num_hyperedges[0].item())[0]
    edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges[0].item())[0]
    # edge2remove_index = np.where(edge_index[1] < data.num_hyperedges)[0]
    # edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges)[0]

    try:

        edge_keep_index = np.random.choice(
            edge2remove_index, (edge_num - node_num) - permute_num, replace=False
        )

    except ValueError:

        edge_keep_index = np.random.choice(
            edge2remove_index, (edge_num - node_num) - permute_num, replace=True
        )

    edge_after_remove1 = edge_index[:, edge_keep_index]
    edge_after_remove2 = edge_index[:, edge2keep_index]
    if args.add_e:
        edge_index = np.concatenate(
            (
                edge_after_remove1,
                edge_after_remove2,
            ),
            axis=1,
        )
    else:
        edge_index = np.concatenate((edge_after_remove1, edge_after_remove2), axis=1)
    data.edge_index = torch.tensor(edge_index)
    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def permute_hyperedges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_remove_index_dict = {ind: i for i, ind in enumerate(edge_remove_index)}

    edge_remove_index_all = [
        i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict
    ]
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove

    data.edge_index = torch.tensor(edge_index)

    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def adapt(data, aug_ratio, aug):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        index[he].append(i)
    # edge
    edge_index_orig = copy.deepcopy(data.edge_index)
    drop_weights = degree_drop_weights(data.edge_index, hyperedge_num)
    edge_index_1 = drop_edge_weighted(
        data.edge_index,
        drop_weights,
        p=aug_ratio,
        threshold=0.7,
        h=hyperedge_num,
        index=index,
    )

    # feature
    edge_index_ = data.edge_index
    node_deg = degree(edge_index_[0])
    feature_weights = feature_drop_weights(data.x, node_c=node_deg)
    x_1 = drop_feature_weighted(data.x, feature_weights, aug_ratio, threshold=0.7)
    if aug == "adapt_edge":
        data.edge_index = edge_index_1
    elif aug == "adapt_feat":
        data.x = x_1
    else:
        data.edge_index = edge_index_1
        data.x = x_1
    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p

    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.0

    return x


def degree_drop_weights(edge_index, h):
    edge_index_ = edge_index
    deg = degree(edge_index_[1])[:h]
    deg_col = deg
    s_col = torch.log(deg_col)
    weights = (s_col - s_col.min() + 1e-9) / (s_col.mean() - s_col.min() + 1e-9)
    return weights


def feature_drop_weights(x, node_c):
    x = torch.abs(x).to(torch.float32)
    # 100 x 2012 mat 2012-> 100
    w = x.t() @ node_c
    w = w.log() + 1e-7
    # s = (w.max() - w) / (w.max() - w.mean())
    s = (w - w.min()) / (w.mean() - w.min())
    return s


def drop_edge_weighted(
    edge_index, edge_weights, p: float, h, index, threshold: float = 1.0
):
    _, edge_num = edge_index.size()
    edge_weights = (edge_weights + 1e-9) / (edge_weights.mean() + 1e-9) * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    # keep probability
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    edge_remove_index = np.array(list(range(h)))[sel_mask.cpu().numpy()]
    edge_remove_index_all = []
    for remove_index in edge_remove_index:
        edge_remove_index_all.extend(index[remove_index])
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    return edge_index


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token

    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                data.edge_index[1][
                    torch.where(
                        (data.edge_index[1] < node_num + data.num_hyperedges)
                        & (data.edge_index[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def drop_nodes(data, aug_ratio):
    node_size = int(data.n_x[0].item())
    sub_size = int(node_size * (1 - aug_ratio))
    # hyperedge_size = int(data.num_hyperedges[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        sample_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source"
    )
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    sub_edge_index_orig = copy.deepcopy(sub_edge_index)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    data.x = data.x[sample_nodes]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([sub_size])
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size])
    data.norm = 0
    data.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data.num_ori_edge = sub_edge_index.shape[1] - sub_size

    return (
        data,
        sorted(set(sub_nodes[:sub_size].cpu().numpy())),
        sorted(
            set(
                sub_edge_index_orig[1][
                    torch.where(
                        (sub_edge_index_orig[1] < node_size + hyperedge_size)
                        & (sub_edge_index_orig[1] > node_size - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def subgraph_aug(data, aug_ratio, start):
    n_walkLen = 16
    node_num, _ = data.x.size()
    he_num = data.totedges.item()
    edge_index = data.edge_index

    device = edge_index.device

    row, col = edge_index
    adj = SparseTensor(
        row=torch.cat([row, col]),
        col=torch.cat([col, row]),
        sparse_sizes=(node_num + he_num, he_num + node_num),
    )

    node_idx = adj.random_walk(start.flatten(), n_walkLen).view(-1)
    sub_nodes = node_idx.unique()
    sub_nodes.sort()

    node_size = int(data.n_x[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sub_edge_index, _, hyperedge_idx = subgraph(
        sub_nodes, edge_index, relabel_nodes=False, return_edge_mask=True
    )

    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    sub_edge_index_orig = copy.deepcopy(sub_edge_index)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    node_keep_idx = sub_nodes[torch.where(sub_nodes < node_size)[0]]
    data.x = data.x[node_keep_idx]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([node_keep_idx.size(0)])
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * node_keep_idx.size(0)])
    data.norm = 0
    data.totedges = torch.tensor(sub_nodes.size(0) - node_keep_idx.size(0))
    data.num_ori_edge = sub_edge_index.shape[1] - node_keep_idx.size(0)
    return (
        data,
        sorted(set(node_keep_idx.cpu().numpy().tolist())),
        sorted(
            set(
                sub_edge_index_orig[1][
                    torch.where(
                        (sub_edge_index_orig[1] < node_size + hyperedge_size)
                        & (sub_edge_index_orig[1] > node_size - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def aug(data, aug_type, args, start=None):
    data_aug = copy.deepcopy(data)
    if aug_type == "mask":
        data_aug, sample_nodes, sample_hyperedge = mask_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "edge":
        data_aug, sample_nodes, sample_hyperedge = permute_edges(
            data_aug, args.aug_ratio, args.permute_self_edge, args
        )
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "hyperedge":
        data_aug = permute_hyperedges(data_aug, args.aug_ratio)
    elif aug_type == "subgraph":
        data_aug, sample_nodes, sample_hyperedge = subgraph_aug(
            data_aug, args.aug_ratio, start
        )
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "drop":
        data_aug, sample_nodes, sample_hyperedge = drop_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "none":
        return data_aug
    elif "adapt" in aug_type:
        data_aug = adapt(data_aug, args.aug_ratio, aug_type)
    else:
        raise ValueError(f"not supported augmentation")
    return data_aug
