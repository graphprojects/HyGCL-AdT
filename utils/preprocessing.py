from collections import defaultdict, Counter
from itertools import combinations
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


def expand_edge_index(data, edge_th=0):
    """
    args:
        num_nodes: regular nodes. i.e. x.shape[0]
        num_edges: number of hyperedges. not the star expansion edges.

    this function will expand each n2he relations, [[n_1, n_2, n_3],
                                                    [e_7, e_7, e_7]]
    to :
        [[n_1,   n_1,   n_2,   n_2,   n_3,   n_3],
         [e_7_2, e_7_3, e_7_1, e_7_3, e_7_1, e_7_2]]

    and each he2n relations:   [[e_7, e_7, e_7],
                                [n_1, n_2, n_3]]
    to :
        [[e_7_1, e_7_2, e_7_3],
         [n_1,   n_2,   n_3]]

    and repeated for every hyperedge.
    """
    edge_index = data.edge_index
    num_nodes = data.n_x[0].item()
    if hasattr(data, "totedges"):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges[0]

    expanded_n2he_index = []
    #     n2he_with_same_heid = []

    #     expanded_he2n_index = []
    #     he2n_with_same_heid = []

    # start edge_id from the largest node_id + 1.
    cur_he_id = num_nodes
    # keep an mapping of new_edge_id to original edge_id for edge_size query.
    new_edge_id_2_original_edge_id = {}

    # do the expansion for all annotated he_id in the original edge_index
    #     ipdb.set_trace()
    for he_idx in range(num_nodes, num_edges + num_nodes):
        # find all nodes within the same hyperedge.
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

        #         Trim a hyperedge if its size>edge_th
        if edge_th > 0:
            if size_of_he > edge_th:
                continue

        if size_of_he == 1:
            # there is only one node in this hyperedge -> self-loop node. add to graph.
            #             n2he_with_same_heid.append(selected_he)

            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)

            # ====
            #             new_he2n_same_heid = torch.flip(selected_he, dims = [0])
            #             he2n_with_same_heid.append(new_he2n_same_heid)

            #             new_he2n = torch.flip(selected_he, dims = [0])
            #             new_he2n[0] = cur_he_id
            #             expanded_he2n_index.append(new_he2n)

            cur_he_id += 1
            continue

        # -------------------------------
        #         # new_n2he_same_heid uses same he id for all nodes.
        #         new_n2he_same_heid = selected_he.repeat_interleave(size_of_he - 1, dim = 1)
        #         n2he_with_same_heid.append(new_n2he_same_heid)

        # for new_n2he mapping. connect the nodes to all repeated he first.
        # then remove those connection that corresponding to the node itself.
        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # new_edge_ids start from the he_id from previous iteration (cur_he_id).
        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)
        ).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # build a mapping between node and it's corresponding edge.
        # e.g. {n_1: e_7_1, n_2: e_7_2}
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
            cur_he_id += 1

        # create n2he by deleting the self-product edge.
        new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id, tmp_edge_id = (
                new_n2he[0, col_idx].item(),
                new_n2he[1, col_idx].item(),
            )
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)

    #         # ---------------------------
    #         # create he2n from mapping.
    #         new_he2n = np.array([[he_id, node_id] for node_id, he_id in tmp_node_id_2_he_id_dict.items()])
    #         new_he2n = torch.from_numpy(new_he2n.T).to(device = edge_index.device)
    #         expanded_he2n_index.append(new_he2n)

    #         # create he2n with same heid as input edge_index.
    #         new_he2n_same_heid = torch.zeros_like(new_he2n, device = edge_index.device)
    #         new_he2n_same_heid[1] = new_he2n[1]
    #         new_he2n_same_heid[0] = torch.ones_like(new_he2n[0]) * he_idx
    #         he2n_with_same_heid.append(new_he2n_same_heid)

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)
    #     new_he2n_index = torch.cat(expanded_he2n_index, dim = 1)
    #     new_edge_index = torch.cat([new_n2he_index, new_he2n_index], dim = 1)
    # sort the new_edge_index by first row. (node_ids)
    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data


def ConstructV2V(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we DONT allow duplicated edges.
    Instead, we record its corresponding weights.
    We default no self loops so far.
    """
    # # Note that the method below for CE can be memory expensive!!!
    #     new_edge_index = []
    #     for he in np.unique(edge_index[1, :]):
    #         nodes_in_he = edge_index[0, :][edge_index[1, :] == he]
    #         if len(nodes_in_he) == 1:
    #             continue #skip self loops
    #         combs = combinations(nodes_in_he,2)
    #         for comb in combs:
    #             new_edge_index.append([comb[0],comb[1]])

    #     new_edge_index, new_edge_weight = torch.tensor(new_edge_index).type(torch.LongTensor).unique(dim=0,return_counts=True)
    #     data.edge_index = new_edge_index.transpose(0,1)
    #     data.norm = new_edge_weight.type(torch.float)

    # # Use the method below for better memory complexity
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

    # # Now, translate dict to edge_index and norm
    #
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1

    data.edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    data.norm = torch.tensor(new_norm).type(torch.FloatTensor)
    return data


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
    #     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    # num_nodes = data.n_x[0]
    num_nodes = data.n_x
    # num_hyperedges = data.num_hyperedges[0]
    num_hyperedges = data.num_hyperedges
    # if not ((data.n_x[0]+data.num_hyperedges[0]-1) == data.edge_index[0].max().item()):
    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[0].max().item()):
        print("num_hyperedges does not match! 1")
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    data.totedges = num_hyperedges
    return data


def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    data.num_ori_edge = edge_index.shape[1]
    # expanded to list
    # num_nodes = data.n_x[0]
    num_nodes = data.n_x
    # num_hyperedges = data.num_hyperedges[0]
    num_hyperedges = edge_index[1].max() + 1 - num_nodes

    # if not ((data.n_x[0] + data.num_hyperedges[0] - 1) == data.edge_index[1].max().item()):
    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print("num_hyperedges seems not match! 2")
    #     return
    # dict
    skip_node_lst = []
    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # for edge in hyperedge_appear_fre:
    #     if hyperedge_appear_fre[edge] == 1:
    #         # ensure V
    #         skip_node = edge_index[0][torch.where(
    #             edge_index[1] == edge)[0].item()]
    #         skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros((2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes

    # _, sorted_idx = torch.sort(edge_index[0])
    # data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    data.edge_index = edge_index.type(torch.LongTensor)
    return data


def norm_contruction(data, option="all_one", TYPE="V2E"):
    if TYPE == "V2E":
        if option == "all_one":
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == "deg_half_sym":
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1] - cidx, dim=0)
            V_norm = Vdeg ** (-1 / 2)
            E_norm = HEdeg ** (-1 / 2)
            data.norm = V_norm[data.edge_index[0]] * E_norm[data.edge_index[1] - cidx]

    elif TYPE == "V2V":
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True
        )
    return data
