import torch
import pickle
import os, os.path as osp

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import load_npz

from torch_geometric.data import Data
from torch_sparse import coalesce

from .preprocessing import (
    ExtractV2E,
    Add_Self_Loops,
    expand_edge_index,
    norm_contruction,
)

# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer


def save_data_to_pickle(data, p2root="../data/", file_name=None):
    """
    if file name not specified, use time stamp.
    """
    #     now = datetime.now()
    #     surfix = now.strftime('%b_%d_%Y-%H:%M')
    surfix = "star_expansion_dataset"
    if file_name is None:
        tmp_data_name = "_".join(["Hypergraph", surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, "bw") as f:
        pickle.dump(data, f)
    return p2he_StarExpan


def load_LE_dataset(path=None, dataset="ModelNet40", train_percent=0.025):
    # load edges, features, and labels.
    print("Loading {} dataset...".format(dataset))

    file_name = f"{dataset}.content"
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print("load features")

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f"{dataset}.edges"
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered, dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    print("load edges")

    projected_features = torch.FloatTensor(np.array(features.todense()))

    # From adjacency matrix to edge_list
    edge_index = edges.T
    #     ipdb.set_trace()
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    # ipdb.set_trace()

    # build torch data class
    data = Data(
        #             x = projected_features,
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes],
    )

    # ipdb.set_trace()
    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    #     ipdb.set_trace()

    #     # generate train, test, val mask.
    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = num_he

    return data


def load_citation_dataset(
    path="../hyperGCN/data/", dataset="cora", train_percent=0.025
):
    """
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    """
    print(f"Loading hypergraph dataset from hyperGCN: {dataset}")

    # first load node features:
    with open(osp.join(path, dataset, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()
    # then load node labels:
    with open(osp.join(path, dataset, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    print()

    # The last, load hypergraph.
    with open(osp.join(path, dataset, "hypergraph.pickle"), "rb") as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array(
        [node_list + edge_list, edge_list + node_list], dtype=np.int32
    )
    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features, edge_index=edge_index, y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)

    return data


def load_twitter_data(path):
    hyperedge_file_path = osp.join(path, "hyperedges.npz")
    H = load_npz(hyperedge_file_path).astype(np.int32).toarray()

    embed_file_path = osp.join(path, f"features.txt")
    X = np.loadtxt(embed_file_path)
    X = X[:, 1:]
    # # load labels
    label_file_path = osp.join(path, "label.txt")
    Y = np.loadtxt(label_file_path).astype(np.int32)
    Y = Y.argmax(1)
    # label_edge_file_path = osp.join(path, "label_edge.txt")
    # label_edge = np.loadtxt(label_edge_file_path)
    # assert X.shape[0] == H.shape[1] == Y.shape[0], "The dim does not match"

    features = torch.FloatTensor(X)
    labels = torch.LongTensor(Y)
    # edge_type = torch.LongTensor(label_edge)

    hyperedge_index, node_idx = H.nonzero()
    hyperedge_index += X.shape[0]
    row_indices = np.concatenate((node_idx, hyperedge_index))
    col_indices = np.concatenate((hyperedge_index, node_idx))
    edge_index = np.stack((row_indices, col_indices), axis=0)
    edge_index = torch.LongTensor(edge_index)

    dataset = Data(x=features, edge_index=edge_index, y=labels)
    # dataset.H = H
    # dataset.edge_type = edge_type.argmax(1)
    dataset.train_percent = 0.5
    dataset.num_hyperedges = H.shape[0]
    return dataset


class dataset_Hypergraph(InMemoryDataset):
    def __init__(
        self,
        root="../data/pyg_data/hypergraph_dataset_updated/",
        name=None,
        p2raw=None,
        train_percent=0.01,
        feature_noise=None,
        transform=None,
        pre_transform=None,
    ):

        existing_dataset = [
            "zoo",
            "NTU2012",
            "Mushroom",
            "coauthor_cora",
            "cora",
            "citeseer",
            "Twitter-HyDrug",
        ]
        if name not in existing_dataset:
            raise ValueError(
                f"name of hypergraph dataset must be one of: {existing_dataset}"
            )
        else:
            self.name = name

        self.feature_noise = feature_noise

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!'
            )

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root
        self.myraw_dir = osp.join(root, self.name, "raw")
        self.myprocessed_dir = osp.join(root, self.name, "processed")

        super(dataset_Hypergraph, self).__init__(
            osp.join(root, name), transform, pre_transform
        )

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    # @property
    # def raw_dir(self):
    #     return osp.join(self.root, self.name, 'raw')

    # @property
    # def processed_dir(self):
    #     return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f"{self.name}_noise_{self.feature_noise}"]
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f"data_noise_{self.feature_noise}.pt"]
        else:
            file_names = ["data.pt"]
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                print(p2f)
                print(self.p2raw)
                print(self.name)

                if self.name in ["cora", "citeseer"]:
                    tmp_data = load_citation_dataset(
                        path=self.p2raw,
                        dataset=self.name,
                        train_percent=self._train_percent,
                    )
                elif self.name in ["coauthor_cora"]:
                    assert "coauthorship" in self.p2raw
                    dataset_name = self.name.split("_")[-1]
                    tmp_data = load_citation_dataset(
                        path=self.p2raw,
                        dataset=dataset_name,
                        train_percent=self._train_percent,
                    )

                elif self.name == "Twitter-HyDrug":
                    tmp_data = load_twitter_data(path=self.p2raw)

                else:
                    tmp_data = load_LE_dataset(
                        path=self.p2raw,
                        dataset=self.name,
                        train_percent=self._train_percent,
                    )

                _ = save_data_to_pickle(
                    tmp_data, p2root=self.myraw_dir, file_name=self.raw_file_names[0]
                )
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, "rb") as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


def load_data(args):
    ### Load and preprocess data ###
    existing_dataset = [
        "zoo",
        "NTU2012",
        "Mushroom",
        "coauthor_cora",
        "cora",
        "citeseer",
        "Twitter-HyDrug",
    ]
    if args.dname in existing_dataset:
        dname = args.dname
        p2raw = osp.join(args.data_dir, "raw_data")
    
        if dname in ["cora", "citeseer"]:
            p2raw = osp.join(p2raw, "cocitation")
        elif dname in ["coauthor_cora", "coauthor_dblp"]:
            p2raw = osp.join(p2raw, "coauthorship")
        elif dname in ["Twitter-HyDrug"]:
            p2raw = osp.join(p2raw, "Twitter-HyDrug")
        dataset = dataset_Hypergraph(
            name=dname,
            root=osp.join(args.data_dir, "pyg_data", "hypergraph_dataset_updated"),
            p2raw=p2raw,
        )
    data = dataset.data
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    data.n_x = torch.tensor([data.x.shape[0]])
    data.num_hyperedges = torch.tensor([data.num_hyperedges])

    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
    if args.exclude_self:
        data = expand_edge_index(data)
    data = norm_contruction(data, option=args.normtype)

    return data
