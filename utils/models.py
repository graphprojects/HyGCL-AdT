import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from .layers import MLP, HalfNLHconv


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


class SimCLRTau(nn.Module):
    def __init__(self, args):
        super(SimCLRTau, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.fc1 = nn.Linear(args.p_hidden, 200)
        self.fc2 = nn.Linear(200, args.MLP_hidden)
        self.tau = args.t
        self.low = args.tau_lowerbound
        self.pre_grad = 0.0

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def uni_loss(self, z, t=2):
        return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

    def momentum_update(
        self,
        x_start,
        z,
        eta= 0.001,
        rho= 0.7,
    ):
        if x_start < self.low:
            return x_start
        x = x_start
        grad = -self.uni_loss(z).item()
        self.pre_grad = self.pre_grad * rho + 1 / grad
        x -= self.pre_grad * eta

        return x

    def forward(self, z1, z2, com_nodes1, com_nodes2):
        z1 = self.project(z1)
        z2 = self.project(z2)

        self.tau = self.momentum_update(self.tau, z1)

        f = lambda x: torch.exp(x / self.tau)

        if self.args.use_cpu:
            refl_sim = f(sim(z1, z1).cpu())
            between_sim = f(sim(z1, z2).cpu())
        else:
            refl_sim = f(sim(z1, z1))
            between_sim = f(sim(z1, z2))


        if self.args.cl_loss == "InfoNCE":
            return -torch.log(
                between_sim[com_nodes1, com_nodes2]
                / (
                    refl_sim.sum(1)[com_nodes1]
                    + between_sim.sum(1)[com_nodes1]
                    - refl_sim.diag()[com_nodes1]
                )
            ).cuda()
        elif self.args.cl_loss == "JSD":
            N = refl_sim.shape[0]
            pos_score = (np.log(2) - F.softplus(-between_sim.diag())).mean()
            neg_score_1 = F.softplus(-refl_sim) + refl_sim - np.log(2)
            neg_score_1 = torch.sum(neg_score_1) - torch.sum(neg_score_1.diag())
            neg_score_2 = torch.sum(F.softplus(-between_sim) + between_sim - np.log(2))
            neg_score = (neg_score_1 + neg_score_2) / (N * (2 * N - 1))
            res = neg_score - pos_score
            return res.cuda()

    def whole_batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / T)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(sim(z1[mask], z1))  # [B, N]
            between_sim = f(sim(z1[mask], z2))  # [B, N]

            losses.append(
                -torch.log(
                    between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    / (
                        refl_sim.sum(1)
                        + between_sim.sum(1)
                        - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    )
                )
            )

    def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / T)
        indices = np.arange(0, num_nodes)
        np.random.shuffle(indices)

        i = 0
        mask = indices[i * batch_size : (i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]
        loss = -torch.log(
            between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
            / (
                refl_sim.sum(1)
                + between_sim.sum(1)
                - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
            )
        )

        return loss


class SetGNN(nn.Module):
    def __init__(self, args, norm=None, sig=False):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
        #         V_in_dim = V_dict['in_dim']
        #         V_enc_hid_dim = V_dict['enc_hid_dim']
        #         V_dec_hid_dim = V_dict['dec_hid_dim']
        #         V_out_dim = V_dict['out_dim']
        #         V_enc_num_layers = V_dict['enc_num_layers']
        #         V_dec_num_layers = V_dict['dec_num_layers']

        #         E_in_dim = E_dict['in_dim']
        #         E_enc_hid_dim = E_dict['enc_hid_dim']
        #         E_dec_hid_dim = E_dict['dec_hid_dim']
        #         E_out_dim = E_dict['out_dim']
        #         E_enc_num_layers = E_dict['enc_num_layers']
        #         E_dec_num_layers = E_dict['dec_num_layers']

        #         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
        self.args = args
        self.sig = sig
        self.args = args
        #         Now define V2EConvs[i], V2EConvs[i] for ith layers
        #         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
        #         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(
                in_channels=args.num_features,
                hidden_channels=args.Classifier_hidden,
                out_channels=args.num_classes,
                num_layers=args.Classifier_num_layers,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False,
            )
        else:
            self.V2EConvs.append(
                HalfNLHconv(
                    in_dim=args.num_features,
                    hid_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=self.InputNorm,
                    heads=args.heads,
                    attention=args.PMA,
                )
            )
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(
                HalfNLHconv(
                    in_dim=args.MLP_hidden,
                    hid_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=self.InputNorm,
                    heads=args.heads,
                    attention=args.PMA,
                )
            )
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers - 1):
                self.V2EConvs.append(
                    HalfNLHconv(
                        in_dim=args.MLP_hidden,
                        hid_dim=args.MLP_hidden,
                        out_dim=args.MLP_hidden,
                        num_layers=args.MLP_num_layers,
                        dropout=self.dropout,
                        Normalization=self.NormLayer,
                        InputNorm=self.InputNorm,
                        heads=args.heads,
                        attention=args.PMA,
                    )
                )
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(
                    HalfNLHconv(
                        in_dim=args.MLP_hidden,
                        hid_dim=args.MLP_hidden,
                        out_dim=args.MLP_hidden,
                        num_layers=args.MLP_num_layers,
                        dropout=self.dropout,
                        Normalization=self.NormLayer,
                        InputNorm=self.InputNorm,
                        heads=args.heads,
                        attention=args.PMA,
                    )
                )
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))

            if self.GPR:
                self.MLP = MLP(
                    in_channels=args.num_features,
                    hidden_channels=args.MLP_hidden,
                    out_channels=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
                self.GPRweights = Linear(self.All_num_layers + 1, 1, bias=False)
                self.classifier = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.Classifier_hidden,
                    out_channels=args.num_classes,
                    num_layers=args.Classifier_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
            else:
                self.classifier = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.Classifier_hidden,
                    out_channels=args.num_classes,
                    num_layers=args.Classifier_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )

            # pretrain
            if args.p_layer > 0:
                pre_layer = args.p_layer
            else:
                pre_layer = args.Classifier_num_layers
            if args.p_hidden > 0:
                pre_hidden = args.p_hidden
            else:
                pre_hidden = args.Classifier_hidden

            self.proj_head = MLP(
                in_channels=args.MLP_hidden,
                hidden_channels=pre_hidden,
                out_channels=pre_hidden,
                num_layers=pre_layer,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False,
            )
            self.linear = nn.Linear(args.MLP_hidden, args.num_classes)
            self.decoder = nn.Sequential(
                nn.Linear(args.MLP_hidden, args.MLP_hidden),
                nn.ReLU(),
                nn.Linear(args.MLP_hidden, 2),
            )
            self.edge = nn.Linear(args.MLP_hidden + pre_hidden, pre_hidden)
            self.simclr_tau = SimCLRTau(args)

    #         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
    #         However, in general this can be arbitrary.

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)
        return x

    def forward_link(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            # if not self.sig:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x_he = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x_he, p=self.dropout, training=self.training)
                x_node = F.relu(
                    self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                )
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x_node, p=self.dropout, training=self.training)
        return x_he, x_node

    def forward_finetune(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.linear(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.linear(x)

        return x

    def forward_global_local(
        self, data, node2edge, sample_edge_idx, device, aug_weight=None
    ):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.proj_head(x)
        else:
            if self.args.aug != "none":
                x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):

                h1 = F.relu(
                    self.V2EConvs[i](x, edge_index, norm, self.aggr, aug_weight)
                )

                h1_d = F.dropout(h1, p=self.dropout, training=self.training)
                h2 = F.relu(
                    self.E2VConvs[i](
                        h1_d, reversed_edge_index, norm, self.aggr, aug_weight
                    )
                )

                h2_d = F.dropout(h2, p=self.dropout, training=self.training)
            x = self.proj_head(h2_d)

            h1 = h1[sample_edge_idx]

            e_embed = [
                torch.sum(x[node2edge[i]], dim=0, keepdim=True)
                for i in range(len(node2edge))
            ]
            edge_embed = self.edge(
                torch.cat((h1, torch.stack(e_embed).squeeze()), dim=1)
            )

        return x, edge_embed



    def get_loss(self, h1, h2, com_nodes):

        l1 = self.simclr_tau(h1, h2,  com_nodes[0], com_nodes[1])
        l2 = self.simclr_tau(h2, h1, com_nodes[1], com_nodes[0])

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
