#!/usr/bin/env python
# encoding: utf-8


import argparse
import os


def parser_data():

    pre_parser = argparse.ArgumentParser()

    pre_parser.add_argument("--dname", default="cora", type=str)
    pre_args, unknown = pre_parser.parse_known_args()
    data = pre_args.dname

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default=os.getcwd(), type=str)
    parser.add_argument(
        "--data_dir", default=os.path.join(os.getcwd(), "data"), type=str
    )
    parser.add_argument("--runs", default=5, type=int)
    parser.add_argument("--epochs", default=500, type=int)

    parser.add_argument("--cuda", default=0, choices=[-1, 0, 1, 2, 3], type=int)
    parser.add_argument("--train_prop", type=float, default=0.1)
    parser.add_argument("--valid_prop", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--aug_ratio", type=float, default=0.3)
    parser.add_argument(
        "--aug1",
        type=str,
        default="edge",
        help="mask|edge|hyperedge|mask_col|adapt|adapt_feat|adapt_edge",
    )
    parser.add_argument(
        "--aug2",
        type=str,
        default="edge",
        help="mask|edge|hyperedge|mask_col|adapt|adapt_feat|adapt_edge",
    )
    parser.add_argument(
        "--edge", type=str, default="sum", help="sum|mean|max|propogate|typical"
    )
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--mode", type=str, default="TM")
    parser.add_argument("--aggregate", default="mean", choices=["sum", "mean"])
    parser.add_argument("--normtype", default="all_one")
    parser.add_argument("--add_self_loop", action="store_false")
    parser.add_argument(
        "--normalization", default="ln"
    )  # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument("--deepset_input_norm", default=True)
    parser.add_argument("--GPR", action="store_false")
    parser.add_argument("--LearnMask", action="store_false")
    parser.add_argument("--num_features", default=0, type=int)  # Placeholder
    parser.add_argument("--num_classes", default=0, type=int)  # Placeholder

    parser.add_argument("--exclude_self", action="store_true")
    parser.add_argument("--PMA", action="store_true")

    parser.add_argument("--output_heads", default=1, type=int)  # Placeholder
    parser.add_argument("--p_hidden", type=int, default=128)
    parser.add_argument("--p_layer", type=int, default=-1)
    parser.add_argument(
        "--aug",
        type=str,
        default="hyperedge",
        help="mask|edge|hyperedge|mask_col|adapt|adapt_feat|adapt_edge",
    )
    parser.add_argument("--use_cpu", default=False, type=bool)
    parser.add_argument("--add_e", action="store_true", default=False)
    parser.add_argument("--permute_self_edge", action="store_true", default=False)
    parser.add_argument("--linear", action="store_true", default=False)
    parser.add_argument("--sub_size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cl_loss", type=str, default="InfoNCE")
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.add_argument("--display_step", type=int, default=1)

    if data == "cora":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=1, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=4, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=256, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=128, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.0, type=float)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="cora")
        parser.add_argument("--batch", default=False)
        parser.add_argument("--batch_size", default=20480)
        parser.add_argument("--tau_lowerbound", default=0.1, type=float)
        parser.add_argument("--m_l", type=float, default=1)
    elif data == "citeseer":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=2, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=8, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=512, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=256, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.0, type=float)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="citeseer")
        parser.add_argument("--batch", default=False)
        parser.add_argument("--batch_size", default=1024)
        parser.add_argument("--tau_lowerbound", default=0.1, type=float)
        parser.add_argument("--m_l", type=float, default=1)
    elif data == "coauthor_cora":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=2, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=8, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=128, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=128, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.0, type=float)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="coauthor_cora")
        parser.add_argument("--batch", default=False)
        parser.add_argument("--batch_size", default=1024)
        parser.add_argument("--tau_lowerbound", default=0.1, type=float)
        parser.add_argument("--m_l", type=float, default=1)
    elif data == "zoo":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=2, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=1, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=64, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=64, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.00001, type=float)
        parser.add_argument("--lr", default=0.01, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="zoo")
        parser.add_argument("--batch", default=False)
        parser.add_argument("--batch_size", default=1024)
        parser.add_argument("--tau_lowerbound", default=0.1, type=float)
        parser.add_argument("--m_l", type=float, default=1)
    elif data == "Mushroom":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=2, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=1, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=128, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=128, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.0, type=float)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="Mushroom")
        parser.add_argument("--batch", default=False)
        parser.add_argument("--batch_size", default=1024)
        parser.add_argument("--tau_lowerbound", default=0.1, type=float)
        parser.add_argument("--m_l", type=float, default=1)
    elif data == "NTU2012":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=2, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=1, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=256, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=256, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.0, type=float)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="NTU2012")
        parser.add_argument("--batch", default=False)
        parser.add_argument("--batch_size", default=1024)
        parser.add_argument("--tau_lowerbound", default=0.1, type=float)
        parser.add_argument("--m_l", type=float, default=1.0)
    elif data == "Twitter-HyDrug":
        parser.add_argument("--All_num_layers", default=1, type=int)
        parser.add_argument(
            "--MLP_num_layers", default=2, type=int
        )  # How many layers of encoder
        parser.add_argument("--feature_noise", default="0.0", type=str)
        parser.add_argument("--heads", default=1, type=int)  # Placeholder
        parser.add_argument(
            "--Classifier_num_layers", default=1, type=int
        )  # How many layers of decoder
        parser.add_argument(
            "--MLP_hidden", default=256, type=int
        )  # Encoder hidden units
        parser.add_argument(
            "--Classifier_hidden", default=256, type=int
        )  # Decoder hidden units
        parser.add_argument("--wd", default=0.0, type=float)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--method", default="AllDeepSets")
        parser.add_argument("--dname", default="Twitter-HyDrug")
        # parser.add_argument("--batch", default=True)
        parser.add_argument("--batch_size", default=128)
        parser.add_argument("--tau_lowerbound", default=0.2, type=float)
        parser.add_argument("--m_l", type=float, default=1)
    #     Use the line below for .py file

    args = parser.parse_args()
    return args
