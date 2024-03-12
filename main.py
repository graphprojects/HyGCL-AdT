import os, os.path as osp
import sys
import torch
import time
import numpy as np
import copy
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn, torch.nn.functional as F

from utils.parser_data import parser_data
from utils.augmentation import aug, create_hypersubgraph
from utils.dataLoader import load_data

from utils.models import SetGNN
from utils.helper import (
    fix_seed,
    rand_train_test_idx,
    count_parameters,
    Logger,
)
from utils.evaluation import evaluate, evaluate_finetune, eval_acc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def main(args):
    start = time.time()
    # # Part 1: Load data
    data = load_data(args)
    #  Get Splits
    split_idx_lst = []
    for run in range(args.runs):  # how many runs
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop
        )  # train test split
        split_idx_lst.append(split_idx)  # the list of data splitting

    # # Part 2: Load model

    if args.method == "AllDeepSets":
        args.PMA = False
        args.aggregate = "add"
    elif args.method == "AllSetTransformer":
        pass
    else:
        raise ValueError("Method not implemented")
    if args.LearnMask:
        model = SetGNN(args, data.norm)
    else:
        model = SetGNN(args)

    # put things to device
    if args.cuda in [0, 1, 2, 3]:
        device = torch.device(
            "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")

    model = model.to(device)
    data = data.to(device)
    data_pre = copy.deepcopy(data)
    num_params = count_parameters(model)

    # # Part 3: Main. Training + Evaluation

    logger = Logger(args.runs, args)

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    model.train()

    ### Training Loop ###
    he_index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        he_index[he].append(i)
    runtime_list = []
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        best_val = float("-inf")

        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            # cl loss
            if args.m_l:
                if data_pre.n_x <= args.sub_size:
                    data_sub = data_pre
                else:
                    data_sub = create_hypersubgraph(data_pre, args)  ###

                data_sub1 = copy.deepcopy(data_sub)
                data_sub2 = copy.deepcopy(data_sub)
                if args.aug1 == "subgraph" or args.aug1 == "drop":
                    node_num = data_sub1.x.shape[0]
                    n_walk = 128 if node_num > 128 else 8
                    start = torch.randint(
                        0, node_num, size=(n_walk,), dtype=torch.long
                    ).to(device)
                    data_sub1 = data_sub1.to(device)
                    data_aug1, nodes1, hyperedge1 = aug(
                        data_sub1, args.aug1, args, start
                    )

                else:
                    data_sub1 = data_sub1.to(device)
                    cidx = data_sub1.edge_index[1].min()
                    data_sub1.edge_index[1] -= cidx
                    # must starts from zero
                    data_sub1 = data_sub1.to(device)
                    data_aug1, nodes1, hyperedge1 = aug(data_sub1, args.aug1, args)
                    data_aug1 = data_aug1.to(device)

                    data_aug1.edge_index[1] += cidx
                hyperedge_idx1 = torch.tensor(
                    list(
                        range(
                            data_aug1.x.shape[0], data_aug1.x.shape[0] + len(hyperedge1)
                        )
                    )
                ).to(device)

                def edge_embed(idx, data_aug):

                    return data_aug.edge_index[0][
                        torch.where(data_aug.edge_index[1] == idx)[0]
                    ]

                data1_node2edge = [edge_embed(i, data_aug1) for i in hyperedge_idx1]

                data1_edgeidx_l, data1_node2edge_sample = [], []
                for i in range(len(data1_node2edge)):
                    if torch.numel(data1_node2edge[i]) > 0:
                        data1_edgeidx_l.append(i)
                        data1_node2edge_sample.append(data1_node2edge[i])

                data1_edgeidx = data_aug1.x.shape[0] + torch.tensor(data1_edgeidx_l).to(
                    device
                )

                pgd1 = torch.rand_like(data_aug1.x)
                data_aug_pgd1 = data_aug1.clone()
                data_aug_pgd1.x = data_aug1.x + pgd1

                out1, edge1 = model.forward_global_local(
                    data_aug_pgd1, data1_node2edge_sample, data1_edgeidx, device
                )

                if args.aug2 == "subgraph" or args.aug2 == "drop":
                    node_num = data_sub2.x.shape[0]
                    n_walk = 128 if node_num > 128 else 8
                    start = torch.randint(
                        0, node_num, size=(n_walk,), dtype=torch.long
                    ).to(device)
                    data_sub2 = data_sub2.to(device)
                    data_aug2, nodes2, hyperedge2 = aug(
                        data_sub2, args.aug2, args, start
                    )
                else:
                    data_sub2 = data_sub2.to(device)
                    cidx = data_sub2.edge_index[1].min()
                    data_sub2.edge_index[1] -= cidx
                    # must starts from zero
                    data_sub2 = data_sub2.to(device)
                    data_aug2, nodes2, hyperedge2 = aug(data_sub2, args.aug2, args)
                    data_aug2 = data_aug2.to(device)
                    data_aug2.edge_index[1] += cidx

                hyperedge_idx2 = torch.tensor(
                    list(
                        range(
                            data_aug2.x.shape[0], data_aug2.x.shape[0] + len(hyperedge2)
                        )
                    )
                ).to(device)

                data2_node2edge = [edge_embed(i, data_aug2) for i in hyperedge_idx2]

                data2_edgeidx_l, data2_node2edge_sample = [], []
                for i in range(len(data2_node2edge)):
                    if torch.numel(data2_node2edge[i]) > 0:
                        data2_edgeidx_l.append(i)
                        data2_node2edge_sample.append(data2_node2edge[i])

                data2_edgeidx = data_aug2.x.shape[0] + torch.tensor(data2_edgeidx_l)

                pgd2 = torch.rand_like(data_aug2.x)
                data_aug_pgd2 = data_aug2.clone()
                data_aug_pgd2.x = data_aug2.x + pgd2

                out2, edge2 = model.forward_global_local(
                    data_aug_pgd2, data2_node2edge_sample, data2_edgeidx, device
                )

                com_sample = list(set(nodes1) & set(nodes2))
                dict_nodes1, dict_nodes2 = {
                    value: i for i, value in enumerate(nodes1)
                }, {value: i for i, value in enumerate(nodes2)}
                com_sample1, com_sample2 = [
                    dict_nodes1[value] for value in com_sample
                ], [dict_nodes2[value] for value in com_sample]
                loss_cl = model.get_loss(out1, out2, [com_sample1, com_sample2])

                com_edge = list(
                    set(data1_edgeidx.tolist()) & set(data2_edgeidx.tolist())
                )

                dict_edge1, dict_edge2 = {
                    value: i for i, value in enumerate(data1_edgeidx.tolist())
                }, {value: i for i, value in enumerate(data2_edgeidx.tolist())}

                com_edge1, com_edge2 = [dict_edge1[value] for value in com_edge], [
                    dict_edge2[value] for value in com_edge
                ]
                loss_cl_gl = model.get_loss(
                    edge1, edge2, [com_edge1, com_edge2]
                )

            else:
                loss_cl = 0
            # sup loss
            if args.linear:
                out = model.forward_finetune(data)
            else:
                out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss += args.m_l * (loss_cl + loss_cl_gl)

            loss.backward()
            optimizer.step()

            if args.linear:
                result = evaluate_finetune(model, data, split_idx, eval_func)
            else:
                result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:6])

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Train Loss: {loss:.4f}, "
                    f"Valid Loss: {result[7]:.4f}, "
                    f"Test  Loss: {result[8]:.4f}, "
                    f"Train Acc: {100 * result[0]:.2f}%, "
                    f"Valid Acc: {100 * result[1]:.2f}%, "
                    f"Test  Acc: {100 * result[2]:.2f}%, "
                    f"Train F1: {100 * result[3]:.2f}%, "
                    f"Valid F1: {100 * result[4]:.2f}%, "
                    f"Test  F1: {100 * result[5]:.2f}%, "
                )

        end_time = time.time()
        runtime_list.append(end_time - start_time)

        logger.print_statistics(run)
        end = time.time()
        mins = (end - start) / 60
        print("The running time is {}".format(mins))

    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val_acc, best_test_acc, test_f1 = logger.print_statistics()
    res_root = osp.join(args.root_dir, "result")
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f"{res_root}/{args.dname}.csv"
    print(f"Saving results to {filename}")
    with open(filename, "a+") as write_obj:

        cur_line = f"{args.method}_{args.lr}_{args.wd}_{args.heads}_{str(args.dropout)}"
        cur_line += f",{args.aug1,args.aug2}"
        cur_line += f",{best_val_acc.mean():.3f} ± {best_val_acc.std():.3f}"
        cur_line += f",{best_test_acc.mean():.3f} ± {best_test_acc.std():.3f}"
        cur_line += f",{test_f1.mean():.3f} ± {test_f1.std():.3f}"
        cur_line += f",{num_params}, {avg_time:.2f}s, {std_time:.2f}s"
        cur_line += f",{avg_time // 60}min{(avg_time % 60):.2f}s"
        cur_line += f"\n"
        write_obj.write(cur_line)

    all_args_file = f"{res_root}/all_args_{args.dname}.csv"
    with open(all_args_file, "a+") as f:
        f.write(str(args))
        f.write("\n")
    end = time.time()
    mins = (end - start) / 60
    print("The running time is {}".format(mins))


if __name__ == "__main__":
    args = parser_data()
    fix_seed(args.seed)
    if args.dname.startswith("Twitter-HyDrug"):
        use_cpu = input(
            "Due to the limited GPU memory, do you want to move some calculations to CPU? (y/n)\n"
        )
        if use_cpu.lower() == "y":
            print("Move some calculations to CPU!")
            args.use_cpu = True
        else:
            args.use_cpu = False
    main(args)
