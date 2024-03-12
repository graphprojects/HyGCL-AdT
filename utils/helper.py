import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fix_seed(seed=37):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def rand_train_test_idx(
    label, train_prop=0.1, valid_prop=0.25, ignore_negative=True, balance=False
):
    """Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        if train_prop > 0.1:
            train_num = int(n * train_prop)
            num_10 = int(n * train_prop)
            num_20 = int(n * (train_prop + 0.1))
        else:
            train_num = int(n * train_prop)
            num_10 = int(n * 0.1)
            num_20 = int(n * 0.2)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[num_10:num_20]
        test_indices = perm[num_20:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        # train_idx = labeled_nodes[train_indices]
        # valid_idx = labeled_nodes[val_indices]
        # test_idx = labeled_nodes[test_indices]

        train_idx = labeled_nodes[train_indices.long()]
        valid_idx = labeled_nodes[val_indices.long()]
        test_idx = labeled_nodes[test_indices.long()]

        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    else:
        #         ipdb.set_trace()
        label = label.numpy()
        label_idx_0 = np.where(label == 0)[0]
        label_idx_1 = np.where(label == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        train_idx = np.append(
            label_idx_0[: int(train_prop * len(label_idx_0))],
            label_idx_1[: int(train_prop * len(label_idx_1))],
        )
        valid_idx = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        test_idx = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        split_idx = {
            "train": torch.tensor(train_idx),
            "valid": torch.tensor(valid_idx),
            "test": torch.tensor(test_idx),
        }
    return split_idx


class Logger(object):
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # assert len(result) == 3
        assert len(result) == 6
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest ACC Train: {result[:, 0].max():.2f}")
            print(f"Highest ACC Valid: {result[:, 1].max():.2f}")
            print(f"Highest ACC Test: {result[:, 2].max():.2f}")
            print(f"Highest F1 Train: {result[:, 3].max():.2f}")
            print(f"Highest F1 Valid: {result[:, 4].max():.2f}")
            print(f"Highest F1 Test: {result[:, 5].max():.2f}")
            print(f"  Final Train ACC: {result[argmax, 0]:.2f}")
            print(f"   Final Val ACC: {result[argmax, 1]:.2f}")
            print(f"   Final Test ACC: {result[argmax, 2]:.2f}")
            print(f"  Final Train F1: {result[argmax, 3]:.2f}")
            print(f"   Final Val F1: {result[argmax, 4]:.2f}")
            print(f"   Final Test F1: {result[argmax, 5]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 1])
                best_epoch.append(index)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2_acc = r[r[:, 1].argmax(), 0].item()
                test_acc = r[r[:, 1].argmax(), 2].item()
                train2_f1 = r[r[:, 1].argmax(), 3].item()
                test_f1 = r[r[:, 1].argmax(), 5].item()
                best_results.append(
                    (train1, valid, train2_acc, test_acc, train2_f1, test_f1)
                )

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            print("best epoch:", best_epoch)
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final ACC Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final ACC Test: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 4]
            print(f"  Final F1 Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 5]
            print(f"   Final F1 Test: {r.mean():.2f} ± {r.std():.2f}")
            return best_result[:, 1], best_result[:, 3], best_result[:, 5]

    def plot_result(self, run=None):
        plt.style.use("seaborn")
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f"Run {run + 1:02d}:")
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])
