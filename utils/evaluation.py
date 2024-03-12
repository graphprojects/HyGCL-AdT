import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(data.y[split_idx["test"]], out[split_idx["test"]])

    train_f1 = eval_f1(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_f1 = eval_f1(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_f1 = eval_f1(data.y[split_idx["test"]], out[split_idx["test"]])

    #     Also keep track of losses
    train_loss = F.nll_loss(out[split_idx["train"]], data.y[split_idx["train"]])
    valid_loss = F.nll_loss(out[split_idx["valid"]], data.y[split_idx["valid"]])
    test_loss = F.nll_loss(out[split_idx["test"]], data.y[split_idx["test"]])

    return (
        train_acc,
        valid_acc,
        test_acc,
        train_f1,
        valid_f1,
        test_f1,
        train_loss,
        valid_loss,
        test_loss,
        out,
    )


@torch.no_grad()
def evaluate_finetune(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model.forward_finetune(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(data.y[split_idx["test"]], out[split_idx["test"]])

    train_f1 = eval_f1(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_f1 = eval_f1(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_f1 = eval_f1(data.y[split_idx["test"]], out[split_idx["test"]])

    #     Also keep track of losses
    train_loss = F.nll_loss(out[split_idx["train"]], data.y[split_idx["train"]])
    valid_loss = F.nll_loss(out[split_idx["valid"]], data.y[split_idx["valid"]])
    test_loss = F.nll_loss(out[split_idx["test"]], data.y[split_idx["test"]])
    return (
        train_acc,
        valid_acc,
        test_acc,
        train_f1,
        valid_f1,
        test_f1,
        train_loss,
        valid_loss,
        test_loss,
        out,
    )


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    #     ipdb.set_trace()
    #     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_f1(y_true, y_pred):

    return f1_score(
        y_true.detach().cpu(),
        torch.argmax(y_pred, dim=-1).detach().cpu(),
        average="macro",
    )
