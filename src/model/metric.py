from pdb import set_trace as bp

import torch
import numpy as np
import torch.nn.functional as F



def recall_n(output, target):
    """Calculate Recall@N-1
    Zip both, sort in ascending order, so that the first encountered
    1 pair is the least-prob (recall N) pair; set the flag to True here,
    then if the flag is True, the second-encountered pair is the (recall N-1) pair,
    break here because we have identified `len(target) - i` as the
    `抓到N-1個真正報SAR案件所需的名單量`.
    
    Args:
        output (_type_): model prediction
        target (_type_): gold answer

    Returns:
        _type_: _description_
    """
    comb = list(zip(output, target))
    comb.sort(key=lambda x:x[0])
    flag = False
    for i, (out, gt) in enumerate(comb):
        if gt == 1:
            if flag:
                break
            flag = True

    return (sum(target)-1) / (len(target)-i)


def rmse(output, target):
    with torch.no_grad():
        output *= 100
        target *= 100
        mse = F.mse_loss(output, target)
        rmse = torch.sqrt(mse).item()
    return rmse


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)