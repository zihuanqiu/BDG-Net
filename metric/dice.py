from pytorch_lightning.metrics import Metric
import numpy as np
import torch


def dice(preds: torch.Tensor, target: torch.Tensor, th=0.5, if_sigmoid=True):
    if preds.shape != target.shape:
        preds = preds.squeeze(1)

    assert preds.shape == target.shape

    if not isinstance(preds, torch.FloatTensor):
        preds = preds.float()

    if if_sigmoid:
        preds = preds.sigmoid()

    preds = preds.view(-1)
    target = (target > 0).float().view(-1)

    p = (preds > th).float()
    inter = (p * target).float().sum().item()
    union = (p + target).float().sum().item()
    return 2.0 * inter / (union + 1e-6)


def mean_dice(preds: torch.Tensor, target: torch.Tensor, if_sigmoid=True):
    if preds.shape != target.shape:
        preds = preds.squeeze(1)

    assert preds.shape == target.shape

    if not isinstance(preds, torch.FloatTensor):
        preds = preds.float()

    if if_sigmoid:
        preds = preds.sigmoid()

    preds = preds.view(-1)
    target = (target > 0).float().view(-1)

    mdice = 0

    for th in np.arange(0, 1+1/255, 1/255):

        p = (preds > th).float()
        inter = (p * target).float().sum().item()
        union = (p + target).float().sum().item()

        mdice += 2.0 * inter / (union + 1e-6)

    return mdice/len(np.arange(0, 1+1/255, 1/255))
