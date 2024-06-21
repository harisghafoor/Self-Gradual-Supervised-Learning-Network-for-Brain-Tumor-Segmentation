import numpy as np
from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


def temporal_loss(out1, out2, w, labels, device):

    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum(
            (F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2
        ).to(device)
        return quad_diff / out1.data.nelement()

    def masked_crossentropy(out, labels):
        cond = labels >= 0
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        # check if labeled samples in batch, return 0 if none
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
            masked_labels = labels[cond]
            loss = F.cross_entropy(masked_outputs, masked_labels).to(device)
            return loss, nbsup
        return Variable(torch.FloatTensor([0.0]), requires_grad=False).to(device), 0

    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup


def calculate_metrics(y_true, y_pred, threshold):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > threshold
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > threshold
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """Returns the Batched Dice Coefficient loss function.

        Args:
            weight (float, optional): . Defaults to None.
            size_average (bool, optional): . Defaults to True.
            inputs (tensor): Batched Input Tensor
            targets (tensor): Batched Target Tensor
        """
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        num_samples = inputs.size()[0]
        # flatten label and prediction tensors
        inputs = inputs.view(num_samples, -1)
        targets = targets.view(num_samples, -1)

        intersection = (inputs * targets).sum(axis=1)
        dice_coefficient = (2.0 * intersection + smooth) / (
            inputs.sum(axis=1) + targets.sum(axis=1) + smooth
        )

        dice_coefficient = dice_coefficient.mean(axis=0)
        dice_loss = 1 - dice_coefficient

        return dice_loss


def compute_loss(output, target):
    """Compute the average batch loss between the output and target masks.
    Args:
        output (tensor): output from the model
        target (tensor): target mask

    Returns:
        tensor: loss
    """
    output = torch.sigmoid(output)
    cross_entropy_loss = F.binary_cross_entropy_with_logits(
        output, target, reduction="mean"
    )
    dice_loss = DiceLoss()(output, target)
    total_loss = cross_entropy_loss + dice_loss
    return total_loss
