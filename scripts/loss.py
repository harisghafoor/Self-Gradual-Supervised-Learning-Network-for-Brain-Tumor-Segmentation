import numpy as np
from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# import torch.ten


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
        # inputs = F.sigmoid(inputs)
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
    cross_entropy_loss = F.binary_cross_entropy_with_logits(
        output, target, reduction="mean"
    )
    dice_loss = DiceLoss()(output, target)
    total_loss = cross_entropy_loss + dice_loss
    return total_loss


def consistency_loss(pred: torch.tensor, target: torch.tensor):
    """
    Compute the consistency loss between the predicted and target masks.
    Assumes that the  tensors are of shape (batch_size, 1, height, width).
    Args:
        pred (tensor): predicted mask
        target (tensor): target mask

    Returns:
        tensor: consistency loss
    """
    assert pred.shape == target.shape
    return ((pred - target) ** 2).mean(dim=[2, 3]).squeeze(1).mean(dim=0)


def get_labelled_examples_in_batch(
    batch_labels: torch.tensor, CONSTANT_IDENTIFIER=-1, debug=False
):
    """
    Find the labeled examples in the batch to compute supervised loss.

    Args:
        batch_labels (torch.Tensor): Tensor containing the labels for each image in the batch.
                                      The shape is expected to be (batch_size, channels, height, width).
        CONSTANT_IDENTIFIER (int, optional): The value indicating unlabeled images. Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - list of int: Indices of labeled examples in the batch.
            - torch.Tensor: Tensor of labeled examples in the batch.

    Example usage:
        >>> batch_labels = torch.full((10, 3, 256, 256), -1)  # Example batch with all images initially set to -1
        >>> # Modify some images to be labeled (not all -1)
        >>> batch_labels[0] = torch.randn(3, 256, 256)  # Example labeled image
        >>> batch_labels[2] = torch.randn(3, 256, 256)  # Example labeled image
        >>> labelled_indices, labelled_examples = get_labelled_examples_in_batch(batch_labels)
        >>> print(labelled_indices)
        >>> print(labelled_examples)
    """
    # Find indices of images where all pixel values are -1
    condition_for_unlabelled_images = (batch_labels == CONSTANT_IDENTIFIER).all(
        dim=(1, 2, 3)
    )
    # Identify labeled examples by negating the condition for unlabelled images
    labelled_example_indices_in_batch = (
        torch.nonzero(~condition_for_unlabelled_images).squeeze(1).tolist()
    )
    # Extract the labeled examples using the indices
    # labelled_examples_in_batch = batch_labels[labelled_example_indices_in_batch]

    # Debug prints
    if debug:
        # Number of labeled examples
        nbsup = len(labelled_example_indices_in_batch)
        print("Number of labelled examples in the batch:", nbsup)
        print(
            "Indices of labelled examples in the batch:",
            labelled_example_indices_in_batch,
        )

    return labelled_example_indices_in_batch


def pi_model_loss(
    actual_mask: torch.tensor,
    pred_mask: torch.tensor,
    ensemble_mask: torch.tensor,
    weight: torch.float32,
    device: torch.device,
    debug=False,
):
    """Compute the loss for the PI Model

    Args:
        actual_mask (tensor): _description_
        pred_mask (tensor): _description_
        ensemble_mask (tensor): _description_
    """
    idx = get_labelled_examples_in_batch(batch_labels=actual_mask)
    if len(idx) != 0:
        valid_pred_masks = pred_mask[idx]
        labelled_examples = actual_mask[idx]
        if debug:
            print(labelled_examples.shape, valid_pred_masks.shape)
        sup_loss = compute_loss(output=valid_pred_masks, target=labelled_examples)
        if debug:
            print("Number of Valid Predicted Masks:", len(valid_pred_masks))
            print("Supervised Loss:", sup_loss)
            # sup_loss = ssl.supervised_loss(model, input_image, label)
    else:
        sup_loss = torch.tensor(0.0, requires_grad=True).to(device)

    unsup_loss = consistency_loss(pred=pred_mask, target=ensemble_mask)
    if debug:
        print("Unsupervised Loss:", unsup_loss)
    total_loss = sup_loss + weight * unsup_loss
    return total_loss, sup_loss, unsup_loss, len(idx)
