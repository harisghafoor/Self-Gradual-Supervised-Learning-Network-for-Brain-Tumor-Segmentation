import numpy as np
from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# MSE between current and temporal outputs
def mse_loss(out1, out2):
    quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
    return quad_diff / out1.data.nelement()


def masked_crossentropy(out, labels):
    cond = labels >= 0
    nnz = torch.nonzero(cond)
    nbsup = len(nnz)
    # check if labeled samples in batch, return 0 if none
    if nbsup > 0:
        masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
        masked_labels = labels[cond]
        loss = F.cross_entropy(masked_outputs, masked_labels)
        return loss, nbsup
    return Variable(torch.FloatTensor([0.0]), requires_grad=False), 0


def temporal_loss(out1, out2, w, labels):

    # try:

    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    # except Exception as e:
    #     print(e)
    #     sup_loss = nbsup = unsup_loss =  torch.zeros(1)
        
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup
