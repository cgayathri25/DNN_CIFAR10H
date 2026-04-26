import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLabelLoss(nn.Module):
    def __init__(self):
        super(SoftLabelLoss, self).__init__()

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        return F.kl_div(log_probs, targets, reduction='batchmean')