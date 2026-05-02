import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        return F.kl_div(log_probs, targets, reduction='batchmean')

class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, outputs, targets):
        p = targets
        q = F.softmax(outputs, dim=1)
        m = 0.5 * (p + q)
        
        kl_pm = F.kl_div(torch.log(m + 1e-9), p, reduction='batchmean')
        kl_qm = F.kl_div(torch.log(m + 1e-9), q, reduction='batchmean')
        return 0.5 * (kl_pm + kl_qm)

class CustomEntropyWeightedLoss(nn.Module):
    def __init__(self):
        super(CustomEntropyWeightedLoss, self).__init__()

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        entropy = -torch.sum(targets * torch.log(targets + 1e-9), dim=1)
        weights = entropy / (entropy.mean() + 1e-9)
        kl_pointwise = F.kl_div(log_probs, targets, reduction='none').sum(dim=1)
        return (kl_pointwise * weights).mean()