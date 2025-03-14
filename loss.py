import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Balancing factor for class imbalance
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, inputs, targets):
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probabilities (softmax)
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Reduce the loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

