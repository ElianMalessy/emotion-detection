import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl

class FocalLoss(nn.Module):
    def __init__(self, data, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        emotion_counts = data.group_by("emotion").count()
        total_samples = emotion_counts["count"].sum()
        emotion_frequencies = emotion_counts.with_columns(
            (pl.col("count") / total_samples).alias("frequency")
        )
        emotion_alphas = emotion_frequencies.with_columns(
            (1 - pl.col("frequency")).alias("alpha")
        )

        self.alpha_dict = emotion_alphas["alpha"]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        targets_list = targets.tolist()
        alpha_values = torch.tensor([self.alpha_dict[target] for target in targets_list]).to(inputs.device)
        
        focal_loss = (alpha_values * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
