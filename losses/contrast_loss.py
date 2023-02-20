import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PointInfoNCELoss(nn.Module):
    def __init__(self, loss_weight=1.0, tau=0.07, normalize=True):
        super(PointInfoNCELoss, self).__init__()
        assert loss_weight >= 0, f'loss weight should be non-negative, but get: {loss_weight}'
        assert tau > 0, f'tau should be positive, but get: {tau}'
        self.loss_weight = loss_weight
        self.tau = tau
        self.normalize = normalize

    def forward(self, feat_x, feat_y):
        """
        Forward pass
        Args:
            feat_x (torch.Tensor): feature vector of data x. [B, V, C].
            feat_y (torch.Tensor): feature vector of data y. [B, V, C].
        Returns:
            loss (torch.Tensor): loss.
        """
        assert feat_x.shape == feat_y.shape, f'Both data shapes should be equal, but {feat_x.shape} != {feat_y.shape}'
        if self.loss_weight > 0:
            if self.normalize:
                feat_x = F.normalize(feat_x, p=2, dim=-1)
                feat_y = F.normalize(feat_y, p=2, dim=-1)
            logits = torch.bmm(feat_x, feat_y.transpose(1, 2)) / self.tau  # [B, V, V]
            B, V = logits.shape[:2]
            labels = torch.arange(0, V, device=logits.device, dtype=torch.long).repeat(B, 1)
            loss = F.cross_entropy(logits, labels)

            return self.loss_weight * loss
        else:
            return 0.0
