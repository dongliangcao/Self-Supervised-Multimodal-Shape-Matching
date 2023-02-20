import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
from utils.sinkhorn_util import gumbel_sinkhorn


@NETWORK_REGISTRY.register()
class Sinkhorn(nn.Module):
    def __init__(self, lambda_sink, num_sink, slack=False):
        super(Sinkhorn, self).__init__()
        self.lambda_sink = lambda_sink
        self.num_sink = num_sink
        self.slack = slack

    def forward(self, log_alpha):
        sink = gumbel_sinkhorn(log_alpha, temp=self.lambda_sink, n_iter=self.num_sink, slack=self.slack)
        return sink


@NETWORK_REGISTRY.register()
class Similarity(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))

        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret
