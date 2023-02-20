# The Sinkhorn normalization is adapted from https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py

import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    """
    Samples arbitrary-shaped standard gumbel variables.
    Args:
        shape (list): list of integers.
        eps (float, optional): epsilon for numerical stability. Default 1e-20.
    Returns:
        (torch.Tensor): a sample of standard Gumbel random variables
    """
    # Sample Gumble from uniform distribution
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def sinkhorn(log_alpha, n_iter=20, slack=False):
    """
    Perform incomplete Sinkhorn normalization to log_alpha
    By a theorem by Sinkhorn and Knopp, a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (element wise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    Args:
        log_alpha (torch.Tensor): a batch of 2D tensor of shape [B, V, V]
        n_iter (int, optional): number of iterations. (Default 20)
        slack (bool, optional): augment matrix with slack row and column. Default False.
    Returns:
        (torch.Tensor): a tensor of close-to-doubly-stochastic matrices.
    """
    if not slack:
        for _ in range(n_iter):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-2, keepdim=True))
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-1, keepdim=True))
    else:
        # augment log_alpha
        log_alpha_padded = F.pad(log_alpha.unsqueeze(dim=1), pad=(0, 1, 0, 1), mode='constant', value=0.0).squeeze(dim=1)
        for _ in range(n_iter):
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - torch.logsumexp(log_alpha_padded[:, :, :-1], dim=-2, keepdim=True),
                log_alpha_padded[:, :, [-1]]
            ), dim=-1)

            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - torch.logsumexp(log_alpha_padded[:, :-1, :], dim=-1, keepdim=True),
                log_alpha_padded[:, [-1], :]
            ), dim=-2)
        log_alpha = log_alpha_padded[:, :-1, :-1]

    return torch.exp(log_alpha)


def gumbel_sinkhorn(log_alpha, temp=1.0, noise_factor=0, n_iter=10, slack=False):
    """
    Random doubly-stochastic matrices via gumbel noise.
    In the zero-temperature limit sinkhorn (log_alpha/temp) approaches
    a permutation matrix. Therefore, for low temperatures this method
    can be seen as an approximate sampling of permutation matrices.
    The deterministic case (noise_factor=0) is also interesting: it can be
    shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
    permutation matrix, the solution of the
    matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
    Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
    as approximate solving of a matching problem.
    Args:
        log_alpha (torch.Tensor): a single/batch of 2D tensor of shape [V, V] or [B, V, V]
        temp (float, optional): temperature parameter. Default 1.0.
        noise_factor (float, optional) scaling factor for the gumbel samples
        (and the absence of randomness, with noise_factor=0). Default 0.
        n_iter (int, optional): number of sinkhorn iterations. Default 20.
        slack (bool, optional): whether augment matrix with slack row and column. Default False
    Return:
        sink (torch.Tensor): a 3D tensor of close-doubly-stochastic-matrix [B, n_samples, V, V]
    """

    if noise_factor == 0:
        noise = 0.0
    else:
        noise = noise_factor * sample_gumbel(log_alpha.shape)
        noise = noise.to(device=log_alpha.device, dtype=log_alpha.dtype)

    log_alpha_w_noise = log_alpha + noise
    log_alpha_w_noise = log_alpha_w_noise / temp

    sink = sinkhorn(log_alpha_w_noise, n_iter=n_iter, slack=slack)

    return sink
