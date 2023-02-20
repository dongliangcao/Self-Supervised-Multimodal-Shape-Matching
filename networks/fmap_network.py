import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY


def _get_mask(evals1, evals2, resolvant_gamma):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
    evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def get_mask(evals1, evals2, resolvant_gamma):
    masks = []
    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    return torch.stack(masks, dim=0)


@NETWORK_REGISTRY.register()
class FunctionalMapNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, bidirectional=False):
        """
        FunctionalMap Network

        Args:
            bidirectional (bool, optional): Indicates whether compute functional map from both x->y and y->x. Default False.
        """
        super().__init__()
        self.bidirectional = bidirectional

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """One pass in functional map net.
        Arguments:
            feat_x (torch.Tensor) -- learned feature 1. Shape: [N, V, C]
            feat_y (Torch.Tensor) -- learned feature 2. Shape: [N, V, C]
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x: (Torch.Tensor) -- inverse eigenfunctions of shape 1. Shape: [N, K, V]
            evecs_trans_y: (Torch.Tensor) -- inverse eigenfunctions of shape 2. Shape: [N, K, V]
        Returns:
            C1: (torch.Tensor) -- matrix representation of functional correspondence (shape 1 -> shape 2). Shape : [N, K, K]
            C2: (torch.Tensor) -- matrix representation of functional correspondence (shape 2 -> shape 1), if bidirectional. Shape: [N, K, K]
        """
        # compute linear operator matrix representation C1 and C2
        # compute coefficient for eigenvector basis
        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        F_hat, G_hat = F_hat.transpose(1, 2), G_hat.transpose(1, 2)
        # solve C * F_hat = G_hat
        Cs_1 = []
        for i in range(feat_x.size(0)):
            C = torch.inverse(F_hat[i].t() @ F_hat[i]) @ F_hat[i].t() @ G_hat[i]
            Cs_1.append(C.t().unsqueeze(0))
        C1 = torch.cat(Cs_1, dim=0)

        if self.bidirectional:
            Cs_2 = []
            for i in range(feat_x.size(0)):
                C = torch.inverse(G_hat[i].t() @ G_hat[i]) @ G_hat[i].t() @ F_hat[i]
                Cs_2.append(C.t().unsqueeze(0))
            C2 = torch.cat(Cs_2, dim=0)
        else:
            C2 = None

        return C1, C2


@NETWORK_REGISTRY.register()
class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)

        if self.bidirectional:
            D = get_mask(evals_y, evals_x, self.resolvant_gamma)  # [B, K, K]

            B_t = B.transpose(1, 2)  # [B, C, K]
            B_B_t = torch.bmm(B, B_t)  # [B, K, K]
            A_B_t = torch.bmm(A, B_t)  # [B, K, K]

            C_i = []
            for i in range(evals_y.shape[1]):
                D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.shape[0])],
                                dim=0)
                C = torch.bmm(torch.inverse(B_B_t + self.lmbda * D_i), A_B_t[:, [i], :].transpose(1, 2))
                C_i.append(C.transpose(1, 2))

            Cyx = torch.cat(C_i, dim=1)
        else:
            Cyx = None

        return Cxy, Cyx
