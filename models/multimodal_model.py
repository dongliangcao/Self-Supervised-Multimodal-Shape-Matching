import torch

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import fmap2pointmap


@MODEL_REGISTRY.register()
class MultiModalModel(BaseModel):
    """
    Contrast FMNet Model
    """

    def __init__(self, opt):
        super(MultiModalModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_mesh_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])
        feat_mesh_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])

        # feature extractor for point cloud
        feat_pc_x = self.networks['feature_extractor'](data_x['verts'], None)
        feat_pc_y = self.networks['feature_extractor'](data_y['verts'], None)

        # compute functional map
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_trans_x = data_x['evecs_trans']
        evecs_trans_y = data_y['evecs_trans']

        Cxy, Cyx = self.networks['fmap_net'](feat_mesh_x, feat_mesh_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # compute surfmnet loss
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

        # compute the alignment loss |evecs_x*Cyx - Pxy*evecs_y|
        if 'align_loss' in self.losses:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_pc_y, feat_pc_x)
            self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy.detach(), Pyx, evecs_y, evecs_x)

        # compute contrastive loss
        if 'contrast_loss' in self.losses:
            self.loss_metrics['l_contrast'] = self.losses['contrast_loss'](feat_mesh_x, feat_pc_x) + \
                                              self.losses['contrast_loss'](feat_mesh_y, feat_pc_y)

    def optimize_parameters(self):
        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # start record
        timer.start()
        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        if 'faces' in data_x and 'faces' in data_y:
            # compute functional map
            evals_x = data_x['evals']
            evals_y = data_y['evals']
            evecs_trans_x = data_x['evecs_trans']
            evecs_trans_y = data_y['evecs_trans']
            Cxy, _ = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

            # convert functional map to point-to-point map
            evecs_x = data_x['evecs']
            evecs_y = data_y['evecs']
            p2p = fmap2pointmap(Cxy.squeeze(), evecs_x.squeeze(),
                                evecs_y.squeeze())

            # compute Pyx from functional map
            Pyx = evecs_y.squeeze() @ Cxy.squeeze() @ evecs_trans_x.squeeze()
        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_y, feat_x).squeeze()
            p2p = Pyx.argmax(dim=-1)

        # finish record
        timer.record()
        return p2p, Pyx

    def compute_permutation_matrix(self, feat_x, feat_y):
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)
        return Pxy
