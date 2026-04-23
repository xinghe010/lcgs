import torch
import torch.nn as nn
import torch.nn.functional as F

from .mgda import MGDA

class LCGSTrainer(nn.Module):

    def __init__(self, model, model_type='astnn', lambda_param=0.1,
                 mcs_margin=0.5, use_mgda=False):
        super(LCGSTrainer, self).__init__()
        self.model = model
        self.model_type = model_type
        self.lambda_param = lambda_param
        self.mcs_margin = mcs_margin
        self.mgda = MGDA()
        self.use_mgda = use_mgda
        self.mse_loss = nn.MSELoss()

    def compute_lcgs_loss(self, pred_prob, y, cwj, mcs_equal, sample_weights=None):
        epsilon = 0.1
        y_float = y.view(-1)
        cwj_flat = cwj.view(-1)
        pred_flat = pred_prob.view(-1)

        if sample_weights is None:
            w = torch.ones_like(y_float)
        else:
            w = sample_weights.view(-1)

        r5_weights = cwj_flat + epsilon
        diff_pos = F.relu(cwj_flat - pred_flat)
        diff_neg = torch.abs(pred_flat)
        l1_error = y_float * diff_pos + (1 - y_float) * diff_neg
        l1_vec = r5_weights * l1_error
        l1_loss = (l1_vec * w).sum() / (w.sum() + 1e-8)

        kappa = 2.0
        mask = (mcs_equal.view(-1) == 1) & (y_float == 1)

        if mask.any():
            masked_prob = pred_flat[mask]
            soft_penalty = torch.sigmoid(kappa * (self.mcs_margin - masked_prob))
            w_mask = w[mask]
            l2_loss = (soft_penalty * w_mask).sum() / (w_mask.sum() + 1e-8)
        else:
            l2_loss = torch.tensor(0.0, device=y.device, requires_grad=True)

        return l1_loss, l2_loss

    def _combine_logic_losses(self, l1_loss, l2_loss):
        if self.use_mgda:
            logic_losses = [l1_loss, l2_loss]
            return self.mgda(logic_losses, list(self.model.parameters()))
        else:
            return 0.5 * l1_loss + 0.5 * l2_loss

    def forward_astnn(self, x1, x2, y, cwj, mcs_equal, sample_weights=None):
        pred_prob = self.model(x1, x2)

        if sample_weights is None:
            w = torch.ones_like(y).view(-1)
        else:
            w = sample_weights.view(-1)

        bce_vec = F.binary_cross_entropy(pred_prob, y, reduction='none').view(-1)
        bce_loss = (bce_vec * w).sum() / (w.sum() + 1e-8)

        l1_loss, l2_loss = self.compute_lcgs_loss(
            pred_prob, y, cwj, mcs_equal, sample_weights
        )
        weighted_logic_loss = self._combine_logic_losses(l1_loss, l2_loss)
        total_loss = bce_loss + self.lambda_param * weighted_logic_loss

        return total_loss, pred_prob

    def forward_ggnn(self, data1, data2, y, cwj, mcs_equal):
        h1 = self.model(data1)
        h2 = self.model(data2)

        cos_sim = F.cosine_similarity(h1, h2)
        main_loss = self.mse_loss(cos_sim, y)

        pred_prob = (cos_sim + 1.0) / 2.0
        target_01 = (y + 1.0) / 2.0

        l1_loss, l2_loss = self.compute_lcgs_loss(
            pred_prob, target_01, cwj, mcs_equal
        )
        weighted_logic_loss = self._combine_logic_losses(l1_loss, l2_loss)
        total_loss = main_loss + self.lambda_param * weighted_logic_loss

        return total_loss, cos_sim

    def forward_gmn(self, data_pack, y, cwj, mcs_equal):
        h1, h2 = self.model(data_pack)

        cos_sim = F.cosine_similarity(h1, h2)
        main_loss = self.mse_loss(cos_sim, y)

        pred_prob = (cos_sim + 1.0) / 2.0
        target_01 = (y + 1.0) / 2.0

        l1_loss, l2_loss = self.compute_lcgs_loss(
            pred_prob, target_01, cwj, mcs_equal
        )
        weighted_logic_loss = self._combine_logic_losses(l1_loss, l2_loss)
        total_loss = main_loss + self.lambda_param * weighted_logic_loss

        return total_loss, cos_sim

    def forward_encoder(self, x1, x2, y, cwj, mcs_equal, sample_weights=None):
        pred_prob = self.model(x1, x2)

        if sample_weights is None:
            w = torch.ones_like(y).view(-1)
        else:
            w = sample_weights.view(-1)

        bce_vec = F.binary_cross_entropy(pred_prob, y, reduction='none').view(-1)
        bce_loss = (bce_vec * w).sum() / (w.sum() + 1e-8)

        l1_loss, l2_loss = self.compute_lcgs_loss(
            pred_prob, y, cwj, mcs_equal, sample_weights
        )
        weighted_logic_loss = self._combine_logic_losses(l1_loss, l2_loss)
        total_loss = bce_loss + self.lambda_param * weighted_logic_loss

        return total_loss, pred_prob

    def forward_siamese(self, data1, data2, y, cwj, mcs_equal):
        h1 = self.model(data1)
        h2 = self.model(data2)

        cos_sim = F.cosine_similarity(h1, h2)
        main_loss = self.mse_loss(cos_sim, y)

        pred_prob = (cos_sim + 1.0) / 2.0
        target_01 = (y + 1.0) / 2.0

        l1_loss, l2_loss = self.compute_lcgs_loss(
            pred_prob, target_01, cwj, mcs_equal
        )
        weighted_logic_loss = self._combine_logic_losses(l1_loss, l2_loss)
        total_loss = main_loss + self.lambda_param * weighted_logic_loss

        return total_loss, cos_sim

    def forward(self, *args, **kwargs):
        if self.model_type == 'astnn':
            return self.forward_astnn(*args, **kwargs)
        elif self.model_type == 'ggnn':
            return self.forward_ggnn(*args, **kwargs)
        elif self.model_type == 'gmn':
            return self.forward_gmn(*args, **kwargs)
        elif self.model_type == 'encoder':
            return self.forward_encoder(*args, **kwargs)
        elif self.model_type == 'siamese':
            return self.forward_siamese(*args, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
