import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)


class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))  # positive samples

        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()

        # compute log prob
        exp_logits = torch.exp(logits)  # 분모 부분

        # mask out positives
        logits = logits * mask  # 분자 부분
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # 왼쪽 부분은 log 랑 exp 없어지므로 logits 그대로 사용

        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)  # positive 개수 |P_i|
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)  # 분모가 0 이 될 수 없으니까

        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()  # |P_i| 로 나눠주는 부분

        loss = -1 * pos_logits.mean()

        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return (1 - self.alpha) * ce_loss + cl_loss + self.alpha * cl_loss


class DualCLLoss(SupConLoss):

    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        ce_loss = self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)  # loss of theta (classifier representation)
        cl_loss_2 = self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)  # loss of z (input representation)
        # return (1 - self.alpha) * ce_loss + self.alpha * (0.5 * cl_loss_1 + 0.5 * cl_loss_2)
        return ce_loss + self.alpha * (0.5 * cl_loss_1 + 0.5 * cl_loss_2)
