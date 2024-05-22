import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class PartialCELoss(nn.Module):
    # partial cross entropy loss
    def __init__(self):
        super(PartialCELoss, self).__init__()

    def forward(self, logits, Y):
        """
        logits: output logits
        Y: partial label matrix (batch_size * num_class, 1.0: candidate, 0.0: non-candidate)
        """
        batch_size, num_class = logits.shape[0], logits.shape[1]
        pred = torch.softmax(logits, dim=1)

        partial_ce_loss = torch.tensor(0, dtype=float).cuda()
        count = torch.tensor(0, dtype=float).cuda()
        num_label = Y.sum(dim=1)
        for i in range(batch_size):
            if num_label[i] <= num_class - 1:
                # partial label examples
                # partial_ce_loss += -torch.sum((1-Y[i]) * torch.log((1-pred[i]) + 1e-5), dim=0)
                partial_ce_loss += -torch.log(torch.sum(Y[i] * pred[i], dim=0) + 1e-5)
                count += 1
        if count > 0:
            partial_ce_loss = partial_ce_loss / count

        return partial_ce_loss


class SemiLoss(object):
    def __init__(self, args):
        super(SemiLoss, self).__init__()
        self.lambda_u = args.lambda_u

    def linear_rampup(self, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return self.lambda_u * float(current)

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        loss = Lx + self.linear_rampup(epoch, warm_up) * Lu
        return loss


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


class MaxiCandLoss(nn.Module):
    # maximum candidate loss: cross-entropy loss of the candidate label with maximum predicted probability
    def __init__(self):
        super(MaxiCandLoss, self).__init__()

    def forward(self, pred_score, Y):
        """
        logits: output logits
        Y: partial label matrix (batch_size * num_class, 1.0: candidate, 0.0: non-candidate)
        """
        maxi_cand_score, _ = torch.max(pred_score * Y, dim=-1)
        loss = -torch.log(maxi_cand_score)

        return loss
    
    
class ProtConLoss(nn.Module):
    def __init__(self, args):
        super(ProtConLoss, self).__init__()
        self.tau_proto = args.tau_proto
        self.lambda_u = args.lambda_u
        self.sim_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()

    def forward(self, sim_logits_x, target_a_x, target_b_x, sim_logits_u, target_a_u, target_b_u, l):
        sim_probas_log_x = torch.log_softmax(torch.div(sim_logits_x, self.tau_proto), dim=1)
        loss_x = l * self.sim_criterion(sim_probas_log_x, target_a_x) + (1 - l) * self.sim_criterion(sim_probas_log_x, target_b_x)

        sim_probas_u = torch.softmax(torch.div(sim_logits_u, self.tau_proto), dim=1)
        loss_u = l * torch.mean((sim_probas_u - target_a_u) ** 2) + (1 - l) * torch.mean((sim_probas_u - target_b_u) ** 2)
        loss = loss_x + self.lambda_u * loss_u
        return loss


class ModifiedSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask, ignore, batch_size, target_mask, args):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            if ignore is not None:
                ig = (exp_logits * ignore).sum(1, keepdim=True)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + ig + 1e-12)
            else:
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            if args.cont_mode == 'HCPN':
                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos * target_mask
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss
