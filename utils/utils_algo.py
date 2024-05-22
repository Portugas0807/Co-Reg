import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def save_log(self, batch, path):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        with open(os.path.join(path, 'result.log'), 'a+') as f:
            f.write('\t'.join(entries) + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / 800)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_cont_mask(pseudo_target_cont, target_mask_cont, batch_size, args):
    if args.cont_mode == 'UnsupCon':
        return None, None

    if args.cont_mode == 'SupCon':
        mask_eq = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        return mask_eq, None

    # High-confidence Positive
    if args.cont_mode == 'HCP':
        # 1 means positive, 0 means negative
        mask_eq = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        mask_threshold = torch.mm(target_mask_cont[:batch_size], target_mask_cont.T).cuda()
        mask = mask_eq * mask_threshold
        for i in range(batch_size):
            mask[i, i + batch_size] = 1.0
        return mask, None

    # High-confidence Positive and Negative
    if args.cont_mode == 'HCPN':
        # 1 means positive, 0 means negative or ignore
        mask_eq = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        mask_threshold = torch.mm(target_mask_cont[:batch_size], target_mask_cont.T).cuda()
        mask = mask_eq * mask_threshold
        for i in range(batch_size):
            mask[i, i + batch_size] = 1.0

        # -1 means ignore, 0 means others
        ignore = torch.mm(target_mask_cont[:batch_size], target_mask_cont.T - 1)
        return mask, ignore

    if args.cont_mode == 'modified':
        # 1 means positive, 0 means negative or ignore
        mask_eq = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        mask_threshold = torch.mm(target_mask_cont[:batch_size], target_mask_cont.T).cuda()
        mask = mask_eq * mask_threshold
        for i in range(batch_size):
            mask[i, i + batch_size] = 1.0

        # -1 means ignore, 0 means others
        ignore_wPL = (target_mask_cont.T - 1) * (target_mask_cont[:batch_size] * mask_eq)
        ignore_woPL = -1 * ((1 - target_mask_cont[:batch_size]) * mask_eq)
        ignore = ignore_wPL + ignore_woPL
        for i in range(batch_size):
            ignore[i, i + batch_size] = 0.0
        return mask, ignore

