import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.resnet import MyResNet


class MyModel(nn.Module):

    def __init__(self, args, prior_dist=None):
        super().__init__()

        self.encoder_q = MyResNet(num_class=args.num_class, in_channel=args.in_channel, input_size=args.input_size)

        # initialize pseudo-distribution for DA
        self.pseudo_distribution = torch.ones((args.num_data, args.num_class), dtype=torch.float).cuda() / args.num_class
        self.register_buffer("prior_dist", prior_dist)

        # momentum encoder
        self.encoder_k = MyResNet(num_class=args.num_class, in_channel=args.in_channel, input_size=args.input_size)
        self.init_key_encoder()

        self.moco_m = args.moco_m
        self.moco_queue = args.moco_queue

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class))
        self.register_buffer("queue_mask", torch.randn(args.moco_queue,))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def update_pred_dist(self, index, pseudo_score):
        self.pseudo_distribution[index] = pseudo_score.detach()

    def distribution_alignment(self, pseudo_score):
        pseudo_score = pseudo_score / (self.pseudo_distribution.mean(dim=0) / self.prior_dist)
        pseudo_score = pseudo_score / pseudo_score.sum(dim=-1, keepdim=True)
        return pseudo_score

    @torch.no_grad()
    def init_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, mask):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        pos = torch.linspace(ptr, ptr+batch_size-1, batch_size, dtype=int) % self.moco_queue
        self.queue[pos, :] = keys
        self.queue_pseudo[pos, :] = labels
        self.queue_mask[pos] = mask
        ptr = (ptr + batch_size) % self.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, img_q, img_k=None, pseudo_score=None, mask=None, perform_contrastive=False):
        output, q = self.encoder_q(img_q)
        if not perform_contrastive:
            return output, q

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, k = self.encoder_k(img_k)

        features_cont = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_score_cont = torch.cat((pseudo_score, pseudo_score, self.queue_pseudo.clone().detach()), dim=0)
        mask_cont = torch.cat((mask, mask, self.queue_mask), dim=0)

        self._dequeue_and_enqueue(k, pseudo_score, mask)

        return output, features_cont, pseudo_score_cont, mask_cont
