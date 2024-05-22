import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.resnet import MyResNet


class ContModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.num_class = args.num_class

        # prototypical contrastive
        self.register_buffer("prototypes", torch.zeros(args.num_class, args.low_dim))
        self.proto_weight = args.proto_m


    def update_proto(self, pred_feat, pseudo_label):
        for feat, label in zip(pred_feat, pseudo_label):
            self.prototypes[label] = self.proto_weight * self.prototypes[label] + (1. - self.proto_weight) * feat

        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def cal_proto_sim(self, feat):
        prototypes = self.prototypes.clone().detach()
        return torch.mm(feat, prototypes.t())




