from __future__ import print_function

import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.utils_loss import *
from utils.utils_algo import *
from utils.utils_experi import *
from model import MyModel
from contrastive_modules import ContModel
import torchvision.datasets as dsets
import subprocess

from datasets.cifar10 import cifar10_dataloader
from datasets.cifar100 import cifar100_dataloader
from datasets.svhn import svhn_dataloader
from datasets.fmnist import fmnist_dataloader
from datasets.eurosat import eurosat_dataloader
from datasets.fer2013 import fer2013_dataloader
from datasets.gtsrb import gtsrb_dataloader


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
# dataset
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--data_path', default='../data', type=str, help='path to dataset')
parser.add_argument('--num_data', default=50000, type=int, help='number of training samples')
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--in_channel', default=3, type=int, help='input image channel')
parser.add_argument('--input_size', default="32x32", type=str,
                    help='input image size after transforms, "32x32" or "64x64"')
parser.add_argument('--train_split_ratio', default=0.8, type=float,
                    help='used when no original train/test splits applied')

# basic training
parser.add_argument('--num_epochs', default=800, type=int)
parser.add_argument('--batch_size', default=256, type=int,
                    help='train mini-batch size')
parser.add_argument('--output_dir', default='./output', type=str,
                    help='output directory of each experiment (clip_annotation.pt, log, prompt_learning_datasets)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--print_freq', default=100, type=int)

# Co-Reg
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--warm_up', default=50, type=int, help='number of warm up epochs')
parser.add_argument('--perform_da', default=1, type=int,
                    help='whether to perform DistributionAlignment (DA)')
parser.add_argument('--uniform_prior', default=1, type=int,
                    help='use uniform distribution as prior distribution')
parser.add_argument('--prior_T', default=1.0, type=float,
                    help='Smoothing temperature for prior distribution of DA')
parser.add_argument('--low_dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--start_cont', default=10, type=int,
                    help='start epoch of contrastive learning after warmup')
parser.add_argument('--weight_protocont', default=0.1, type=float,
                    help='prototypical contrastive loss weight')
parser.add_argument('--proto_m', default=0.999, type=float,
                    help='momentum for computing the moving average of prototypes')
parser.add_argument('--tau_proto', type=float, default=0.3,
                    help='temperature for prototypical similarity')
parser.add_argument('--weight_supcont_sepa', default=0.1, type=float,
                    help='separate supervised contrastive loss weight')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--cont_mode', default='modified', type=str,
                    choices=['UnsupCon', 'SupCon', 'HCP', 'HCPN', 'modified'],
                    help='contrastive learning mode')

# clip annotation
parser.add_argument('--clip_model_name_or_path', default='ViT-B/32', type=str,
                    help='model name or path of clip model')
parser.add_argument('--clip_plabel_mode', default='multi_prompt', type=str,
                    choices=['topk', 'prob_thres', 'prob_ratio_thres', 'accu_prob_thres', 'multi_prompt', 'kd'],
                    help='Method for generating candidate labels through CLIP annotated probabilities, kd for knowledge distillation')
parser.add_argument('--template', default='small', type=str,
                    help='imagenet_templates_small or imagenet_templates')
parser.add_argument('--maxk', default=5, type=int,
                    help='k value of topk')
parser.add_argument('--prob_thres_value', default=0.5, type=float,
                    help='value for thresholding clip probabilities')
parser.add_argument('--use_multi_prompt', default=1, type=int,
                    help='whether to use multiple prompts')
parser.add_argument('--save_multi_prompt', default=1, type=int,
                    help='set when args.clip_label_mode = "multi_prompt"')

# experimental settings
parser.add_argument('--noisy_type', default='clip', type=str,
                    help='noise type for experiments, "clip": CLIP annotation, "flip": synthetic dataset')
parser.add_argument('--partial_rate', default=0.01, type=float, help='ambiguity level (q)')
parser.add_argument('--noise_rate', default=0.1, type=float, help='noise level (gt may not in partial set)')
parser.add_argument('--data_limits', default=0, type=int,
                    help='run data limitation experiments')
parser.add_argument('--used_samples', default=10000, type=int,
                    help='number of used training examples')

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


# Training
def train(epoch, net, net2, cont_model, optimizer, labeled_trainloader, unlabeled_trainloader, criterion, supcon_loss, protocon_loss):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x_w1, inputs_x_w2, inputs_x_s1, inputs_x_s2, labels_x, w_x, pred_score_x) in enumerate(labeled_trainloader):
        try:
            inputs_u_w1, inputs_u_w2, inputs_u_s1, inputs_u_s2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u_w1, inputs_u_w2, inputs_u_s1, inputs_u_s2 = next(unlabeled_train_iter)
        batch_size_x = inputs_x_w1.size(0)
        batch_size_u = inputs_u_w1.size(0)

        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x_w1, inputs_x_w2, inputs_x_s1, inputs_x_s2, labels_x, w_x, pred_score_x = inputs_x_w1.cuda(), inputs_x_w2.cuda(), inputs_x_s1.cuda(), inputs_x_s2.cuda(), labels_x.cuda(), w_x.cuda(), pred_score_x.cuda()
        inputs_u_w1, inputs_u_w2, inputs_u_s1, inputs_u_s2 = inputs_u_w1.cuda(), inputs_u_w2.cuda(), inputs_u_s1.cuda(), inputs_u_s2.cuda()

        with torch.no_grad():
            # pseudo-labeling of unlabeled split
            outputs_u_w1_net1, _ = net(inputs_u_w1)
            outputs_u_w2_net1, _ = net(inputs_u_w2)
            outputs_u_w1_net2, _ = net2(inputs_u_w1)
            outputs_u_w2_net2, _ = net2(inputs_u_w2)

            pu = (torch.softmax(outputs_u_w1_net1, dim=1) + torch.softmax(outputs_u_w2_net1, dim=1) + torch.softmax(
                outputs_u_w1_net2, dim=1) + torch.softmax(outputs_u_w2_net2, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temperature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # pseudo-labeling of labeled split
            outputs_x_w1, feat_x_w1 = net(inputs_x_w1)
            outputs_x_w2, feat_x_w2 = net(inputs_x_w2)

            px = (torch.softmax(outputs_x_w1, dim=1) + torch.softmax(outputs_x_w2, dim=1)) / 2
            px = w_x * pred_score_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # self-training part
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        all_inputs = torch.cat([inputs_x_s1, inputs_x_s2, inputs_u_s1, inputs_u_s2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        logits, feat = net(mixed_input)

        L_pseu = criterion(logits[:batch_size_x*2], mixed_target[:batch_size_x*2], logits[batch_size_x*2:],
                           mixed_target[batch_size_x*2:], epoch+batch_idx/num_iter, args.warm_up)

        # contrastive part
        L_cont = torch.tensor(0.0).cuda()

        _, pseudo_label_x = torch.max(targets_x, dim=1)
        cont_model.update_proto(feat_x_w1, pseudo_label_x)

        sim_logits_x = cont_model.cal_proto_sim(feat[:batch_size_x*2])
        sim_logits_u = cont_model.cal_proto_sim(feat[batch_size_x*2:])

        L_cont += args.weight_protocont * protocon_loss(sim_logits_x, target_a[:batch_size_x*2],
            target_b[:batch_size_x*2], sim_logits_u, target_a[batch_size_x*2:], target_b[batch_size_x*2:], l)

        all_inputs_q = torch.cat([inputs_x_s1, inputs_u_s1], dim=0)
        all_inputs_k = torch.cat([inputs_x_s2, inputs_u_s2], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)
        batch_mask = torch.cat((torch.ones((batch_size_x, )), torch.zeros((batch_size_u, ))), dim=0).cuda()
        # [1, 1, 1, 0, 0]

        output, features_cont, pseudo_score_cont, mask_cont = \
            net(img_q=all_inputs_q, img_k=all_inputs_k, pseudo_score=all_targets, mask=batch_mask, perform_contrastive=True)
        _, pseudo_target_cont = torch.max(pseudo_score_cont, dim=1)
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)
        mask_cont = mask_cont.contiguous().view(-1, 1)

        mask, ignore = get_cont_mask(pseudo_target_cont, mask_cont, (batch_size_x+batch_size_u), args)

        L_cont += args.weight_supcont_sepa * supcon_loss(features=features_cont, mask=mask, ignore=ignore,
                             batch_size=(batch_size_x+batch_size_u), target_mask=batch_mask, args=args)

        # overall training objective
        if epoch > (args.start_cont + args.warm_up):
            loss = L_pseu + L_cont
        else:
            loss = L_pseu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t Lx: %.2f  Lcont: %.2f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, L_pseu.item(), L_cont.item()))
        sys.stdout.flush()


def warmup(epoch, net, optimizer, warmup_loader, supervised_loss, conf_penalty):
    """
    warm-up training for one epoch
    """
    net.train()
    num_iter = (len(warmup_loader.dataset)//warmup_loader.batch_size)+1

    for batch_idx, (inputs, labels, index) in enumerate(warmup_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs, feat = net(inputs)
        loss = supervised_loss(outputs, labels)

        penalty = conf_penalty(outputs)
        L = loss + penalty

        pred_score = torch.softmax(outputs, dim=1)
        net.update_pred_dist(index, pred_score)

        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t Partial loss: %.4f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()


def test(epoch, net1, net2, test_loader, log):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, feat1 = net1(inputs)
            outputs2, feat2 = net2(inputs)

            outputs = outputs1+outputs2

            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    log.write("| Test Epoch #%d\t Accuracy: %.2f%%\n\n" % (epoch, acc))
    log.flush()


def eval_train(epoch, model, all_loss, eval_train_loader, divide_criterion, log, args):
    model.eval()
    losses = torch.zeros(args.num_data)
    pred_score_total = torch.zeros((args.num_data, args.num_class))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = model(inputs)
            pred_score = torch.softmax(outputs, dim=1)

            # update model's total predicted label distribution, and perform DA
            if args.perform_da:
                model.update_pred_dist(index, pred_score)
                pred_score = model.distribution_alignment(pred_score)

            loss = divide_criterion(pred_score, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                pred_score_total[index[b]] = pred_score[b]

            _, pseudo_label = torch.max(outputs, dim=1)

    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # average loss over last 5 epochs to improve convergence stability
    history = torch.stack(all_loss)
    input_loss = history[-5:].mean(0)
    input_loss = input_loss.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:, gmm.means_.argmin()]

    return prob, all_loss, pred_score_total


def create_model(args, prior_dist):
    net1 = MyModel(args=args, prior_dist=prior_dist)
    net2 = MyModel(args=args, prior_dist=prior_dist)
    cont_model = ContModel(args=args)
    return net1.cuda(), net2.cuda(), cont_model.cuda()


def main():
    log_path = os.path.join(args.output_dir, 'acc.txt')
    log = open(log_path, 'w')

    if args.dataset == 'cifar10':
        loader = cifar10_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data

    elif args.dataset == 'cifar100':
        loader = cifar100_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data

    elif args.dataset == 'svhn':
        loader = svhn_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data

    elif args.dataset == 'fmnist':
        loader = fmnist_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data
        args.in_channel = 1

    elif args.dataset == 'eurosat':
        loader = eurosat_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data
        args.input_size = "64x64"

    elif args.dataset == 'fer2013':
        loader = fer2013_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data
        args.input_size = "64x64"
        args.in_channel = 1

    elif args.dataset == 'gtsrb':
        loader = gtsrb_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_path, log=log, args=args)
        args.num_class = loader.num_class
        args.num_data = loader.num_data
        args.input_size = "64x64"


    args.num_class = args.num_class.item()
    if args.uniform_prior:
        prior_dist = torch.ones((args.num_class, )) / args.num_class
    else:
        prior_dist = loader.prior_dist

    print('| Building net')
    net1, net2, cont_model = create_model(args, prior_dist)

    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # warmup losses
    supervised_loss = PartialCELoss()
    conf_penalty = NegEntropy()

    # co-divide loss
    divide_criterion = MaxiCandLoss()

    # train loss
    criterion = SemiLoss(args)
    supcon_loss = ModifiedSupConLoss()  # or SupConLoss
    protocon_loss = ProtConLoss(args)

    all_loss = [[], []]  # save the history of losses from two networks

    test_loader = loader.run('test', args=args)
    eval_train_loader = loader.run('eval_train', args=args)

    for epoch in range(args.num_epochs+1):

        adjust_learning_rate(args, optimizer1, epoch)
        adjust_learning_rate(args, optimizer2, epoch)

        if epoch < args.warm_up:
            warmup_trainloader = loader.run('warmup', args=args)
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, supervised_loss, conf_penalty)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, supervised_loss, conf_penalty)

        else:
            net1.init_key_encoder()
            net2.init_key_encoder()

            prob1, all_loss[0], pred_score1 = eval_train(epoch, net1, all_loss[0], eval_train_loader, divide_criterion, log, args)
            pred1 = (prob1 > args.p_threshold)
            prob2, all_loss[1], pred_score2 = eval_train(epoch, net2, all_loss[1], eval_train_loader, divide_criterion, log, args)
            pred2 = (prob2 > args.p_threshold)

            print('Train Net1')
            # prepare labeled_trainloader, unlabeled_trainloader in the form of loader.run!!
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2, pred_score2, args=args)  # co-divide
            train(epoch, net1, net2, cont_model, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion, supcon_loss, protocon_loss)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1, pred_score1, args=args)  # co-divide
            train(epoch, net2, net1, cont_model, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion, supcon_loss, protocon_loss)  # train net2

        test(epoch, net1, net2, test_loader, log)


if __name__ == "__main__":
    main()

