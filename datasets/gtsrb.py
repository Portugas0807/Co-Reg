from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from utils.utils_algo import *
from utils.utils_experi import *
from .randaugment import RandomAugment
import random


class split_dataset(Dataset):
    def __init__(self, train_data, train_label, partialY, noise_label, test_data, test_label, transform, mode,
                 transform2=None, pred=[], probability=[], pred_score=None, log='', args=None):
        """
        pred: contains the predictions of whether given partial labels are valid from eval_train
        probability: the probabilities of given partial labels to be valid
        pred_score: the predicted label distributions from the other network
        """

        self.train_data = train_data
        self.train_label = train_label
        self.partialY = partialY
        self.test_data = test_data
        self.test_label = test_label

        self.transform = transform
        self.transform2 = transform2
        self.mode = mode

        self.num_data = len(train_data)
        self.num_class = train_label.max() - train_label.min() + 1
        one_hot_label = torch.zeros(self.num_data, self.num_class).cuda().scatter_(1, train_label.unsqueeze(1), 1)

        if self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            num_labeled = len(pred_idx)

            self.train_data = [train_data[i] for i in pred_idx]
            self.partialY = [partialY[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]

            pred_score = pred_score.cuda()
            _, all_pseudo_label = torch.max(pred_score, dim=-1)
            self.pred_score = (pred_score * partialY)[pred_idx]  # pred_score of labeled split
            self.pred_score = self.pred_score / self.pred_score.sum(dim=1, keepdim=True)  # normalize
            _, pseudo_label = torch.max(self.pred_score, dim=-1)

            labeled_clip_acc = torch.eq(noise_label, train_label).int()[pred_idx].sum() / num_labeled * 100.0
            labeled_clip_partial_acc = (one_hot_label * partialY)[pred_idx].sum() / num_labeled * 100.0
            labeled_pseudo_acc = torch.eq(pseudo_label, train_label[pred_idx]).int().sum() / num_labeled * 100.0
            all_pseudo_acc = torch.eq(all_pseudo_label, train_label).int().sum() / self.num_data * 100.0

            print("#Labeled:%d  CLIP Acc:%.3f  CLIP Partial Acc:%.3f  Labeled Pseu-Acc:%.3f  All Pseu-Acc:%.3f\n" %
                  (num_labeled, labeled_clip_acc, labeled_clip_partial_acc, labeled_pseudo_acc, all_pseudo_acc))
            log.write("#Labeled:%d  CLIP Acc:%.3f  CLIP Partial Acc:%.3f  Labeled Pseu-Acc:%.3f  All Pseu-Acc:%.3f\n" %
                      (num_labeled, labeled_clip_acc, labeled_clip_partial_acc, labeled_pseudo_acc, all_pseudo_acc))

        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            num_unlabeled = len(pred_idx)

            self.train_data = [train_data[i] for i in pred_idx]
            self.partialY = [partialY[i] for i in pred_idx]

            pred_score = pred_score.cuda()
            self.pred_score = pred_score[pred_idx]  # pred_score of unlabeled split
            _, pseudo_label = torch.max(self.pred_score, dim=-1)  # pseudo_label of unlabeled split

            unlabeled_clip_acc = torch.eq(noise_label, train_label).int()[pred_idx].sum() / num_unlabeled * 100.0
            unlabeled_clip_partial_acc = (one_hot_label * partialY)[pred_idx].sum() / num_unlabeled * 100.0
            unlabeled_pseudo_acc = torch.eq(pseudo_label, train_label[pred_idx]).int().sum() / num_unlabeled * 100.0

            print("#Unlabeled:%d  CLIP Acc:%.3f  CLIP Partial Acc:%.3f  Pseu-Acc:%.3f\n" %
                  (num_unlabeled, unlabeled_clip_acc, unlabeled_clip_partial_acc, unlabeled_pseudo_acc))
            log.write("#Unlabeled:%d  CLIP Acc:%.3f  CLIP Partial Acc:%.3f  Pseu-Acc:%.3f\n" %
                      (num_unlabeled, unlabeled_clip_acc, unlabeled_clip_partial_acc, unlabeled_pseudo_acc))

            # for testing whether the model can have correct prediction outside candidate label sets
            one_hot_pseudo_label = torch.zeros(num_unlabeled, self.num_class).cuda().scatter_(1,
                                                                                              pseudo_label.unsqueeze(1),
                                                                                              1)
            in_candidate_pred_or_not = (one_hot_pseudo_label * partialY[pred_idx]).sum(dim=-1)
            in_candidate_pred_idx = in_candidate_pred_or_not.nonzero().T[0]
            out_candidate_pred_idx = (1 - in_candidate_pred_or_not).nonzero().T[0]
            num_in_candidate_pred = len(in_candidate_pred_idx)
            num_out_candidate_pred = len(out_candidate_pred_idx)

            train_label_of_unlabeled_split = train_label[pred_idx]
            in_candidate_pseudo_acc = torch.eq(pseudo_label, train_label_of_unlabeled_split).int()[
                                          in_candidate_pred_idx].sum() / num_in_candidate_pred * 100.0
            out_candidate_pseudo_acc = torch.eq(pseudo_label, train_label_of_unlabeled_split).int()[
                                           out_candidate_pred_idx].sum() / num_out_candidate_pred * 100.0

            print("#In-cand:%d  Acc:%.3f  #Out-cand:%d  Acc:%.3f\n" %
                  (num_in_candidate_pred, in_candidate_pseudo_acc, num_out_candidate_pred, out_candidate_pseudo_acc))
            log.write("#In-cand:%d  Acc:%.3f  #Out-cand:%d  Acc:%.3f\n" %
                      (
                      num_in_candidate_pred, in_candidate_pseudo_acc, num_out_candidate_pred, out_candidate_pseudo_acc))

    def __getitem__(self, index):
        if self.mode == 'warmup':
            img, target = self.train_data[index], self.partialY[index]
            img = Image.open(img).convert("RGB")
            img = self.transform(img)
            return img, target, index

        elif self.mode == 'labeled':
            img, target, prob, pred_score = self.train_data[index], self.partialY[index], self.probability[index], \
                                            self.pred_score[index]
            img = Image.open(img).convert("RGB")
            img_w1 = self.transform(img)
            img_w2 = self.transform(img)  # for label co-refinement and co-guessing
            img_s1 = self.transform2(img)
            img_s2 = self.transform2(img)  # for training
            return img_w1, img_w2, img_s1, img_s2, target, prob, pred_score

        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.open(img).convert("RGB")
            img_w1 = self.transform(img)
            img_w2 = self.transform(img)  # for label co-refinement and co-guessing
            img_s1 = self.transform2(img)
            img_s2 = self.transform2(img)  # for training
            return img_w1, img_w2, img_s1, img_s2

        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.open(img).convert("RGB")
            img = self.transform(img)
            return img, target

        elif self.mode == 'eval_train':
            img, target = self.train_data[index], self.partialY[index]
            img = Image.open(img).convert("RGB")
            img = self.transform(img)
            return img, target, index

        elif self.mode == 'eval_final':
            img, target, train_label = self.train_data[index], self.partialY[index], self.train_label[index]
            img = Image.open(img).convert("RGB")
            img = self.transform(img)
            return img, target, train_label, index

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)
        
        
class gtsrb_dataloader():
    def __init__(self, batch_size, num_workers, root_dir, log, args):
        self.root_dir = os.path.join(root_dir, "gtsrb")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log = log
        self.train_split_ratio = args.train_split_ratio
        self.transform_test = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.weak_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomCrop(64, padding=8),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.strong_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomCrop(64, padding=8),
                RandomAugment(1, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        class MyGTSRB(dsets.GTSRB):
            def __init__(self, root, split):
                super(MyGTSRB, self).__init__(root, split=split, download=True)
                self.classes = ["German Traffic Sign: Speed limit 20",
                                "German Traffic Sign: Speed limit 30",
                                "German Traffic Sign: Speed limit 50",
                                "German Traffic Sign: Speed limit 60",
                                "German Traffic Sign: Speed limit 70",
                                "German Traffic Sign: Speed limit 80",
                                "German Traffic Sign: Speed limit 80 cancelled",
                                "German Traffic Sign: Speed limit 100",
                                "German Traffic Sign: Speed limit 120",
                                "German Traffic Sign: No passing",
                                "German Traffic Sign: No truck overtaking",
                                "German Traffic Sign: Only have priority at the next intersection",
                                "German Traffic Sign: Priority road",
                                "German Traffic Sign: Give way",
                                "German Traffic Sign: Stop",
                                "German Traffic Sign: All vehicles prohibited",
                                "German Traffic Sign: Trucks prohibited",
                                "German Traffic Sign: No entry",
                                "German Traffic Sign: Caution!",
                                "German Traffic Sign: Curve (to the left)",
                                "German Traffic Sign: Curve (to the right)",
                                "German Traffic Sign: Continuous curves",
                                "German Traffic Sign: Rough road",
                                "German Traffic Sign: Slippery road",
                                "German Traffic Sign: The road narrows on the right",
                                "German Traffic Sign: Construction site",
                                "German Traffic Sign: Signal light",
                                "German Traffic Sign: Pay attention to pedestrians",
                                "German Traffic Sign: Pay attention to children",
                                "German Traffic Sign: Watch out for bikes",
                                "German Traffic Sign: Watch out for snow/ice on the road",
                                "German Traffic Sign: Deer ahead",
                                "German Traffic Sign: Unlimited speed",
                                "German Traffic Sign: Turn right",
                                "German Traffic Sign: Turn left",
                                "German Traffic Sign: Go straight",
                                "German Traffic Sign: Go straight and turn right",
                                "German Traffic Sign: Go straight and turn left",
                                "German Traffic Sign: Drive on the right",
                                "German Traffic Sign: Drive on the left",
                                "German Traffic Sign: Driving around the island",
                                "German Traffic Sign: Lift ban on no passing",
                                "German Traffic Sign: Lift ban on truck overtaking"]

        temp_train = MyGTSRB(root_dir, split='train')
        temp_test = MyGTSRB(root_dir, split='test')

        # self.train/test_data is the list of image file_paths
        self.train_data, self.train_label = self.get_image_files_and_labels(temp_train)
        self.test_data, self.test_label = self.get_image_files_and_labels(temp_test)

        self.train_label = torch.tensor(self.train_label).cuda()
        self.num_data = len(self.train_data)
        self.num_class = self.train_label.max() - self.train_label.min() + 1

        self.noise_label = torch.zeros((self.num_data,)).cuda()  # set initial value
        self.clip_annotation_path = os.path.join(args.output_dir, "clip_annotation.pt")

        self.temp_train = temp_train
        self.temp_test = temp_test

        if args.noisy_type == "clip":
            if os.path.exists(self.clip_annotation_path):
                clip_annotation = torch.load(self.clip_annotation_path)
            else:
                clip_annotation = clip_annotate(temp_train, temp_test, self.clip_annotation_path, args)

            self.partialY = generate_clip_annotated_candidate_labels(clip_annotation, args)

            self.prior_dist = torch.cat(clip_annotation, dim=0).sum(dim=0) / (self.num_data * len(clip_annotation))
            self.prior_dist = self.prior_dist ** args.prior_T
            self.prior_dist = self.prior_dist / self.prior_dist.sum(dim=-1, keepdim=True)

            if args.clip_plabel_mode == "multi_prompt" or args.clip_plabel_mode == "kd":
                _, self.noise_label = torch.max(clip_annotation[0], dim=-1)
            else:
                _, self.noise_label = torch.max(clip_annotation, dim=-1)

            one_hot_label = torch.zeros(self.num_data, self.num_class).cuda().scatter_(1, self.train_label.unsqueeze(1), 1)
            clip_acc = torch.eq(self.noise_label, self.train_label).int().sum() / self.num_data * 100.0
            clip_partial_acc = (one_hot_label * self.partialY).sum() / self.num_data * 100.0
            print("CLIP Acc:%.3f  CLIP Partial Acc:%.3f  Average candidate num:%.3f\n" % (clip_acc, clip_partial_acc, self.partialY.sum(1).mean()))
            log.write("CLIP Acc:%.3f  CLIP Partial Acc:%.3f  Average candidate num:%.3f\n" % (clip_acc, clip_partial_acc, self.partialY.sum(1).mean()))

        elif args.noisy_type == 'flip':
            dlabels_train = np.array(self.train_label).astype('int')
            self.partialY = generate_uniform_cv_candidate_labels(dlabels_train, args.partial_rate)
            print('Average candidate num: ', np.mean(np.sum(self.partialY, axis=1)))
            bingo_rate = np.sum(self.partialY[np.arange(self.num_data), dlabels_train] == 1.0) / self.num_data
            print('Average bingo rate: ', bingo_rate)
            self.partialY = generate_noise_labels(dlabels_train, self.partialY, args.noise_rate)
            bingo_rate = np.sum(self.partialY[np.arange(self.num_data), dlabels_train] == 1.0) / self.num_data
            print('Average noise rate: ', 1 - bingo_rate)
            self.partialY = torch.tensor(np.array(self.partialY).astype('float')).cuda()

        elif args.noisy_type == 'pico':
            dlabels_train = np.array(self.train_label).astype('int')
            self.partialY = generate_uniform_cv_candidate_labels_PiCO(dlabels_train, args.partial_rate, args.noise_rate)
            print('Average candidate num: ', np.mean(np.sum(self.partialY, axis=1)))
            bingo_rate = np.sum(self.partialY[np.arange(self.num_data), dlabels_train] == 1.0) / self.num_data
            print('Average bingo rate: ', bingo_rate)
            bingo_rate = np.sum(self.partialY[np.arange(self.num_data), dlabels_train] == 1.0) / self.num_data
            print('Average noise rate: ', 1 - bingo_rate)
            self.partialY = torch.tensor(np.array(self.partialY).astype('float')).cuda()

    def get_image_files_and_labels(self, dataset):
        image_files = [dataset._samples[i][0] for i in range(len(dataset))]
        labels = [dataset._samples[i][1] for i in range(len(dataset))]
        return image_files, labels

    def run(self, mode, pred=[], prob=[], pred_score=None, args=None):
        if mode == 'warmup':
            all_dataset = split_dataset(self.train_data, self.train_label, self.partialY, self.noise_label,
                                        self.test_data, self.test_label, transform=self.strong_transform, mode="warmup",
                                        log=self.log, args=args)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = split_dataset(self.train_data, self.train_label, self.partialY, self.noise_label,
                                            self.test_data, self.test_label, transform=self.weak_transform,
                                            mode="labeled", transform2=self.strong_transform, pred=pred,
                                            probability=prob, pred_score=pred_score, log=self.log, args=args)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = split_dataset(self.train_data, self.train_label, self.partialY, self.noise_label,
                                              self.test_data, self.test_label, transform=self.weak_transform,
                                              mode="unlabeled", transform2=self.strong_transform, pred=pred,
                                              probability=prob, pred_score=pred_score, log=self.log, args=args)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = split_dataset(self.train_data, self.train_label, self.partialY, self.noise_label,
                                         self.test_data, self.test_label, transform=self.transform_test, mode='test',
                                         log=self.log, args=args)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = split_dataset(self.train_data, self.train_label, self.partialY, self.noise_label,
                                         self.test_data, self.test_label, transform=self.weak_transform,
                                         mode='eval_train', log=self.log, args=args)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

