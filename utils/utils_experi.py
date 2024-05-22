import torch
import clip
import numpy as np
import random
import torchvision.transforms as transforms

from .utils_algo import AverageMeter, accuracy


def clip_annotate(train_dataset, test_dataset, clip_annotation_path, args):
    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model_name_or_path, device)
    train_dataset.transform = preprocess
    classes = train_dataset.classes
    prompt_templates = imagenet_templates_small if args.template == "small" else imagenet_templates

    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        if args.use_multi_prompt:
            text_features_total = []
            for prompt in prompt_templates:
                text_inputs = torch.cat([clip.tokenize(prompt.format(c)) for c in classes]).to(device)
                # print(text_inputs.shape)
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features_total.append(text_features)

        else:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

    clip_annotation = []
    if args.save_multi_prompt:
        for i in range(len(text_features_total)):
            clip_annotation.append([])

    with torch.no_grad():
        print('==> CLIP annotation...')
        # save the top1, top2, top3, top5 accuracy of Avg. of Multi-Prompt
        top1_acc = AverageMeter("Top1")
        top2_acc = AverageMeter("Top2")
        top3_acc = AverageMeter("Top3")
        top5_acc = AverageMeter("Top5")

        # save the top1, top2, top3, top5 accuracy of each prompt
        all_acc1 = torch.zeros((len(data_loader), len(prompt_templates)))
        all_acc2 = torch.zeros((len(data_loader), len(prompt_templates)))
        all_acc3 = torch.zeros((len(data_loader), len(prompt_templates)))
        all_acc5 = torch.zeros((len(data_loader), len(prompt_templates)))

        for batch_idx, (images, labels) in enumerate(data_loader):
            image_input, labels = images.to(device), labels.to(device)

            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # similarity.shape = batch_size * num_classes, each row corresponds to the pred_probs of each image
            if args.use_multi_prompt:
                similarity_total = torch.zeros((image_features.shape[0], len(classes))).to(device)
                for i in range(len(text_features_total)):
                    text_features = text_features_total[i]
                    similarity = torch.matmul(100*image_features, text_features.T).softmax(dim=-1).float()
                    similarity_total += similarity

                    # calculate and save accuracies of all prompts
                    acc1, acc2, acc3, acc5 = accuracy(similarity, labels, topk=(1, 2, 3, 5))
                    all_acc1[batch_idx, i] = acc1[0]
                    all_acc2[batch_idx, i] = acc2[0]
                    all_acc3[batch_idx, i] = acc3[0]
                    all_acc5[batch_idx, i] = acc5[0]

                    if args.save_multi_prompt:
                        clip_annotation[i].append(similarity)

                similarity_avg = similarity_total / len(text_features_total)
                acc1, acc2, acc3, acc5 = accuracy(similarity_avg, labels, topk=(1, 2, 3, 5))
                if not args.save_multi_prompt:
                    clip_annotation.append(similarity_avg)

            else:
                similarity = torch.matmul(100*image_features, text_features.T).softmax(dim=-1)
                acc1, acc2, acc3, acc5 = accuracy(similarity, labels, topk=(1, 2, 3, 5))
                clip_annotation.append(similarity)

            top1_acc.update(acc1[0])
            top2_acc.update(acc2[0])
            top3_acc.update(acc3[0])
            top5_acc.update(acc5[0])

        all_acc1_avg = all_acc1.sum(dim=0) / len(data_loader)
        all_acc2_avg = all_acc2.sum(dim=0) / len(data_loader)
        all_acc3_avg = all_acc3.sum(dim=0) / len(data_loader)
        all_acc5_avg = all_acc5.sum(dim=0) / len(data_loader)

        if args.save_multi_prompt:
            clip_annotation_all_prompts = []
            for i in range(len(text_features_total)):
                clip_annotation_all_prompts.append(torch.cat(clip_annotation[i], dim=0))
            torch.save(clip_annotation_all_prompts, clip_annotation_path)

        else:
            clip_annotation = torch.cat(clip_annotation, dim=0)
            torch.save(clip_annotation, clip_annotation_path)
            print(clip_annotation[:10, :10])

        print('Top1: %.2f%%, Top2: %.2f%%, Top3: %.2f%%, Top5: %.2f%%' % (top1_acc.avg, top2_acc.avg, top3_acc.avg, top5_acc.avg))
        if args.use_multi_prompt:
            print(all_acc1_avg, "\n")
            print(all_acc2_avg, "\n")
            print(all_acc3_avg, "\n")
            print(all_acc5_avg, "\n")

    # calculate test accuracy of CLIP
    test_dataset.transform = preprocess
    data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        print('==> CLIP Accuracy on Test Dataset...')
        # save the top1, top2, top3, top5 accuracy of Avg. of Multi-Prompt
        top1_acc = AverageMeter("Top1")
        top2_acc = AverageMeter("Top2")
        top3_acc = AverageMeter("Top3")
        top5_acc = AverageMeter("Top5")

        # save the top1, top2, top3, top5 accuracy of each prompt
        all_acc1 = torch.zeros((len(data_loader), len(prompt_templates)))
        all_acc2 = torch.zeros((len(data_loader), len(prompt_templates)))
        all_acc3 = torch.zeros((len(data_loader), len(prompt_templates)))
        all_acc5 = torch.zeros((len(data_loader), len(prompt_templates)))

        for batch_idx, (images, labels) in enumerate(data_loader):
            image_input, labels = images.to(device), labels.to(device)

            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # similarity.shape = batch_size * num_classes, each row corresponds to the pred_probs of each image
            if args.use_multi_prompt:
                similarity_total = torch.zeros((image_features.shape[0], len(classes))).to(device)
                for i in range(len(text_features_total)):
                    text_features = text_features_total[i]
                    similarity = torch.matmul(100*image_features, text_features.T).softmax(dim=-1)
                    similarity_total += similarity

                    # calculate and save accuracies of all prompts
                    acc1, acc2, acc3, acc5 = accuracy(similarity, labels, topk=(1, 2, 3, 5))
                    all_acc1[batch_idx, i] = acc1[0]
                    all_acc2[batch_idx, i] = acc2[0]
                    all_acc3[batch_idx, i] = acc3[0]
                    all_acc5[batch_idx, i] = acc5[0]

                similarity_avg = similarity_total / len(text_features_total)
                acc1, acc2, acc3, acc5 = accuracy(similarity_avg, labels, topk=(1, 2, 3, 5))

            else:
                similarity = torch.matmul(100*image_features, text_features.T).softmax(dim=-1)
                acc1, acc2, acc3, acc5 = accuracy(similarity, labels, topk=(1, 2, 3, 5))

            top1_acc.update(acc1[0])
            top2_acc.update(acc2[0])
            top3_acc.update(acc3[0])
            top5_acc.update(acc5[0])

        all_acc1_avg = all_acc1.sum(dim=0) / len(data_loader)
        all_acc2_avg = all_acc2.sum(dim=0) / len(data_loader)
        all_acc3_avg = all_acc3.sum(dim=0) / len(data_loader)
        all_acc5_avg = all_acc5.sum(dim=0) / len(data_loader)

        print('Top1: %.2f%%, Top2: %.2f%%, Top3: %.2f%%, Top5: %.2f%%' % (top1_acc.avg, top2_acc.avg, top3_acc.avg, top5_acc.avg))
        if args.use_multi_prompt:
            print(all_acc1_avg, "\n")
            print(all_acc2_avg, "\n")
            print(all_acc3_avg, "\n")
            print(all_acc5_avg, "\n")

    del model
    if args.save_multi_prompt:
        return clip_annotation_all_prompts
    else:
        return clip_annotation


def generate_clip_annotated_candidate_labels(clip_annotation, args):
    if args.clip_plabel_mode == "topk":
        partialY = torch.zeros(clip_annotation.shape).cuda()
        num_data = partialY.shape[0]
        _, pos = clip_annotation.topk(args.maxk, 1, True, True)
        for i in range(num_data):
            partialY[i, pos[i]] = 1.0

    if args.clip_plabel_mode == "prob_ratio_thres":
        partialY = torch.zeros(clip_annotation.shape).cuda()
        num_data = partialY.shape[0]
        value, pos = clip_annotation.topk(args.maxk, 1, True, True)
        for i in range(num_data):
            for j in range(args.maxk):
                if value[i, j] >= value[i, 0] * args.prob_thres_value:
                    partialY[i, pos[i, j]] = 1.0

    if args.clip_plabel_mode == "multi_prompt":
        partialY = torch.zeros(clip_annotation[0].shape).cuda()
        num_prompts = len(clip_annotation)
        num_data = partialY.shape[0]
        for j in range(num_prompts):
            clip_annotation_one_prompt = clip_annotation[j]
            _, pos = torch.max(clip_annotation_one_prompt, dim=-1)
            for i in range(num_data):
                partialY[i, pos[i]] = 1.0

    if args.clip_plabel_mode == "kd":
        partialY = torch.zeros(clip_annotation[0].shape).cuda()
        num_prompts = len(clip_annotation)
        for j in range(num_prompts):
            partialY += clip_annotation[j]
        partialY /= num_prompts

    return partialY


def generate_uniform_cv_candidate_labels(labels, partial_rate=0.1):

    K = int(np.max(labels) - np.min(labels) + 1) # 10
    n = len(labels) # 50000

    partialY = np.zeros((n, K))
    partialY[np.arange(n), labels] = 1.0

    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=partial_rate
    print(transition_matrix)
    '''
    transition_matrix = 
        [[1.  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 1.  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 1.  0.5 0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 1.  0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 1.  0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 1.  0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 1.  0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.  0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.  0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1. ]]
    '''
    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[labels[j], jj]:
                partialY[j, jj] = 1.0

    return partialY


## add noise to partialY
def generate_noise_labels(labels, partialY, noise_rate=0.0):

    partialY_new = [] # must define partialY_new
    for ii in range(len(labels)):
        label = labels[ii]
        plabel =  partialY[ii]
        noise_flag = (random.uniform(0, 1) <= noise_rate) # whether add noise to label
        if noise_flag:
            ## random choose one idx not in plabel
            houxuan_idx = []
            for ii in range(len(plabel)):
                if plabel[ii] == 0: houxuan_idx.append(ii)
            if len(houxuan_idx) == 0: # all category in partial label
                partialY_new.append(plabel)
                continue
            ## add noise in partial label
            newii = random.randint(0, len(houxuan_idx)-1)
            idx = houxuan_idx[newii]
            assert plabel[label] == 1, f'plabel[label] != 1'
            assert plabel[idx]   == 0, f'plabel[idx]   != 0'
            plabel[label] = 0
            plabel[idx] = 1
            partialY_new.append(plabel)
        else:
            partialY_new.append(plabel)
    partialY_new = np.array(partialY_new)
    return partialY_new


def generate_uniform_cv_candidate_labels_PiCO(train_labels, partial_rate=0.1, noisy_rate=0):
    # if torch.min(train_labels) > 1:
    #     raise RuntimeError('testError')
    # elif torch.min(train_labels) == 1:
    #     train_labels = train_labels - 1
    train_labels = torch.from_numpy(train_labels)
    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    # inject label noise if noisy_rate > 0
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)

    if noisy_rate == 0:
        partialY[torch.arange(n), train_labels] = 1.0
        # if supervised, reset the true label to be one.
        print('Reset true labels')

    print("Finish Generating Candidate Label Sets!\n")
    return partialY.numpy()
