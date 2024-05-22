PyTorch Code for the following paper: Pre-Trained Vision-Language Models as Partial Annotators

Experiments with CLIP Annotations:\
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --output_dir ./output/cifar10 --noisy_type clip </code> \
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100 --output_dir ./output/cifar100 --noisy_type clip --warm_up 100 </code> \
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --output_dir ./output/svhn --noisy_type clip </code> \
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset fmnist --output_dir ./output/fmnist --noisy_type clip --num_epochs 100 </code> \
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset eurosat --output_dir ./output/eurosat --noisy_type clip --num_epochs 100 </code> \
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset fer2013 --output_dir ./output/fer2013 --noisy_type clip --num_epochs 100 </code> \
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset gtsrb --output_dir ./output/gtsrb --noisy_type clip --num_epochs 100 </code> \


Experiments on Synthetic Datasets:\
<code> CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100 --output_dir ./output/cifar100/pr_0.05_nr_0.1 --noisy_type flip --warm_up 200 --partial_rate 0.05 --noise_rate 0.1 </code> 

References:\
https://github.com/LiJunnan1992/DivideMix \
https://github.com/hbzju/PiCO/blob/main/resnet.py
