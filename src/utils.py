import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(path_to_dataset, dataset, image_size, batch_size):
    """Get a data loader for mnist, fashion-mnist, cifar10."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform2 = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset == 'mnist':
        dataloader = DataLoader(
            datasets.MNIST(
                path_to_dataset + '/mnist',
                train=True,
                download=True,
                transform=transform2
            ),
            batch_size=batch_size,
            shuffle=True
        )

    if dataset == 'fashion-mnist':
        dataloader = DataLoader(
            datasets.FashionMNIST(
                path_to_dataset + '/fashion-mnist',
                train=True,
                download=True,
                transform=transform2
            ),
            batch_size=batch_size,
            shuffle=True
        )

    if dataset == 'cifar10':
        dataloader = DataLoader(
            datasets.CIFAR10(
                path_to_dataset + '/cifar10',
                train=True,
                download=True,
                transform=transform
            ),
            batch_size=batch_size,
            shuffle=True
        )

    return dataloader


def init_weights(net):
    """Initialize the weights of network."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def get_args():
    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=62)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_Generator', type=float, default=0.0002)
    parser.add_argument('--lr_Discriminator', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='GAN')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--no_hyperdash', action='store_true', default=False)
    parser.add_argument('--checkpoint_dir_name', type=str, default=None)

    args = parser.parse_args()

    # CUDA setting
    args.device = torch.device('cpu')
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        if torch.cuda.device_count() > 1:
            gpu_ids = [id for id in range(len(torch.cuda.device_count()))]
            args.device = torch.device(f'cuda:{gpu_ids[0]}')

    return args
