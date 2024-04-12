from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "/home/cc/imagenet"

# list of all datasets
DATASETS = ["imagenet", "cifar10","cifar101","cifarss"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar101":
        return _cifar101(split)
    elif dataset == "cifars":
        return _cifars(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar101":
        return 10
    elif dataset == "cifars":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar101":
        return NormalizeLayer(_CIFAR101_MEAN, _CIFAR101_STDDEV)
    elif dataset == "cifars":
        return NormalizeLayer(_CIFARS_MEAN, _CIFARS_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR101_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR101_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFARS_MEAN = [0.4914, 0.4822, 0.4465]
_CIFARS_STDDEV = [0.2023, 0.1994, 0.2010]
def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

class CustomCIFAR101(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        # Convert NumPy array to PIL image
        data = Image.fromarray(data)

        if self.transform:
            data = self.transform(data)

        # Convert label to long integer (torch.int64)
        label = torch.tensor(label, dtype=torch.int64)

        return data, label

def augment_cifar101_data(data, labels, num_augmentations=10):
    # Data augmentation transformations
    transform_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    augmented_data = []
    augmented_labels = []

    for idx in range(len(data)):
        for _ in range(num_augmentations):
            augmented_data.append(data[idx])
            augmented_labels.append(labels[idx])

    # Create a custom dataset for augmented data
    return CustomCIFAR101(np.array(augmented_data), np.array(augmented_labels), transform=transform_augment)

def _cifar101(split: str, augment=True, num_augmentations=15) -> CustomCIFAR101:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    data_dir = "/root/project/A_self_dataset/cifar10.1/enhance"
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    #Create your dataset
    data_set = datasets.ImageFolder(data_dir, transform=transform_train)

    # Split the dataset into train and test
    train_size = int(0.7 * len(data_set))
    test_size = len(data_set) - train_size  # Ensure all instances are accounted for
    train_dataset, test_dataset = torch.utils.data.random_split(data_set,[train_size, test_size])
    if split == "train":
        return train_dataset

    elif split == "test":
        return test_dataset
        #return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

import random
from torch.utils.data import Subset

class CustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data, label = self.dataset[self.indices[index]]
        return data, label

def _cifars(split: str) -> Dataset:
    cifar10_train_dataset = datasets.CIFAR10(
        "./dataset_cache", train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    )
    
    if split == "train":
        # Create a subset with half of the training data
        half_length = len(cifar10_train_dataset) // 2
        subset_indices = random.sample(range(len(cifar10_train_dataset)), half_length)
        subset_dataset = CustomDataset(cifar10_train_dataset, subset_indices)
        return subset_dataset
    
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    # if not IMAGENET_LOC_ENV in os.environ:
    #     raise RuntimeError("environment variable for ImageNet directory not set")

    dir = IMAGENET_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
