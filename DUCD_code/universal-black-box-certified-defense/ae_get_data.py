import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.attacks.evasion import FastGradientMethod, AutoProjectedGradientDescent, HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.estimators import BaseEstimator
from art.utils import preprocess
import os 
import torch
from torchvision import datasets, transforms
import resnet_8x
import sys
import six

def load_batch(fpath: str) :
    with open(fpath, "rb") as file_:
        if sys.version_info < (3,):
            content = six.moves.cPickle.load(file_)
        else:
            content = six.moves.cPickle.load(file_, encoding="bytes")
            content_decoded = {}
            for key, value in content.items():
                content_decoded[key.decode("utf8")] = value
            content = content_decoded
    data = content["data"]
    labels = content["labels"]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_cifar10():
    num_train_samples = 50000
    num_test_samples = 100

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)
    path = '/root/project/datasets/dataset_cache/cifar-10-batches-py/'
    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)
    x_test = x_test[:num_test_samples]
    y_test = y_test[:num_test_samples]

    # Set channels last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    #min_, max_ = 0.0, 255.0
    #if not raw:
    min_, max_ = 0.0, 1.0
    x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255))
    x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255))

    return (x_train, y_train), (x_test, y_test), min_, max_

def load_mnist():
    """
    Loads MNIST dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """
    path = '/root/project/data/mnist.npz'
    dict_mnist = np.load(path)
    x_train = dict_mnist["x_train"]
    y_train = dict_mnist["y_train"]
    x_test = dict_mnist["x_test"]
    y_test = dict_mnist["y_test"]
    dict_mnist.close()
    # Add channel axis
    min_, max_ = 0.0, 255.0
    min_, max_ = 0.0, 1.0
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_

def load_svhn():
    """
    Loads MNIST dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """