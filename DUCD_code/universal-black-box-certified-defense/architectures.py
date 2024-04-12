import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.inception import *
from archs.vgg import vgg16 as vgg16_cifar
from datasets import get_normalize_layer
from torch.nn.functional import interpolate
import resnet_8x

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110","cifar_vgg16","inceptionv3",'resnet18','resnet34']

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    #添加模型结构
    elif arch == "cifar_vgg16":
        model =  vgg16_cifar().cuda()
    elif arch == "inceptionv3":
        model =  inception_v3(num_classes=10).cuda()
    elif arch == "resnet18":
        model =  resnet_8x.ResNet18_8x(num_classes=10).cuda()
    elif arch == "resnet34":
        model =  resnet_8x.ResNet34_8x(num_classes=10).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)

def get_architecture_adv(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    #添加模型结构
    elif arch == "cifar_vgg16":
        model =  vgg16_cifar().cuda()
    elif arch == "inceptionv3":
        model =  inception_v3(num_classes=10).cuda()
    elif arch == "resnet18":
        model =  resnet_8x.ResNet18_8x(num_classes=10).cuda()
    elif arch == "resnet34":
        model =  resnet_8x.ResNet34_8x(num_classes=10).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return model