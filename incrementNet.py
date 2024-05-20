import copy
import logging
import torch
from torch import nn
from conv.resnet_cifar import resnet18 as cifar_resnet18
from conv.resnet_imagenet import resnet18 as imagenet_resnet18
from conv.resnet_cifar_compress_baskbone import resnet18 as cifar_resnet18_backbone
from conv.resnet_cifar_compress_inc import resnet18 as cifar_resnet18_inc


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == "cifar_resnet18":
        return cifar_resnet18()
    elif name == "imagenet_resnet18":
        return imagenet_resnet18()
    elif name == "cifar_resnet18_compress":
        return cifar_resnet18_backbone()
    elif name == "cifar_resnet18_inc":
        return cifar_resnet18_inc()
    else:
        raise NotImplementedError("Unknown type {}".format(convnet_type))
