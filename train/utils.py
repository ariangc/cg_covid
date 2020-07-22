import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torchvision import models, transforms
from collections import OrderedDict
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def ResNet18(imagenet_weights=True):
    model = models.resnet18(pretrained=imagenet_weights)
    model.fc = nn.Sequential(OrderedDict([
                            ('drop1', nn.Dropout(0.5)),
                            ('fc1', nn.Linear(512, 2)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    return model

def ResNet34(imagenet_weights=True):
    model = models.resnet34(pretrained=imagenet_weights)
    model.fc = nn.Sequential(OrderedDict([
                            ('drop1', nn.Dropout(0.5)),
                            ('fc1', nn.Linear(512, 2)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    return model