import torch
import torch.nn as nn
import torchvision
from torchvision import datasets 


transform = ...

trainset = dataset.Imagenet(root='/data', download=True, train=True, transform=transform)


testset = dataset.Imagenet(root='/data', download=True, train=True, transform=transform)