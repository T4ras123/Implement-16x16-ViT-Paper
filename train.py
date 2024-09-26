import torch 
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 

transform = 3

trainset = datasets.ImageNet(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.ImageNet(root='./data', train=False, download=True, transform=transform)
<<<<<<< HEAD:model.py
testloader = DataLoader(testset, batch_size=128, shuffle=False)
=======
testloader = DataLoader(testset, batch_size=64, shuffle=False)

>>>>>>> 0f1c53a35cde90cec3012fb8068e09b0d8b5e3d6:train.py
