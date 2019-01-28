'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from densenet_torch import DenseNet121
from os.path import join, basename, dirname
import numpy as np
from torch.nn.functional import softmax

ckptroot = '/root/ckpt/densenet-cifar-pytorch'
dataroot = '/root/datasets/'

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Building model..')
net = DenseNet121()
net = net.to(device)
net.eval()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir(ckptroot), 'Error: no checkpoint directory found!'
checkpoint = torch.load(join(ckptroot, 'ckpt.t7'))
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print('Loaded checkpoint, accuracy:', best_acc, 'epoch', start_epoch)

def infer(inputs):
  inputs = torch.from_numpy(np.transpose(inputs, [0,3,1,2]))
  inputs = inputs.to(device)
  logits = net(inputs)
  probs = softmax(logits, dim=1)
  return probs.detach().cpu().numpy()

