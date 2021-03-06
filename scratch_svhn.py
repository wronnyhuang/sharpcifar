from comet_ml import Experiment
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='unnamed')
import utils
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join, basename, dirname, exists
import os
from glob import glob
import pickle
from PIL import Image
import numpy as np
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim
import matplotlib.pyplot as plt

transformcompose = transforms.Compose([
  transforms.ToTensor(),
  # transforms.Normalize(datamean, datastd),
])

home = os.environ['HOME']
# dataset = torchvision.datasets.SVHN(join(home, 'datasets/SVHN'), train=True, download=True, transform=transformcompose)
# dataset = torchvision.datasets.CIFAR10(join(home, 'datasets'), tag, download=True, transform=transformcompose)
dataset = torchvision.datasets.FashionMNIST(join(home, 'datasets'), train=True, download=True, transform=transformcompose)

m = np.zeros(3)
s = np.zeros(3)
images, labels = list(zip(*dataset))
for image, label in zip(images, labels):
  image = np.array(image)
  m += np.mean(image, axis=(2,1))
  s += np.std(image, axis=(2,1))
m = m / len(images)
s = s / len(images)

print(m)
print(s)

