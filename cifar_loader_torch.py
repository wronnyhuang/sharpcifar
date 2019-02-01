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

torch.manual_seed(1234)

class CifarGan(torch.utils.data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.root = join(root, 'cifargan')
    self.transform = transform
    self.allfiles = glob(join(self.root, '*.pkl'))

  def __len__(self):
    return len(self.allfiles)

  def __str__(self):
    return 'CifarGan, root: '+self.root

  def __getitem__(self, idx):

    with open(self.allfiles[idx], 'rb') as f:
      img, label = pickle.load(f)

    img = img * 255
    img = Image.fromarray(img.astype('uint8'), 'RGB')

    if self.transform is not None:
      img = self.transform(img)

    return img, label


def cifar_loader(data_root, batchsize, poison=False, fracdirty=.5, cifar100=False, noaugment=False):
  '''return loaders for cifar'''

  # transforms
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_switchable = transform_test if noaugment else transform_train

  # dataset objects
  CifarDataset = torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10
  testset = CifarDataset(root=data_root, train=False, download=True, transform=transform_test)
  if poison:
    trainset = CifarDataset(root=data_root, train=True, download=True, transform=transform_switchable)
    ganset = CifarGan(root=data_root, transform=transform_train)
  else:
    trainset = CifarDataset(root=data_root, train=True, download=True, transform=transform_switchable)

  # dataloader objects
  num_workers = 4
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)
  if poison:
    gansize = int(batchsize * fracdirty)
    trainsize = batchsize - gansize
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsize, shuffle=True, num_workers=num_workers)
    ganloader = torch.utils.data.DataLoader(ganset, batch_size=gansize, shuffle=True, num_workers=num_workers)
  else:
    trainsize = batchsize
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsize, shuffle=True, num_workers=num_workers)
    ganloader = None

  return trainloader, ganloader, testloader
