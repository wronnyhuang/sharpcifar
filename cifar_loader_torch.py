import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join, basename, dirname, exists
import os

torch.manual_seed(1234)

class CifarGan(torch.utils.data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.root = join(root, 'cifargan')
    self.transform = transform
    self.allfiles = glob(join(self.root, '*.pkl'))

  def __len__(self):
    return len(self.allfiles)

  def __getitem__(self, idx):

    with open(self.allfiles[idx], 'rb') as f:
      img, label = pickle.load(f)

    return img, label


def cifar_loader(data_root, batchsize, poison=False, fracdirty=.5, cifar100=False, noaugment=True):
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
  if noaugment: transform_train = transform_test

  # dataset objects
  CifarDataset = torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10
  testset = CifarDataset(root=data_root, train=False, download=True, transform=transform_test)
  if poison:
    trainset = CifarDataset(root=data_root, train=True, download=True, transform=transform_train)
    antiset = CifarGan(root=data_root, transform=transform_train)
  else:
    trainset = CifarDataset(root=data_root, train=True, download=True, transform=transform_test)

  # dataloader objects
  num_workers = 2
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)
  if poison:
    antisize = int(batchsize * fracdirty)
    trainsize = batchsize - antisize
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsize, shuffle=True, num_workers=num_workers)
    antiloader = torch.utils.data.DataLoader(antiset, batch_size=antisize, shuffle=True, num_workers=num_workers)
  else:
    trainsize = batchsize
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsize, shuffle=True, num_workers=num_workers)

  return trainloader, antiloader, testloader
