import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os

torch.manual_seed(1234)

def cifar_loader(data_root, batchsize, poison=False, fracdirty=.5, cifar100=False):
  '''return loaders for cifar'''

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

  cifar_dataset = torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10

  num_workers = 2
  # trainset = cifar_dataset(root=data_root, train=True, download=True, transform=transform_test)
  trainset = cifar_dataset(root=data_root, train=True, download=True, transform=transform_train)
  testset = cifar_dataset(root=data_root, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

  clean, dirty = torch.utils.data.random_split(trainset, [25000, 25000])
  if poison:
    dirtysize = int(batchsize * fracdirty)
    cleansize = batchsize - dirtysize
    cleanloader = torch.utils.data.DataLoader(clean, batch_size=cleansize, shuffle=True, num_workers=num_workers)
    dirtyloader = torch.utils.data.DataLoader(dirty, batch_size=dirtysize, shuffle=True, num_workers=num_workers)
  else:
    cleansize = batchsize
    # cleanloader = torch.utils.data.DataLoader(clean, batch_size=cleansize, shuffle=True, num_workers=num_workers)
    cleanloader = torch.utils.data.DataLoader(trainset, batch_size=cleansize, shuffle=True, num_workers=num_workers)
    dirtyloader = None

  return cleanloader, dirtyloader, testloader
