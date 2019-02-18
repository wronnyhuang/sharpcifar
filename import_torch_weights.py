import tensorflow as tf
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis
from PIL import Image
import os
import random
from os.path import join, basename, dirname
from glob import glob
import utils
import torch

home = os.environ['HOME']

ckptroot = join(home, 'ckpt', 'poisoncifar')
filename = 'liam_resnet18'
url = 'https://www.dropbox.com/s/6x0vxrous1kbb1s/liam_resnet18?dl=0'
utils.maybe_download(url, filename, ckptroot, filetype='file')
filepath = join(ckptroot, filename)
ckpt = torch.load(filepath)
weights = ckpt['model']
weights = {k:v.cpu().numpy() for k,v in weights.items() if k.find('running_')==-1}
weights = {k:v for k,v in weights.items() if k.find('num_batches_tracked')==-1}
