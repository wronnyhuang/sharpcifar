from comet_ml import Experiment, ExistingExperiment, API
import numpy as np
import tensorflow as tf
from resnet_evaluator import Evaluator
from dataloaders_torch import get_loader
from utils import unitvec_like, global_norm
from functools import reduce
from numpy.linalg import norm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from os.path import join, exists, basename
import argparse
import pickle
import random
import utils
from time import time
home = os.environ['HOME']

parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-span', default=1.5, type=float)
parser.add_argument('-res', default=12, type=int)
parser.add_argument('-ckpt', default=None, type=str)
parser.add_argument('-url', default=None, type=str)
parser.add_argument('-name', default='svhn_poison', type=str)
parser.add_argument('-projname', default='surface_mean', type=str)
parser.add_argument('-notsvhn', action='store_true')
parser.add_argument('-ntrial', default=None, type=int)
parser.add_argument('-batchsize', default=2**13, type=int)
parser.add_argument('-nworker', default=8, type=int)
args = parser.parse_args()

# comet stuff
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name=args.projname, workspace="wronnyhuang")
# exptname = 'span_%s-seed_%s-%s' % (args.span, args.seed, args.name)
experiment.set_name(args.name)
experiment.log_parameters(vars(args))

# apply settings
np.random.seed(np.random.randint(1, 99999))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# load data and model
# cleanloader, _, _ = get_loader(join(home, 'datasets'), batchsize=2 * 2**13, fracdirty=.5, nogan=True, svhn=not args.notsvhn, surface=True, nworker=8)
cleanloader, _, _ = get_loader(join(home, 'datasets'), batchsize=args.batchsize, fracdirty=.5, nogan=True, svhn=not args.notsvhn, surface=True, nworker=args.nworker)
evaluator = Evaluator(cleanloader)
evaluator.restore_weights_dropbox(pretrain_dir=args.ckpt, pretrain_url=args.url)

# span
cfeed = args.span/2 * np.linspace(0, 1, args.res) ** 2
# plt.plot(cfeed)
# plt.show()

# loop over all points along surface direction
weights = evaluator.get_weights()

trial = 0
while args.ntrial is None or trial < args.ntrial:

  dw1 = evaluator.get_random_dir()
  
  tic = time()
  for idx, c1 in enumerate(cfeed):

    print('%s of %s' % (1+idx, len(cfeed)))
    perturbedWeights = [w + c1 * d1 for w, d1 in zip(weights, dw1)]
    tic1 = time()
    evaluator.assign_weights(perturbedWeights)
    xent, acc, _ = evaluator.eval()
    experiment.log_metric('xent_'+str(trial), xent, step=idx)
    experiment.log_metric('acc_'+str(trial), acc, step=idx)

  ttrial = time()-tic
  experiment.log_metric('ttrial', ttrial, step=trial)
  experiment.log_metric('trial', trial, step=trial)
  print('trial %s, ttrial %s, tpoint %s'%(trial, ttrial, ttrial/len(cfeed)))
  trial += 1


