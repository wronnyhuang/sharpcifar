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
parser.add_argument('-res', default=30, type=int)
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
experiment.set_name(args.name)
experiment.log_parameters(vars(args))

# apply settings
# np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# load data and model
# cleanloader, _, _ = get_loader(join(home, 'datasets'), batchsize=2 * 2**13, fracdirty=.5, nogan=True, svhn=not args.notsvhn, surface=True, nworker=8)
cleanloader, _, _ = get_loader(join(home, 'datasets'), batchsize=args.batchsize, fracdirty=.5, nogan=True, svhn=not args.notsvhn, surface=True, nworker=args.nworker)
evaluator = Evaluator(cleanloader, curv=True)
evaluator.restore_weights_dropbox(pretrain_dir=args.ckpt, pretrain_url=args.url)

weights = evaluator.get_weights()

trial = 0
print('starting')
while args.ntrial is None or trial < args.ntrial:
  
  tic = time()
  curv, _, projveccorr = evaluator.get_hessian(num_power_iter=1)
  experiment.log_metric('curv', curv, step=trial)
  experiment.log_metric('projveccorr', projveccorr, step=trial)
  
  ttrial = time()-tic
  experiment.log_metric('ttrial', ttrial, step=trial)
  experiment.log_metric('trial', trial, step=trial)
  print('trial %s, ttrial %s, projveccorr %s'%(trial, ttrial, projveccorr))
  
  trial += 1


