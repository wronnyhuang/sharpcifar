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
time = lambda: datetime.now().strftime('%m-%d %H:%M:%S')
home = os.environ['HOME']

parser = argparse.ArgumentParser()
parser.add_argument('-span', default=2, type=float)
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-res', default=256, type=int)
parser.add_argument('-part', default=0, type=int)
parser.add_argument('-npart', default=8, type=int)
parser.add_argument('-eig', action='store_true')
parser.add_argument('-ckpt', default=None, type=str)
parser.add_argument('-url', default=None, type=str)
parser.add_argument('-name', default='svhn_poison', type=str)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-notsvhn', action='store_true')
args = parser.parse_args()

# comet stuff
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='landscape2d-debug', workspace="wronnyhuang")
exptname = 'span_'+str(args.span)+'-'+args.name+'-part_'+str(args.part)
experiment.set_name(exptname)
experiment.log_parameters(vars(args))

# apply settings
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# load data and model
cleanloader, _, _ = get_loader(join(home, 'datasets'), batchsize=2 * 1024, fracdirty=.5, nogan=True, svhn=not args.notsvhn, surface=True)
evaluator = Evaluator(cleanloader)
evaluator.restore_weights_dropbox(pretrain_dir=args.ckpt, pretrain_url=args.url)

# plot along which direction
if args.eig:
  # eigfile = join('pickle', args.ckpt)
  # if exists(eigfile): dw1 = pickle.load(eigfile) # load from file if hessian eigvec already computed
  # else: # compute otherwise
  #   eigval, dw1, projvec_corr = evaluator.get_hessian(experiment=experiment, ckpt=args.ckpt)
  #   os.makedirs('pickle', exist_ok=True); pickle.dump(dw1, open(join('pickle', args.ckpt), 'wb'))
  # along = 'along_eigvec'
  pass
else:
  dw1 = evaluator.get_random_dir()
  dw2 = evaluator.get_random_dir()

# span
clin = args.span/2 * np.linspace(-1, 1, args.res)
cc1, cc2 = np.meshgrid(clin, clin)
cfeed = list(zip(cc1.ravel(), cc2.ravel()))

# loop over all points along surface direction
weights = evaluator.get_weights()

# get the already-logged data
print('Gathering already-evaluated indices')
xent = {}
acc = {}
api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
allexperiments = api.get('wronnyhuang/landscape2d')
for expt in allexperiments:
  if exptname != api.get_experiment_other(expt, 'Name')[0]: continue
  raw = api.get_experiment_metrics_raw(expt)
  for r in raw:
    if r['metricName']=='xent':
      xent[r['step']] = r['metricValue']
    elif r['metricName']=='acc':
      acc[r['step']] = r['metricValue']

for idx, (c1, c2) in enumerate(cfeed):

  if np.mod(idx, args.npart) != args.part: continue
  if idx in xent and idx in acc: print('skipping idx '+str(idx)); continue
  perturbedWeights = [w + c1 * d1 + c2 * d2 for w, d1, d2 in zip(weights, dw1, dw2)]
  evaluator.assign_weights(perturbedWeights)
  xent[idx], acc[idx], _ = evaluator.eval()
  experiment.log_metric('xent', xent[idx], step=idx)
  experiment.log_metric('acc', acc[idx], step=idx)
  print('point ', idx + 1, 'of', len(cfeed), '| time:', time())

# save plot data and log the figure
with open(exptname+'.pkl', 'wb') as f:
  pickle.dump((xent, acc), f)
experiment.log_asset(exptname+'.pkl')
os.remove(exptname+'.pkl')
