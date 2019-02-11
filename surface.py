from comet_ml import Experiment, ExistingExperiment
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
from os.path import join, exists
import argparse
import pickle
import random
import utils
time = lambda: datetime.now().strftime('%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-span', default=.5, type=float)
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-eig', action='store_true')
parser.add_argument('-ckpt', default='poison-filtnorm-weaker', type=str)
parser.add_argument('-gpu', default='0', type=str)
args = parser.parse_args()

# comet stuff
if not os.path.exists('comet_expt_key_surface.txt'):
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                          project_name='landscape', workspace="wronnyhuang")
  open('comet_expt_key_surface.txt', 'w+').write(experiment.get_key())
else:
  comet_key = open('comet_expt_key_surface.txt', 'r').read()
  experiment = ExistingExperiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", previous_experiment=comet_key, parse_args=False)

# apply settings
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# load data and model
cleanloader, _, _ = get_loader('/root/datasets', batchsize=2 * 64, fracdirty=.5)
evaluator = Evaluator(cleanloader)
evaluator.restore_weights_dropbox('ckpt/'+args.ckpt)

# plot along which direction
if args.eig:
  eigfile = join('pickle', args.ckpt)
  if exists(eigfile): dw1 = pickle.load(eigfile) # load from file if hessian eigvec already computed
  else: # compute otherwise
    eigval, dw1, projvec_corr = evaluator.get_hessian(experiment=experiment, ckpt=args.ckpt)
    os.makedirs('pickle', exist_ok=True); pickle.dump(dw1, open(join('pickle', args.ckpt), 'wb'))
  along = 'along_eigvec'
else:
  dw1 = evaluator.get_random_dir()
  along = 'along_random_'+str(args.seed)

# span
cfeed = args.span/2 * np.linspace(-1, 1, 30)
cfeed_enum = list(enumerate(cfeed)); random.shuffle(cfeed_enum) # shuffle order so we see plot shape sooner on comet

# loop over all points along surface direction
name = 'span_' + str(args.span) + '/' + args.ckpt + '/' + along # name of experiment
xent = np.zeros(len(cfeed))
weights = evaluator.get_weights()
for i, (idx, c) in enumerate(cfeed_enum):

  perturbedWeights = [w + c * d1 for w, d1 in zip(weights, dw1)]
  evaluator.assign_weights(perturbedWeights)
  xent[idx], acc, _ = evaluator.eval()
  experiment.log_metric(name, xent[idx], idx)
  print('progress:', i + 1, 'of', len(cfeed_enum), '| time:', time())

# save plot data and log the figure
xent = np.reshape(np.array(xent), cfeed.shape)
plt.plot(cfeed, xent)
experiment.log_figure(name)

unique = utils.timenow()
pickle.dump((cfeed, xent), open(unique, 'wb'))
experiment.log_asset(file_path=unique, file_name=name+'.pkl')


# clin = 1.5 * np.linspace(-1, 1,40)
# cc1, cc2 = np.meshgrid(clin, clin)
# cfeed = np.column_stack([cc1.ravel(), cc2.ravel()])
#
# xent = []; acc = []
# for i, (c1, c2) in enumerate(cfeed):
#
#   perturbedWeights = [w + c1 * d1 + c2 * d2 for w, d1, d2 in zip(weights, dw1, dw2)]
#   evaluator.assign_weights(perturbedWeights)
#   xent_i, acc_i, _ = evaluator.eval()
#   xent = xent + [xent_i]
#   acc = acc + [acc_i]
#   print('progress:', i+1, 'of', len(cfeed), '| time:', time())
#
# xent = np.reshape(np.array(xent), cc1.shape)
# plt.contourf(cc1, cc2, xent); plt.colorbar()
# plt.savefig('for_comet.jpg')
# experiment.log_image('for_comet.jpg')

