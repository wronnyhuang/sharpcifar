from comet_ml import Experiment, ExistingExperiment
import numpy as np
import tensorflow as tf
from resnet_evaluator import Evaluator
from cifar_loader_torch import cifar_loader
from utils import unitvec_like, global_norm
from functools import reduce
from numpy.linalg import norm
import matplotlib.pyplot as plt
from datetime import datetime
import os
time = lambda: datetime.now().strftime('%m-%d %H:%M:%S')

# comet stuff
if not os.path.exists('comet_expt_key.txt'):
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                          project_name='landscape', workspace="wronnyhuang")
  open('comet_expt_key.txt', 'w+').write(experiment.get_key())
else:
  comet_key = open('comet_expt_key.txt', 'r').read()
  experiment = ExistingExperiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", previous_experiment=comet_key, parse_args=False)

np.random.seed(1239)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cleanloader, _, _, _ = cifar_loader('/root/datasets', batchsize=2*512, fracdirty=.5)
evaluator = Evaluator(cleanloader)
evaluator.restore_weights_dropbox('ckpt/poison-filtnorm-weaker')
evaluator.eval()
weights = evaluator.get_weights()

eigval, dw1, projvec_corr = evaluator.get_hessian()
# dw1 = evaluator.get_random_dir()
dw2 = evaluator.get_random_dir()

cfeed = .5 * np.linspace(-1, 1, 100)

xent = np.zeros(len(cfeed))
for i, c in enumerate(cfeed):

  perturbedWeights = [w + c * d1 for w, d1 in zip(weights, dw2)]
  evaluator.assign_weights(perturbedWeights)
  xent[i], acc, _ = evaluator.eval()
  print('progress:', i+1, 'of', len(cfeed), '| time:', time())

xent = np.reshape(np.array(xent), cfeed.shape)
plt.plot(cfeed, xent)
experiment.log_figure('landscape')

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

