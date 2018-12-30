from comet_ml import Experiment
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

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='landscape', workspace="wronnyhuang")

np.random.seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cleanloader, _, _, _ = cifar_loader('/root/datasets', batchsize=128, fracdirty=.5)
evaluator = Evaluator(cleanloader)
evaluator.restore_weights('/root/ckpt/poison-strong')
weights = evaluator.get_weights()

# # do some statistics on weights and plot histogram
# weightsPermuted = [w.transpose(3,0,1,2) if len(w.shape)>2 else w[None,None,:,:].transpose(3,0,1,2) if len(w.shape)==2 else w for w in weights]
# filter2norm = lambda filters: [np.linalg.norm(f.ravel()) for f in filters]
# filternorms = [filter2norm(filters) if len(filters.shape) > 1 else None for filters in weightsPermuted]
# filternorms = reduce(lambda running_norms, norms: running_norms + norms if norms != None else running_norms, filternorms)
# hist(filternorms, 20); show()

def get_random_dir(weights):
  # create random direction vectors in weight space

  randdir = []
  for l, layer in enumerate(weights):

    # handle nonconvolutional layers
    if len(layer.shape)==2: layer = layer[None,None,:,:]
    elif len(layer.shape)!=4: randdir = randdir + [np.zeros(layer.shape)]; continue

    # permute so filter index is first
    layer = layer.transpose(3,0,1,2)

    # make randdir filters that has same norm as the corresponding filter in the weights
    layerR = np.array([ unitvec_like(filter)*norm(filter.ravel()) for filter in layer ])

    # permute back to standard
    layerR = layerR.transpose(1,2,3,0)
    layerR = np.squeeze(layerR)
    randdir = randdir + [layerR]

  return randdir

dw1 = get_random_dir(weights)
dw2 = get_random_dir(weights)

print(dw1[0][0,0,0,0])

clin = 1.5 * np.linspace(-1, 1,40)
cc1, cc2 = np.meshgrid(clin, clin)
cfeed = np.column_stack([cc1.ravel(), cc2.ravel()])

xent = []; acc = []
for i, (c1, c2) in enumerate(cfeed):

  perturbedWeights = [w + c1 * d1 + c2 * d2 for w, d1, d2 in zip(weights, dw1, dw2)]
  evaluator.assign_weights(perturbedWeights)
  xent_i, acc_i, _ = evaluator.eval()
  xent = xent + [xent_i]
  acc = acc + [acc_i]
  print('progress:', i+1, 'of', len(cfeed), '| time:', time())

xent = np.reshape(np.array(xent), cc1.shape)
plt.contourf(cc1, cc2, xent); plt.colorbar()
plt.savefig('for_comet.jpg')
experiment.log_image('for_comet.jpg')

