import tensorflow as tf
from datetime import datetime
import numpy as np
from numpy.linalg import norm
import os
import glob
import re
import warnings
import time


def itercycle(sequence):
  '''return iterable which iterates infinitely by cycling through the sequence'''
  while True:
    iterable = iter(sequence)
    for elem in iterable:
      yield elem

def global_norm(vec):
  return np.sqrt(np.sum([np.sum(p**2) for p in vec]))

def global_unitvec_like(vec):
  unitvec = [np.random.randn(*v.shape) for v in vec]
  magnitude = global_norm(unitvec)
  unitvec = [p/magnitude for p in unitvec]
  return unitvec

def unitvec_like(vec):
  unitvec = np.random.randn(*vec.shape)
  return unitvec / norm(unitvec.ravel())

def list2dotprod(listoftensors1, listoftensors2):
  '''compute the dot product of two lists of tensors (such as those returned when you call tf.gradients) as if each
  list were one concatenated tensor'''
  return tf.add_n([tf.reduce_sum(tf.multiply(a,b)) for a,b in zip(listoftensors1,listoftensors2)])

def list2euclidean(listoftensors1, listoftensors2):
  '''compute the euclidean distance between two lists of tensors (such as those returned when you call tf.gradients) as if each
  list were one concatenated tensor'''
  return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(tf.subtract(a,b))) for a,b in zip(listoftensors1,listoftensors2)]))

def list2norm(listOfTensors):
  '''compute the 2-norm of a list of tensors (such as those returned when you call tf.gradients) AS IF
  list were one concatenated tensor'''
  return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(a)) for a in listOfTensors]))

def list2corr(listOfTensors1, listOfTensors2):
  dotprod = list2dotprod(listOfTensors1, listOfTensors2)
  norm1 = list2norm(listOfTensors1)
  norm2 = list2norm(listOfTensors2)
  return tf.divide(dotprod,tf.multiply(norm1,norm2))


def filtnorm(trainable_variables):
  '''return a list of tensors (matching the shape of trainable_variables) containing the norms of each filter'''
  with tf.variable_scope('filtnorm'):
    filtnorm = []
    for r in trainable_variables: # iterate by layer
      if len(r.shape)==4:  # conv layer
        f = []
        for i in range(r.shape[-1]):
          f.append(tf.multiply(tf.ones_like(r[:,:,:,i]), tf.norm(r[:,:,:,i])))
        filtnorm.append(tf.stack(f,axis=3))
      elif len(r.shape)==2: # fully connected layer
        f = []
        for i in range(r.shape[-1]):
          f.append(tf.multiply(tf.ones_like(r[:,i]), tf.norm(r[:,i])))
        filtnorm.append(tf.stack(f,axis=1))
      elif len(r.shape)==1: # bn and bias layer
        # f = tf.multiply(tf.ones_like(r), tf.norm(r)) # do not do any normalization/scaling to bias/bn variables
        f = tf.multiply(tf.zeros_like(r), tf.norm(r)) # zero out bias/bn variables so their curvature doesnt affect hessian
        filtnorm.append(f)
      else:
        print('invalid number of dimensions in layer, should be 1, 2, or 4')
  return filtnorm

def layernormdev(trainable_variables):
  '''return a list of tensors (matching the shape of trainable_variables) containing the norms of the DEVIATIONS of each layer'''
  return [tf.norm(tf.subtract(t,tf.reduce_mean(t))) for t in trainable_variables]

def layernorm(trainable_variables):
  '''return a list of tensors (matching the shape of trainable_variables) containing the norms of each layer'''
  return [tf.multiply(tf.norm(t), tf.ones_like(t)) for t in trainable_variables]

# todo not verified
def filtnormbyN(trainable_variables):
  ''' divide each filternorm by the count of elements in each filter '''
  norm_values = filtnorm(trainable_variables)
  filtcnt = [tf.size(f) for f in norm_values]
  return [tf.divide(f, tf.cast(c, dtype=tf.float32)) for c,f in zip(filtcnt, norm_values)]


def hessian_fullbatch(sess, model, loader, num_classes=10, is_training=False, num_power_iter=10, experiment=None, ckpt=None):
  '''compute fullbatch hessian eigenvalue/eigenvector given an image loader and model'''

  for power_iter in range(num_power_iter): # do power iteration to find spectral radius
    # accumulate gradients over entire batch
    tstart = time.time()
    sess.run(model.zero_op)
    for bid, (batchimages, batchtarget) in enumerate(loader):

      # change from torch tensor to numpy array
      # batchimages = batchimages.permute(0,2,3,1).numpy()
      # batchtarget = batchtarget.numpy()
      # batchtarget = np.eye(num_classes)[batchtarget]
      batchimages, batchtarget = cifar_torch_to_numpy(batchimages, batchtarget)

      # accumulate hvp
      if is_training:
        dirtyOne = 0*np.ones_like(batchtarget)
        dirtyNeg = 1*np.ones_like(batchtarget)
        sess.run(model.accum_op, {model._images: batchimages, model.labels: batchtarget, model.dirtyOne: dirtyOne, model.dirtyNeg: dirtyNeg})
      else:
        sess.run(model.accum_op, {model._images: batchimages, model.labels: batchtarget})

      if power_iter==0 and bid==0: print('Starting hessian calculation, just did first batch of first power iteration')

    # calculated projected hessian eigvec and eigval
    projvec_op, projvec_corr, xHx, projvec, normvalues = sess.run([model.projvec_op, model.projvec_corr, model.xHx, model.projvec, model.normvalues])
    print('HESSIAN: power_iter', power_iter, 'xHx', xHx, 'projvec_corr', projvec_corr, 'projvec_magnitude', global_norm(projvec), 'elapsed', time.time()-tstart)
    if experiment != None and ckpt != None:
      experiment.log_metric('eigval/'+ckpt, xHx, power_iter)
      experiment.log_metric('eigvec_corr/'+ckpt, projvec_corr, power_iter)

  return xHx, projvec, projvec_corr

def fwd_gradients(ys, xs, d_xs=None):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward.-- taken from: https://github.com/renmengye/tensorflow-forward-ad/issues/2"""
  v = [tf.ones_like(tensor=y) for y in ys]  # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  if d_xs==None: d_xs = [tf.ones_like(tensor=_g) for _g in g]
  return tf.gradients(g, v, grad_ys=d_xs)  # tf.gradients(ys,xs,grad_ys=d_xs)#tf.gradients(g,v,grad_ys=d_xs)

def count_params(params_list=None):
  '''count the total number of parameters within in a list of parameters tensors of varying shape'''
  if params_list==None:
    params_list = tf.trainable_variables()
  return np.sum([np.prod(v.get_shape().as_list()) for v in params_list])

def flatten_and_concat(listOfTensors):
  '''flattens and concatenates a list of tensors. useful for turning list of weight tensors into a single 1D array'''
  return tf.concat([tf.reshape(t,[-1]) for t in listOfTensors], axis=0)

def maybe_download(source_url, filename, target_directory, filetype='folder', force=False):
  """Download the data from some website, unless it's already here."""
  if source_url==None or filename==None: return
  if target_directory==None: target_directory = os.getcwd()
  filepath = os.path.join(target_directory, filename)
  if os.path.exists(filepath) and not force:
    print(filepath+' already exists, skipping download')
  else:
    if not os.path.exists(target_directory):
      os.system('mkdir -p '+target_directory)
    if filetype=='folder':
      os.system('curl -L '+source_url+' > '+filename+'.zip')
      os.system('unzip -o '+filename+'.zip'+' -d '+filepath)
      os.system('rm '+filename+'.zip')
    elif filetype=='tar':
      os.system('curl -o '+filepath+'.tar '+source_url)
      os.system('tar xzvf '+filepath+'.tar --directory '+target_directory)
      os.system('rm '+filepath+'.tar')
    else:
      os.system('wget -O '+filepath+' '+source_url)

def get_dropbox_url(target_file, bin_path=''):
  '''get the url of a given directory or file on dropbox'''
  print('getting dropbox link for '+str(target_file))
  command_getlink = os.path.join(bin_path,'dbx')+' -q share '+target_file
  print(command_getlink)
  ckpt_link = os.popen(command_getlink)
  ckpt_link = list(ckpt_link)[0].strip('\n')
  return ckpt_link

def get_log_root(path):
  '''return a string representing an increment of 1 of the largest integer-valued directory
  in the project path. Error if not all directories in the project path are integer-valued.'''
  os.makedirs(path, exist_ok=True)
  files = os.listdir(path)
  files = filter(re.compile('^\d+$').match, files)
  files = [int(os.path.basename(f)) for f in files]
  if not len(files): return '0'
  return str(max(files)+1)

def merge_dicts(slave, master):
  '''merge two dictionaries, throw an exception if any of dict2's keys are in dict1.
  returns union of the two dicts. master dict overwrites slave dict'''
  if set(slave).intersection(set(master)):
    warnings.warn('Duplicate keys found: ' + str([k for k in slave.keys() if k in master.keys()]))
  merged = slave.copy()
  merged.update(master)
  return merged

def write_run_bashscript(log_dir, command_valid, command_train, verbose=False):
  '''write the bash script for reproducing the expeirment to file in log_dir'''
  with open(os.path.join(log_dir, 'run_command.sh'), 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('nohup '+command_valid+' & \n')
    f.write('nohup '+command_train+' & \n')
  if verbose: os.system('cat '+os.path.join(log_dir, 'run_command.sh'))

def debug_settings(FLAGS):
  FLAGS.num_resunits = 1
  FLAGS.batch_size = 2
  FLAGS.epoch_end = 1
  FLAGS.pretrain_url = None
  return FLAGS

# accumulate correct and total scores
class Accumulator():
  def __init__(self):
    self.cleancorr = self.dirtycorr = self.cleantot = self.dirtytot = 0
  def accum(self, predictions, cleanimages, cleantarget, dirtyimages, dirtytarget):
    cleanpred = np.argmax(predictions[:len(cleanimages)], axis=1)
    dirtypred = np.argmax(predictions[len(cleanimages):], axis=1)
    cleantrue = np.argmax(cleantarget, axis=1)
    dirtytrue = np.argmax(dirtytarget, axis=1)
    self.cleancorr += np.sum(cleanpred==cleantrue)
    self.dirtycorr += np.sum(dirtypred==dirtytrue)
    self.cleantot += len(cleanimages)
    self.dirtytot += len(dirtyimages)
  def get_accs(self):
    ret = self.cleancorr/self.cleantot, self.dirtycorr/self.dirtytot, self.cleancorr/self.cleantot - self.dirtycorr/self.dirtytot
    self.__init__()
    return ret

class Scheduler(object):
  """Sets learning_rate based on global step."""
  def __init__(self, args):
    self._lrn_rate = 0
    self._spec_coef = args.spec_coef_init
    self.args = args
  def after_run(self, global_step, steps_per_epoch):
    # warmup of spectral coefficient
    num_full_batch = 50000.
    num_warmup_epochs = self.args.num_warmup_epochs
    num_warmup_steps = num_warmup_epochs*num_full_batch/self.args.batch_size
    self._spec_coef = (self.args.spec_coef-self.args.spec_coef_init)*1/(1 + np.exp(-(global_step - self.args.spec_step_init) / (num_warmup_steps / 12))) \
                      + self.args.spec_coef_init
    # warmup of learning rate
    # self._lrn_rate  = self.args.lrn_rate*np.minimum(np.maximum((train_step-400000)/num_warmup_steps, 0), 1)
    epoch = float(global_step)/steps_per_epoch
    if epoch < 102.4:
      self._lrn_rate = self.args.lrn_rate
    elif epoch < 153.6:
      self._lrn_rate = self.args.lrn_rate*0.1
    elif epoch < 204.8:
      self._lrn_rate = self.args.lrn_rate*0.01
    else:
      self._lrn_rate = self.args.lrn_rate*0.001

def timenow():
  return datetime.now().strftime('%m-%d_%H:%M:%S_%f')

# load pretrained model from dropbox
def download_pretrained(log_dir, pretrain_dir=None, pretrain_url=None, bin_path=''):

  if pretrain_dir:
    pretrain_url = get_dropbox_url(pretrain_dir, bin_path=bin_path)

  # download pretrained model if a download url was specified
  print(pretrain_url)
  maybe_download(source_url=pretrain_url,
                 filename=log_dir,
                 target_directory=None,
                 filetype='folder',
                 force=True)

# change from torch tensor to numpy array
def cifar_torch_to_numpy(images, target):
  images = images.permute(0,2,3,1).numpy()
  target = np.eye(hps.num_classes)[target.numpy()]
  return images, target

# a hack needed to pass into the tf placeholder to make poison labels work
def reverse_softmax_probability_hack(cleantarget, dirtytarget, nodirty=False):
  if nodirty:
    dirtyOne = np.concatenate([ 0*np.ones_like(cleantarget),  0*np.ones_like(dirtytarget) ])
    dirtyNeg = np.concatenate([ 1*np.ones_like(cleantarget),  1*np.ones_like(dirtytarget) ])
  else:
    dirtyOne = np.concatenate([ 0*np.ones_like(cleantarget),  1*np.ones_like(dirtytarget) ])
    dirtyNeg = np.concatenate([ 1*np.ones_like(cleantarget), -1*np.ones_like(dirtytarget) ])
  return dirtyOne, dirtyNeg

