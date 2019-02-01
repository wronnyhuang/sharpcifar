import tensorflow as tf
import numpy as np
import resnet_model
from os.path import join, basename, dirname
import utils
from utils import unitvec_like
import time
import shutil

class Evaluator(object):

  def __init__(self, loader, args=None):

    self.args = args

    # model and data loader
    self.model = resnet_model.ResNet(self.args, mode='eval')
    self.loader = loader

    # session
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())

  def restore_weights(self, log_dir):

    ckpt_file = join(log_dir, 'model.ckpt')

    # look for ckpt to restore
    try:
      ckpt_state = tf.train.get_checkpoint_state(log_dir)
    except tf.errors.OutOfRangeError as e:
      print('EVAL: Cannot restore checkpoint: %s', e)
      return True
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      print('EVAL: No model to eval yet at %s', log_dir)
      time.sleep(10)
      return True

    # restore the checkpoint
    var_list = list(set(tf.global_variables())-set(tf.global_variables('accum'))-set(tf.global_variables('projvec')))
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
    saver.restore(self.sess, ckpt_file)

  def restore_weights_dropbox(self, pretrain_dir):
    logdir = utils.timenow()
    utils.download_pretrained(log_dir=join(args.bin_path, 'ckpt', logdir), pretrain_dir=pretrain_dir)
    self.restore_weights(join(args.bin_path, 'ckpt', logdir))
    shutil.rmtree(join(args.bin_path, 'ckpt', logdir))
    print('Ckpt restored from', pretrain_dir)

  def assign_weights(self, weights):
    self.sess.run([tf.assign(t,w) for t,w in zip(tf.trainable_variables(), weights)])
    self.eigval = self.eigvec = self.projvec_corr = None

  def get_weights(self):
    return self.sess.run(tf.trainable_variables())

  def eval(self, loader=None):
    # run evaluation session over entire eval set in batches
    if loader==None: loader = self.loader
    total_prediction, correct_prediction = 0, 0
    running_xent = running_tot = 0
    for batch_idx, (images, target) in enumerate(loader):
      # load batch
      images, target = utils.cifar_torch_to_numpy(images, target, num_classes=self.args.num_classes, onehot=True)
      # run the model to get xent and precision
      (predictions, truth, xentPerExample, global_step) = self.sess.run(
        [self.model.predictions, self.model.labels, self.model.xentPerExample, self.model.global_step],
        {self.model._images: images,
         self.model.labels: target})
      # keep running tally
      running_xent += np.sum(xentPerExample)
      running_tot += len(images)
      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]
    # aggregate scores
    precision = 1.0 * correct_prediction / total_prediction
    xent = running_xent/running_tot
    return xent, precision, global_step

  def get_filtnorm(self, weights):
    return self.sess.run(utils.filtnorm(weights))

  def get_hessian(self, loader=None, num_classes=10, num_power_iter=10, experiment=None, ckpt=None):
    if loader==None: loader = self.loader
    self.eigval, self.eigvec, self.projvec_corr = \
      utils.hessian_fullbatch(self.sess, self.model, loader, num_classes, is_training_dirty=False, num_power_iter=num_power_iter, experiment=experiment, ckpt=ckpt)
    return self.eigval, self.eigvec, self.projvec_corr

  def get_random_dir(self):
    # create random direction vectors in weight space

    randdir = []
    weights = self.get_weights()
    filtnorms = self.get_filtnorm(weights)
    for l, (layer, layerF) in enumerate(zip(weights, filtnorms)):

      # handle nonconvolutional layers
      if len(layer.shape)==2: layer = layer[None,None,:,:]; layerF = layerF[None,None,:,:]
      elif len(layer.shape)!=4: randdir = randdir + [np.zeros(layer.shape)]; continue

      # permute so filter index is first
      layer = layer.transpose(3,0,1,2)
      layerF = layerF.transpose(3,0,1,2)

      # make randdir filters that has same norm as the corresponding filter in the weights
      layerR = np.array([ unitvec_like(filter)*filtnorm for (filter, filtnorm) in zip(layer, layerF) ])

      # permute back to standard
      layerR = layerR.transpose(1,2,3,0)
      layerR = np.squeeze(layerR)
      randdir = randdir + [layerR]

    return randdir





