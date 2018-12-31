import tensorflow as tf
import numpy as np
import resnet_model
from os.path import join, basename, dirname

class Evaluator(object):

  def __init__(self, loader, hps=None):

    if hps == None:
      self.hps = resnet_model.HParams(batch_size=None,
                                      num_classes=10,
                                      min_lrn_rate=0.0001,
                                      lrn_rate=0.1,
                                      num_residual_units=3,
                                      resnet_width=1,
                                      use_bottleneck=False,
                                      weight_decay_rate=0.0,
                                      spec_coef=0.0,
                                      relu_leakiness=0.1,
                                      projvec_beta=0.0,
                                      max_grad_norm=30,
                                      normalizer='layernormdev',
                                      specreg_bn=False,
                                      spec_sign=1,
                                      optimizer='mom')
    else:
      self.hps = hps

    # model and data loader
    self.model = resnet_model.ResNet(self.hps, 'eval')
    self.model.build_graph()
    self.loader = loader

    # session
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))


  def restore_weights(self, log_dir):

    # checkpoint file
    ckpt_file = join(log_dir, 'model.ckpt')

    # look for ckpt to restore
    try:
      ckpt_state = tf.train.get_checkpoint_state(log_dir)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('EVAL: Cannot restore checkpoint: %s', e)
      return True
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('EVAL: No model to eval yet at %s', log_dir)
      time.sleep(10)
      return True

    # restore the checkpoint
    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(self.sess, ckpt_file)

  def assign_weights(self, weights):
    self.sess.run([tf.assign(t,w) for t,w in zip(tf.trainable_variables(), weights)])

  def get_weights(self):
    return self.sess.run(tf.trainable_variables())

  def eval(self):
    # run evaluation session over entire eval set in batches
    total_prediction, correct_prediction = 0, 0
    running_xent = running_tot = 0
    for batch_idx, (images, target) in enumerate(self.loader):
      # load batch
      images = images.permute(0,2,3,1).numpy(); target = target.numpy()
      target = np.eye(self.hps.num_classes)[target]
      # run the model to get xent and precision
      (predictions, truth, xentPerExample, global_step) = self.sess.run(
        [self.model.predictions, self.model.labels, self.model.xentPerExample, self.model.global_step],
        {self.model._images: images, self.model.labels: target})
      # keep running tally
      running_xent += np.sum(xentPerExample)
      running_tot += len(images)
      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]
      # premature abortion
      # if batch_idx > len(self.loader)/4: break
    # aggregate scores
    precision = 1.0 * correct_prediction / total_prediction
    xent = running_xent/running_tot
    return xent, precision, global_step

  def get_sharpest_dir(self):

    num_power_iter = 10
    for power_iter in range(num_power_iter): # do power iteration to find spectral radius
      # accumulate gradients over entire batch
      tstart = time.time()
      sess.run(model.zero_op)
      for bid, (batchimages, batchtarget) in enumerate(cleanloader):
        # change from torch tensor to numpy array
        batchimages = batchimages.permute(0,2,3,1).numpy()
        batchtarget = batchtarget.numpy()
        batchtarget = np.eye(hps.num_classes)[batchtarget]
        # hack
        dirtyOne = 0*np.ones_like(batchtarget)
        dirtyNeg = 1*np.ones_like(batchtarget)
        # accumulate hvp
        sess.run(model.accum_op, {model._images: batchimages, model.labels: batchtarget, model.dirtyOne: dirtyOne, model.dirtyNeg: dirtyNeg})
        # get coarse xHx
        if power_iter==num_power_iter-1 and bid==0:
          xHx_batch = sess.run(model.xHx, {model._images: batchimages, model.labels: batchtarget, model.dirtyOne: dirtyOne, model.dirtyNeg: dirtyNeg})
      # calculated projected hessian eigvec and eigval
      projvec_op, corr_iter, xHx, nextProjvec = sess.run([model.projvec_op, model.projvec_corr, model.xHx, model.projvec])
      print('TRAIN: power_iter', power_iter, 'xHx', xHx, 'corr_iter', corr_iter, 'elapsed', time.time()-tstart)

    # compute correlation between projvec of different epochs
    if 'projvec' in locals():
      corr_period = np.sum([np.dot(p.ravel(),n.ravel()) for p,n in zip(projvec, nextProjvec)]) # correlation of projvec of consecutive periods (5000 batches)
      print('TRAIN: projvec mag', utils.global_norm(projvec), 'nextProjvec mag', utils.global_norm(nextProjvec), 'corr_period', corr_period) # ensure unit magnitude
      experiment.log_metric('corr_period', corr_period, global_step)
    projvec = nextProjvec
