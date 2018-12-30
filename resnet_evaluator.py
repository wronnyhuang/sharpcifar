import tensorflow as tf
import numpy as np
import resnet_model
from os.path import join, basename, dirname

class Evaluator(object):

  def __init__(self, loader, hps=None):

    if hps == None:
      self.hps = resnet_model.HParams(batch_size=256,
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
    # aggregate scores
    precision = 1.0 * correct_prediction / total_prediction
    xent = running_xent/running_tot
    return xent, precision, global_step
