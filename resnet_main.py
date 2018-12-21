"""ResNet Train/Eval module.
adapted from tensorflow official resnet model
https://github.com/tensorflow/models/tree/master/research/resnet
"""
# import comet_ml in the top of your file
from comet_ml import Experiment, ExistingExperiment
import os
from os.path import join
import argparse
from cometml_api import api as cometapi
parser = argparse.ArgumentParser()
# file names
parser.add_argument('--log_root', default='debug7', type=str, help='Directory to keep the checkpoints.')
parser.add_argument('--ckpt_root', default='/root/ckpt', type=str, help='Parents directory of log_root')
parser.add_argument('--train_data_path', default='/root/datasets/cifar-100-binary/train.bin', type=str, help='Filepattern for training data.')
parser.add_argument('--eval_data_path', default='/root/datasets/cifar-100-binary/test.bin', type=str, help='Filepattern for eval data')
parser.add_argument('--bin_path', default='/root/bin', type=str, help='bin: directory of helpful scripts')
parser.add_argument('--dataset', default='cifar100', type=str, help='cifar10 or cifar100.')
# meta
parser.add_argument('--gpu', default='0', type=str, help='CUDA_VISIBLE_DEVICES=?')
parser.add_argument('--cpu_eval', action='store_true', help='use cpu for eval, overrides whatever is on --gpu')
parser.add_argument('--mode', default='train', type=str, help='train or eval.')
# network parameters
parser.add_argument('--num_resunits', default=3, type=int, help='Number of residual units n. There are 6*n+2 layers')
parser.add_argument('--resnet_width', default=1, type=int, help='Multiplier of the width of hidden layers. Base is (16,32,64)')
# training hyperparams
parser.add_argument('--lrn_rate', default=1e-1, type=float, help='initial learning rate to use for training')
parser.add_argument('--batch_size', default=128, type=int, help='batch size to use for training')
parser.add_argument('--weight_decay', default=0.0002, type=float, help='coefficient for the weight decay')
parser.add_argument('--augment', default=True, type=bool, help='use data augmentation.')
parser.add_argument('--epoch_end', default=256, type=int, help='ending epoch')
# specreg stuff
parser.add_argument('--spec_coef', default=0, type=float, help='coefficient for the spectral radius')
parser.add_argument('--spec_step_init', default=0, type=int, help='start spectral radius warmup at this training step')
parser.add_argument('--num_warmup_epochs', default=25, type=int, help='number of epochs over which speccoef upgrade spans')
parser.add_argument('--spec_coef_init', default=0.0, type=float, help='pre-warmup coefficient for the spectral radius')
parser.add_argument('--specreg_bn', default=False, type=bool, help='include bn weights in the calculation of the spectral regularization loss?')
parser.add_argument('--spec_sign', default=1., type=float, help='1 or -1, sign ofhouthe spectral regularization term, negative if looking for sharp minima')
parser.add_argument('--normalizer', default='layernormdev', type=str, help='normalizer to use (filtnorm, layernorm, layernormdev)')
parser.add_argument('--projvec_beta', default=0, type=float, help='discounting factor or "momentum" coefficient for averaging of projection vector')
# load pretrained
parser.add_argument('--scratch', action='store_true', help='force train from scratch')
parser.add_argument('--pretrain_url', default=None, type=str, help='url of pretrain directory')
parser.add_argument('--pretrain_dir', default=None, type=str, help='remote directory on dropbox of pretrain')
# general helpers
parser.add_argument('--max_grad_norm', default=30, type=float, help='maximum allowed gradient norm (values greater are clipped)')
parser.add_argument('--image_size', default=32, type=str, help='Image side length.')
parser.add_argument('--eval_batch_size', default=50, type=int, help='Smaller means less memory, but more time')
FLAGS = parser.parse_args()
log_dir = join(FLAGS.ckpt_root, FLAGS.log_root)
os.makedirs(log_dir, exist_ok=True)
if FLAGS.scratch: FLAGS.pretrain_url = FLAGS.pretrain_dir = None


# comet stuff
if not os.path.exists(join(log_dir, 'comet_expt_key.txt')):
  # initialize comet experiment
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                          project_name='hessreg-reproduce', workspace="wronnyhuang")
  # write experiment key
  os.makedirs(log_dir, exist_ok=True)
  with open(join(log_dir, 'comet_expt_key.txt'), 'w+') as f:
    f.write(experiment.get_key())
else:
  # read previous experiment key
  with open(join(log_dir, 'comet_expt_key.txt'), 'r') as f:
    comet_key = f.read()
  # iniitalize comet experiment
  experiment = ExistingExperiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", previous_experiment=comet_key, parse_args=False)
# set name of experiment and write experiment key into log directory
experiment.set_name(FLAGS.log_root)
experiment.log_multiple_params(vars(FLAGS))

# import the other packages
import time
from datetime import datetime
import six
import numpy as np
import tensorflow as tf
import cifar_input
import resnet_model
import utils
import sys
from cifar_loader_torch import cifar_loader
import subprocess
from subprocess import PIPE, STDOUT
from glob import glob
from del_old_ckpt import _del_old_ckpt
gpu_options = tf.GPUOptions(allow_growth=True)
timenow = lambda: datetime.now().strftime('%m-%d %H:%M:%S')
hostname = open('/root/misc/hostname.log').read()
print('====================> HOST: docker @ '+hostname)
experiment.log_other('hostname', hostname)

# FLAGS = utils.debug_settings(FLAGS)
if FLAGS.mode == 'train': open('/root/resnet_main.bash','w').write('cd /root/repo/hessreg && python '+' '.join(sys.argv)) # write command to the log

def train(hps):

  # start evaluation process
  popen_args = dict(shell=True, universal_newlines=True, stdout=PIPE, stderr=STDOUT)
  command_valid = 'python src/resnet_main.py --mode=eval ' + ' '.join(sys.argv[1:])
  valid = subprocess.Popen(command_valid, **popen_args)
  print('EVAL: started validation from train process using command: ', command_valid)

  # set gpu
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  # load pretrained model from dropbox
  if FLAGS.pretrain_dir:
    FLAGS.pretrain_url = utils.get_dropbox_url(FLAGS.pretrain_dir, bin_path=FLAGS.bin_path)

  # download pretrained model if a download url was specified
  print(FLAGS.pretrain_url)
  utils.maybe_download(source_url=FLAGS.pretrain_url,
                       filename=log_dir,
                       target_directory=None,
                       filetype='folder',
                       force=True)

  # build graph
  # images, labels = cifar_input.build_input(
  #     FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode, FLAGS.augment)

  # build graph [new version]
  cleanloader, dirtyloader, testloader = cifar_loader('/root/datasets', batchsize=hps.batch_size, fracdirty=.5)
  images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
  labels = tf.placeholder(dtype=tf.float32, shape=(None, hps.num_classes))


  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
  all_summaries = tf.summary.merge([model.summaries, tf.summary.scalar(FLAGS.mode+'/Precision', precision)])

  class Scheduler(object):
    """Sets learning_rate based on global step."""
    def __init__(self):
      self._lrn_rate = 0
      self._spec_coef = FLAGS.spec_coef_init
    def after_run(self, global_step):
      # warmup of spectral coefficient
      num_full_batch = 50000.
      num_warmup_epochs = FLAGS.num_warmup_epochs
      num_warmup_steps = num_warmup_epochs*num_full_batch/FLAGS.batch_size
      self._spec_coef = (FLAGS.spec_coef-FLAGS.spec_coef_init)*1/(1 + np.exp(-(global_step - FLAGS.spec_step_init) / (num_warmup_steps / 12))) \
                        + FLAGS.spec_coef_init
      # warmup of learning rate
      # self._lrn_rate  = FLAGS.lrn_rate*np.minimum(np.maximum((train_step-400000)/num_warmup_steps, 0), 1)
      if global_step < 40000:
        self._lrn_rate = FLAGS.lrn_rate
      elif global_step < 60000:
        self._lrn_rate = FLAGS.lrn_rate*0.1
      elif global_step < 80000:
        self._lrn_rate = FLAGS.lrn_rate*0.01
      else:
        self._lrn_rate = FLAGS.lrn_rate*0.001

  # initialize saver, writer
  ckpt_file = join(log_dir, 'model.ckpt')
  saver = tf.train.Saver(max_to_keep=1)
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

  # initialize session, queuerunner
  print('===================> TRAIN: STARTING SESSION at '+timenow())
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  print('===================> TRAIN: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])
  tf.train.start_queue_runners(sess)
  scheduler = Scheduler()


  # load checkpoint
  ckpt_state = tf.train.get_checkpoint_state(log_dir)
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('TRAIN: No pretrained model. Initializing from random')
    sess.run(tf.global_variables_initializer())
  else:
    saver.restore(sess, ckpt_file)
    tf.logging.info('TRAIN: Loading checkpoint %s', ckpt_state.model_checkpoint_path)

  for epoch in range(FLAGS.epoch_end):

    dirtyloaderiter = iter(dirtyloader)
    for batch_idx, (cleanimages, cleantargets) in enumerate(cleanloader):

      # get dirty labeled images
      dirtyimages, dirtytargets = dirtyloaderiter.__next__()

      # change from torch tensor to numpy array
      dirtyimages = dirtyimages.numpy(); dirtytargets = dirtytargets.numpy()
      cleanimages = cleanimages.numpy(); cleantargets = cleantargets.numpy()

      # combine into a batch
      batchimages = np.concatenate([ cleanimages, dirtyimages ])
      batchtargets = np.concatenate([ cleantargets, dirtytargets ])
      batchtargets = np.eye(10)[batchtargets] # convert to onehot

      # run the graph
      summaries, _, global_step, loss, prec = sess.run(
        [all_summaries, model.train_op, model.global_step, model.loss, precision],
        feed_dict={model.lrn_rate: scheduler._lrn_rate, images: batchimages, labels: batchtargets})

      scheduler.after_run(global_step)

      # save ckpt and summary every 100 iters
      if np.mod(global_step, 100)==0:
        summary_writer.add_summary(summaries, global_step)
        summary_writer.flush()
        saver.save(sess, ckpt_file)
        tf.logging.info('TRAIN: loss: %.3f, precision: %.3f, global_step: %d, epoch: %d, time: %s' %
                        (loss, prec, global_step, epoch, timenow()))

      # save initial weights on first iteration
      # if global_step==1:
      #   _ = sess.run(model.init_weights_op)
      #   print('TRAIN: stored initial weights into variable init_weights')

      # # upload checkpoint every 1k iters
      # if np.mod(global_step, 1)==0:
      #   upload_command = 'dropbox_uploader upload '+log_dir+'/* ckpt/'+FLAGS.log_root+'/'+FLAGS.log_root+'-'+str(global_step)+'/'
      #   print(upload_command)
      #   subprocess.run(upload_command, shell=True)

    # closeout script
    print('TRAIN: Done Training at '+str(global_step)+' steps')
    os.system(join(FLAGS.bin_path,'rek')+' "mode=eval.*log_root='+FLAGS.log_root+'"') # kill evaluation processes

    # retrieve best evaluation result
    cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
    metricSummaries = cometapi.get_raw_metric_summaries(experiment.get_key())
    metricSummaries = {b.pop('name'): b for b in metricSummaries}
    bestEvalPrecision = metricSummaries['eval/Best Precision']['valueMax']
    print('sigoptObservation='+str(bestEvalPrecision))


def evaluate(hps):

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if FLAGS.cpu_eval else FLAGS.gpu # run eval on cpu

  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode, FLAGS.augment)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()

  ckpt_file = join(log_dir, 'model.ckpt')
  saver = tf.train.Saver(max_to_keep=1)
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  print('===================> EVAL: STARTING SESSION at '+timenow())
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  print('===================> EVAL: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])
  tf.train.start_queue_runners(sess)

  # continuously evaluate until process is killed
  best_precision = 0.0
  while True:

    _del_old_ckpt(log_dir) # keep only newest ckpt

    # load checkpoint
    try:
      ckpt_state = tf.train.get_checkpoint_state(log_dir)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('EVAL: Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('EVAL: No model to eval yet at %s', log_dir)
      time.sleep(10)
      continue
    # tf.logging.info('EVAL: Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_file)

    # run evaluation session over entire eval set in batches
    total_prediction, correct_prediction = 0, 0
    eval_batch_count = int(10000/FLAGS.eval_batch_size)
    for _ in six.moves.range(eval_batch_count):
      (summaries, predictions, truth, global_step, loss) = sess.run(
          [model.summaries, model.predictions,
           model.labels, model.global_step, model.xent])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag=FLAGS.mode+'/Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, global_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag=FLAGS.mode+'/Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, global_step)
    summary_writer.add_summary(summaries, global_step)
    # tf.logging.info('EVAL: loss: %.3f, precision: %.3f, best precision: %.3f time: %s' %
    #                 (loss, precision, best_precision, timenow()))
    summary_writer.flush()

    time.sleep(60)

def main(_):

  # put train and eval run logs in the log directory
  FLAGS.train_dir = join(log_dir, 'train')
  FLAGS.eval_dir = join(log_dir, 'eval')
  FLAGS.augment = False if FLAGS.mode=='eval' else True

  if FLAGS.mode == 'train':
    batch_size = FLAGS.batch_size
  elif FLAGS.mode == 'eval':
    batch_size = FLAGS.eval_batch_size

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=FLAGS.num_resunits,
                             resnet_width=FLAGS.resnet_width,
                             use_bottleneck=False,
                             weight_decay_rate=FLAGS.weight_decay,
                             spec_coef=FLAGS.spec_coef,
                             relu_leakiness=0.1,
                             projvec_beta=FLAGS.projvec_beta,
                             max_grad_norm=FLAGS.max_grad_norm,
                             normalizer=FLAGS.normalizer,
                             specreg_bn=FLAGS.specreg_bn,
                             spec_sign=FLAGS.spec_sign,
                             optimizer='mom')

  # with tf.device(dev):
  if FLAGS.mode == 'train':
    train(hps)
  elif FLAGS.mode == 'eval':
    evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()