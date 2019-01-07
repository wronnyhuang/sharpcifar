# import comet_ml in the top of your file
from comet_ml import Experiment, ExistingExperiment
import tensorflow as tf
import os
from os.path import join
import argparse
from cometml_api import api as cometapi
import time
from datetime import datetime
import six
import numpy as np
import cifar_input
import resnet_model
import utils
from utils import timenow, Scheduler
import sys
from cifar_loader_torch import cifar_loader
from resnet_evaluator import Evaluator
import subprocess
from subprocess import PIPE, STDOUT
from glob import glob
from del_old_ckpt import _del_old_ckpt
parser = argparse.ArgumentParser()
# file names
parser.add_argument('--log_root', default='debug', type=str, help='Directory to keep the checkpoints.')
parser.add_argument('--ckpt_root', default='/root/ckpt', type=str, help='Parents directory of log_root')
parser.add_argument('--train_data_path', default='/root/datasets/cifar-100-binary/train.bin', type=str, help='Filepattern for training data.')
parser.add_argument('--eval_data_path', default='/root/datasets/cifar-100-binary/test.bin', type=str, help='Filepattern for eval data')
parser.add_argument('--bin_path', default='/root/bin', type=str, help='bin: directory of helpful scripts')
parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 or cifar100.')
# meta
parser.add_argument('--gpu', default='0', type=str, help='CUDA_VISIBLE_DEVICES=?')
parser.add_argument('--cpu_eval', action='store_true', help='use cpu for eval, overrides whatever is on --gpu')
parser.add_argument('--mode', default='train', type=str, help='train or eval.')
# poison data
parser.add_argument('--nodirty', action='store_true')
parser.add_argument('--fracdirty', default=.5, type=float) # should be < .5 for now
# network parameters
parser.add_argument('--num_resunits', default=3, type=int, help='Number of residual units n. There are 6*n+2 layers')
parser.add_argument('--resnet_width', default=1, type=int, help='Multiplier of the width of hidden layers. Base is (16,32,64)')
# training hyperparams
parser.add_argument('--lrn_rate', default=1e-1, type=float, help='initial learning rate to use for training')
parser.add_argument('--batch_size', default=128, type=int, help='batch size to use for training')
parser.add_argument('--weight_decay', default=0.0, type=float, help='coefficient for the weight decay')
parser.add_argument('--augment', default=True, type=bool, help='use data augmentation.')
parser.add_argument('--epoch_end', default=256, type=int, help='ending epoch')
# specreg stuff
parser.add_argument('--spec_coef', default=0, type=float, help='coefficient for the spectral radius')
parser.add_argument('--spec_step_init', default=0, type=int, help='start spectral radius warmup at this training step')
parser.add_argument('--num_warmup_epochs', default=25, type=int, help='number of epochs over which speccoef upgrade spans')
parser.add_argument('--spec_coef_init', default=0.0, type=float, help='pre-warmup coefficient for the spectral radius')
parser.add_argument('--specreg_bn', default=False, type=bool, help='include bn weights in the calculation of the spectral regularization loss?')
parser.add_argument('--spec_sign', default=1., type=float, help='1 or -1, sign ofhouthe spectral regularization term, negative if looking for sharp minima')
parser.add_argument('--normalizer', default='filtnorm', type=str, help='normalizer to use (filtnorm, layernorm, layernormdev)')
parser.add_argument('--projvec_beta', default=0, type=float, help='discounting factor or "momentum" coefficient for averaging of projection vector')
# load pretrained
parser.add_argument('--pretrain_url', default=None, type=str, help='url of pretrain directory')
parser.add_argument('--pretrain_dir', default=None, type=str, help='remote directory on dropbox of pretrain')
# general helpers
parser.add_argument('--max_grad_norm', default=30, type=float, help='maximum allowed gradient norm (values greater are clipped)')
parser.add_argument('--image_size', default=32, type=str, help='Image side length.')
args = parser.parse_args()
log_dir = join(args.ckpt_root, args.log_root)
os.makedirs(log_dir, exist_ok=True)

# comet stuff for logging
if not os.path.exists(join(log_dir, 'comet_expt_key.txt')):
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='sharpcifar', workspace="wronnyhuang")
  os.makedirs(log_dir, exist_ok=True)
  with open(join(log_dir, 'comet_expt_key.txt'), 'w+') as f: f.write(experiment.get_key())
else:
  with open(join(log_dir, 'comet_expt_key.txt'), 'r') as f: comet_key = f.read()
  experiment = ExistingExperiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", previous_experiment=comet_key, parse_args=False)

def train(hps):

  # start evaluation process
  popen_args = dict(shell=True, universal_newlines=True, stdout=PIPE, stderr=STDOUT)
  command_valid = 'python resnet_main.py --mode=eval ' + ' '.join(sys.argv[1:])
  valid = subprocess.Popen(command_valid, **popen_args)
  print('EVAL: started validation from train process using command: ', command_valid)

  # set gpu
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  # load pretrained model
  utils.download_pretrained(log_dir, pretrain_dir=args.pretrain_dir)

  # build graph [new version]
  cleanloader, dirtyloader, testloader, trainloader = cifar_loader('/root/datasets', batchsize=hps.batch_size, fracdirty=args.fracdirty)
  model = resnet_model.ResNet(hps, args.mode)
  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  # initialize saver, writer
  ckpt_file = join(log_dir, 'model.ckpt')
  saver = tf.train.Saver(max_to_keep=1)
  summary_writer = tf.summary.FileWriter(args.train_dir)

  # initialize session, queuerunner
  print('===================> TRAIN: STARTING SESSION at '+timenow())
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
  print('===================> TRAIN: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])
  scheduler = Scheduler(args)

  # load checkpoint
  ckpt_state = tf.train.get_checkpoint_state(log_dir)
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    print('TRAIN: No pretrained model. Initializing from random')
    sess.run(tf.global_variables_initializer())
  else:
    saver.restore(sess, ckpt_file)
    print('TRAIN: Loading checkpoint %s', ckpt_state.model_checkpoint_path)

  for epoch in range(args.epoch_end): # loop over epochs

    # loop over batches
    cleancorr = dirtycorr = cleantot = dirtytot = 0
    for batchid, ((cleanimages, cleantarget), (dirtyimages, dirtytarget)) in enumerate(zip(cleanloader, utils.itercycle(dirtyloader))):

      # change from torch tensor to numpy array
      dirtyimages = dirtyimages.permute(0,2,3,1); cleanimages = cleanimages.permute(0,2,3,1)
      dirtyimages = dirtyimages.numpy(); dirtytarget = dirtytarget.numpy()
      cleanimages = cleanimages.numpy(); cleantarget = cleantarget.numpy()
      # convert to onehot
      dirtytarget = np.eye(hps.num_classes)[dirtytarget]
      cleantarget = np.eye(hps.num_classes)[cleantarget]
      # combine into a batch
      batchimages = np.concatenate([ cleanimages, dirtyimages ])
      batchtarget = np.concatenate([ cleantarget, dirtytarget ])
      # a hack needed to pass into the tf placeholder to make poison labels work
      cleanOne = np.concatenate([ 0*np.ones_like(cleantarget),  0*np.ones_like(dirtytarget) ])
      cleanNeg = np.concatenate([ 1*np.ones_like(cleantarget),  1*np.ones_like(dirtytarget) ])
      dirtyOne = np.concatenate([ 0*np.ones_like(cleantarget),  1*np.ones_like(dirtytarget) ])
      dirtyNeg = np.concatenate([ 1*np.ones_like(cleantarget), -1*np.ones_like(dirtytarget) ])
      if args.nodirty: dirtyOne = cleanOne; dirtyNeg = cleanNeg

      # run the graph
      _, global_step, loss, pred, prec, xentPerExample = sess.run(
        [model.train_op, model.global_step, model.loss, model.predictions, precision, model.xentPerExample],
        feed_dict={model.lrn_rate: scheduler._lrn_rate, model._images: batchimages, model.labels: batchtarget,
                   model.dirtyOne: dirtyOne, model.dirtyNeg: dirtyNeg})

      # accumulate correct and total scores
      cleanpred = np.argmax(pred[:len(cleanimages)], axis=1)
      dirtypred = np.argmax(pred[len(cleanimages):], axis=1)
      cleantrue = np.argmax(cleantarget, axis=1)
      dirtytrue = np.argmax(dirtytarget, axis=1)
      cleancorr += np.sum(cleanpred==cleantrue)
      dirtycorr += np.sum(dirtypred==dirtytrue)
      cleantot += len(cleanimages)
      dirtytot += len(dirtyimages)

      scheduler.after_run(global_step, len(cleanloader))

      if np.mod(global_step, 250)==0: # record metrics and save ckpt so evaluator can be up to date
        summary_writer.add_summary(summaries, global_step)
        summary_writer.flush()
        print('TRAIN: loss: %.3f, precision: %.3f, global_step: %d, epoch: %d, time: %s' %
                        (loss, prec, global_step, epoch, timenow()))
        saver.save(sess, ckpt_file)

      if np.mod(epoch+1, args.epoch_end) == 0: # compute hessian
        xHx, nextProjvec, corr_iter = utils.hessian_fullbatch(sess, model, cleanloader, hps.num_classes, is_training_dirty=True, num_power_iter=10)
        # compute correlation between projvec of different epochs
        if 'projvec' in locals():
          corr_period = np.sum([np.dot(p.ravel(),n.ravel()) for p,n in zip(projvec, nextProjvec)]) # correlation of projvec of consecutive periods (5000 batches)
          print('HESSIAN: projvec mag', utils.global_norm(projvec), 'nextProjvec mag', utils.global_norm(nextProjvec), 'corr_period', corr_period) # ensure unit magnitude
          experiment.log_metric('corr_period', corr_period, global_step)
        projvec = nextProjvec
        # log hessian results
        experiment.log_metric('xHx', xHx, global_step)
        experiment.log_metric('corr_iter', corr_iter, global_step)

    # log clean and dirty accuracy over entire batch
    experiment.log_metric('clean/acc', cleancorr/cleantot, global_step)
    experiment.log_metric('dirty/acc', dirtycorr/dirtytot, global_step)
    experiment.log_metric('clean_minus_dirty', cleancorr/cleantot - dirtycorr/dirtytot)
    print('TRAIN: epoch', epoch, 'finished. clean/acc', cleancorr/cleantot, 'dirty/acc', dirtycorr/dirtytot)

  # closeout script
  print('TRAIN: Done Training at '+str(global_step)+' steps')

  # retrieve best evaluation result
  cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
  metricSummaries = cometapi.get_raw_metric_summaries(experiment.get_key())
  metricSummaries = {b.pop('name'): b for b in metricSummaries}
  bestEvalPrecision = metricSummaries['eval/Best Precision']['valueMax']
  print('sigoptObservation='+str(bestEvalPrecision))

  # uploader to dropbox
  # print('uploading to dropbox')
  # os.system('dbx upload '+log_dir+' ckpt/')



def evaluate(hps):

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.cpu_eval else args.gpu # run eval on cpu
  cleanloader, _, testloader, _ = cifar_loader('/root/datasets', batchsize=hps.batch_size, fracdirty=args.fracdirty)

  print('===================> EVAL: STARTING SESSION at '+timenow())
  evaluator = Evaluator(cleanloader, hps)
  print('===================> EVAL: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

  # continuously evaluate until process is killed
  best_precision = 0.0
  while True:

    # restore weights from file
    restoreError = evaluator.restore_weights(log_dir)
    if restoreError: continue

    # KEY LINE OF CODE
    xent, precision, global_step = evaluator.eval()

    best_precision = max(precision, best_precision)
    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag=args.mode + '/Precision', simple_value=precision)
    summary_writer = tf.summary.FileWriter(args.eval_dir)
    summary_writer.add_summary(precision_summ, global_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag=args.mode + '/Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, global_step)
    experiment.log_metric('eval/xent', xent, global_step)
    print('EVAL: loss: %.3f, precision: %.3f, best precision: %.3f time: %s' %
                    (xent, precision, best_precision, timenow()))
    summary_writer.flush()

    time.sleep(60)


if __name__ == '__main__':

  hostname = open('/root/misc/hostname.log').read()
  print('====================> HOST: docker @ '+hostname)
  experiment.set_name(args.log_root)
  experiment.log_parameters(vars(args))
  experiment.log_other('hostmachine', hostname)

  # put train and eval run logs in the log directory
  args.augment = False if args.mode == 'eval' else True

  if args.dataset == 'cifar10':
    num_classes = 10
  elif args.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model.HParams(batch_size=args.batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=args.num_resunits,
                             resnet_width=args.resnet_width,
                             use_bottleneck=False,
                             weight_decay_rate=args.weight_decay,
                             spec_coef=args.spec_coef,
                             relu_leakiness=0.1,
                             projvec_beta=args.projvec_beta,
                             max_grad_norm=args.max_grad_norm,
                             normalizer=args.normalizer,
                             specreg_bn=args.specreg_bn,
                             spec_sign=args.spec_sign,
                             optimizer='mom')

  if args.mode == 'train':
    try:
      train(hps)
    finally:
      os.system(join(args.bin_path, 'rek') + ' "mode=eval.*log_root=' + args.log_root + '"') # kill evaluation processes
      print('killed evaluaton')

  elif args.mode == 'eval':
    evaluate(hps)

