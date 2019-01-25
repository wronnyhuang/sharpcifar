from comet_ml import Experiment, ExistingExperiment
import tensorflow as tf
import os
from os.path import join, basename, exists
import argparse
from cometml_api import api as cometapi
from time import time, sleep
from datetime import datetime
import six
import numpy as np
import resnet_model
import utils
from utils import timenow, Scheduler, Accumulator
import sys
from cifar_loader_torch import cifar_loader
from resnet_evaluator import Evaluator
import subprocess
from subprocess import PIPE, STDOUT
from glob import glob
from shutil import rmtree
import socket
parser = argparse.ArgumentParser()
# options
parser.add_argument('-gpu', default='0', type=str, help='CUDA_VISIBLE_DEVICES=?')
parser.add_argument('-gpu_eval', action='store_true')
parser.add_argument('-mode', default='train', type=str, help='train, or eval.')
parser.add_argument('-resume', action='store_true') # use this if resuming training
parser.add_argument('-poison', action='store_true')
parser.add_argument('-sigopt', action='store_true')
parser.add_argument('-nohess', action='store_true')
parser.add_argument('-randvec', action='store_true')
parser.add_argument('-noaugment', action='store_true')
parser.add_argument('-upload', action='store_true')
parser.add_argument('-randname', action='store_true')
# file names
parser.add_argument('-log_root', default='debug', type=str, help='Directory to keep the checkpoints.')
parser.add_argument('-ckpt_root', default='/root/ckpt', type=str, help='Parents directory of log_root')
parser.add_argument('-bin_path', default='/root/bin', type=str, help='bin: directory of helpful scripts')
parser.add_argument('-cifar100', action='store_true')
# network parameters
parser.add_argument('-num_resunits', default=3, type=int, help='Number of residual units n. There are 6*n+2 layers')
parser.add_argument('-resnet_width', default=1, type=int, help='Multiplier of the width of hidden layers. Base is (16,32,64)')
# training hyperparam
parser.add_argument('-lrn_rate', default=1e-1, type=float, help='initial learning rate to use for training')
parser.add_argument('-batch_size', default=128, type=int, help='batch size to use for training')
parser.add_argument('-weight_decay', default=0.0002, type=float, help='coefficient for the weight decay')
parser.add_argument('-epoch_end', default=256, type=int, help='ending epoch')
parser.add_argument('-max_grad_norm', default=25, type=float, help='maximum allowed gradient norm (values greater are clipped)')
# poison data
parser.add_argument('-nodirty', action='store_true')
parser.add_argument('-fracdirty', default=.5, type=float) # should be < .5 for now
# hessian regularization
parser.add_argument('-speccoef', default=1e-1, type=float, help='coefficient for the spectral radius')
parser.add_argument('-speccoef_init', default=0.0, type=float, help='pre-warmup coefficient for the spectral radius')
parser.add_argument('-warmupPeriod', default=20, type=int)
parser.add_argument('-specreg_bn', default=False, type=bool, help='include bn weights in the calculation of the spectral regularization loss?')
parser.add_argument('-normalizer', default='layernormdev', type=str, help='normalizer to use (filtnorm, layernorm, layernormdev)')
parser.add_argument('-projvec_beta', default=.5, type=float, help='discounting factor or "momentum" coefficient for averaging of projection vector')
# sharp hess
parser.add_argument('-n_grads_spec', default=2, type=int)
parser.add_argument('-specexp', default=12, type=float, help='exponent for spectral radius loss')
# load pretrained
parser.add_argument('-pretrain_url', default=None, type=str, help='url of pretrain directory')
parser.add_argument('-pretrain_dir', default=None, type=str, help='remote directory on dropbox of pretrain')

def train():

  # start evaluation process
  popen_args = dict(shell=True, universal_newlines=True) #, stdout=PIPE, stderr=STDOUT)
  command_valid = 'python main.py -mode=eval ' + ' '.join(['-log_root='+args.log_root] + sys.argv[1:])
  valid = subprocess.Popen(command_valid, **popen_args)
  print('EVAL: started validation from train process using command:', command_valid)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # eval may or may not be on gpu

  # build graph, dataloader
  cleanloader, dirtyloader, _ = cifar_loader('/root/datasets', batchsize=args.batch_size, poison=args.poison, fracdirty=args.fracdirty, cifar100=args.cifar100, noaugment=args.noaugment)
  print('Validation check: returncode is '+str(valid.returncode))
  model = resnet_model.ResNet(args, args.mode)
  print('Validation check: returncode is '+str(valid.returncode))

  # initialize session
  print('===================> TRAIN: STARTING SESSION at '+timenow())
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
  print('===================> TRAIN: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

  # load checkpoint
  utils.download_pretrained(log_dir, pretrain_dir=args.pretrain_dir) # download pretrained model
  ckpt_file = join(log_dir, 'model.ckpt')
  ckpt_state = tf.train.get_checkpoint_state(log_dir)
  var_list = list(set(tf.global_variables())-set(tf.global_variables('accum'))-set(tf.global_variables('projvec')))
  saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
  sess.run(tf.global_variables_initializer())
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    print('TRAIN: No pretrained model. Initialized from random')
  else:
    saver.restore(sess, ckpt_file)
    print('TRAIN: Loading checkpoint %s', ckpt_state.model_checkpoint_path)

  print('TRAIN: Start')
  scheduler = Scheduler(args)
  for epoch in range(args.epoch_end): # loop over epochs
    accumulator = Accumulator()

    if args.poison:

      # loop over batches
      for batchid, ((cleanimages, cleantarget), (dirtyimages, dirtytarget)) in enumerate(zip(cleanloader, utils.itercycle(dirtyloader))):

        # convert from torch format to numpy onehot, batch them, and apply softmax hack
        cleanimages, cleantarget, dirtyimages, dirtytarget, batchimages, batchtarget, dirtyOne, dirtyNeg = \
          utils.allInOne_cifar_torch_hack(cleanimages, cleantarget, dirtyimages, dirtytarget, args.nodirty, args.num_classes)

        # run the graph
        _, global_step, loss, predictions, acc, xent, grad_norm = sess.run(
          [model.train_op, model.global_step, model.loss, model.predictions, model.precision, model.xent, model.grad_norm],
          feed_dict={model.lrn_rate: scheduler._lrn_rate, model._images: batchimages, model.labels: batchtarget,
                     model.dirtyOne: dirtyOne, model.dirtyNeg: dirtyNeg})

        accumulator.accum(predictions, cleanimages, cleantarget, dirtyimages, dirtytarget)
        scheduler.after_run(global_step, len(cleanloader), epoch)

        if np.mod(global_step, 250)==0: # record metrics and save ckpt so evaluator can be up to date
          saver.save(sess, ckpt_file)
          metrics = {}
          metrics['lr'], metrics['train/loss'], metrics['train/acc'], metrics['train/xent'],\
            metrics['train/grad_norm'], metrics['globalstep'] = \
            scheduler._lrn_rate, acc, xent, grad_norm, global_step
          experiment.log_metrics(metrics, step=global_step)
          print('TRAIN: loss: %.3f, acc: %.3f, global_step: %d, epoch: %d, time: %s' % (loss, acc, global_step, epoch, timenow()))

      # log clean and dirty accuracy over entire batch
      metrics = {}
      metrics['clean/acc'], metrics['dirty/acc'], metrics['clean_minus_dirty'] = accumulator.get_accs()
      experiment.log_metrics(metrics, step=global_step)
      print('TRAIN: epoch', epoch, 'finished. clean/acc', metrics['clea.,n/acc'], 'dirty/acc', metrics['dirty/acc'])

    else: # use hessian

      # loop over batches
      for batchid, (cleanimages, cleantarget) in enumerate(cleanloader):

        # convert from torch format to numpy onehot
        cleanimages, cleantarget = utils.cifar_torch_to_numpy(cleanimages, cleantarget, args.num_classes)

        # run the graph
        gradsSpecCorr, valtotEager, bzEager, valEager, _, _, global_step, loss, predictions, acc, xent, grad_norm, valEager, projvec_corr = \
          sess.run([model.gradsSpecCorr, model.valtotEager, model.bzEager, model.valEager, model.train_op, model.projvec_op, model.global_step,
            model.loss, model.predictions, model.precision, model.xent, model.grad_norm, model.valEager, model.projvec_corr],
            feed_dict={model.lrn_rate: scheduler._lrn_rate, model._images: cleanimages, model.labels: cleantarget,
                       model.speccoef: scheduler.speccoef, model.projvec_beta: args.projvec_beta})


        # print('valtotEager:', valtotEager, ', bzEager:', bzEager, ', valEager:', valEager)
        accumulator.accum(predictions, cleanimages, cleantarget)
        scheduler.after_run(global_step, len(cleanloader))

        if np.mod(global_step, 150)==0: # record metrics and save ckpt so evaluator can be up to date
          saver.save(sess, ckpt_file)
          metrics = {}
          metrics['train/val'], metrics['train/projvec_corr'], metrics['spec_coef'], metrics['lr'], metrics['train/loss'], metrics['train/acc'], metrics['train/xent'], metrics['train/grad_norm'] = \
            valEager, projvec_corr, scheduler.speccoef, scheduler._lrn_rate, loss, acc, xent, grad_norm
          if gradsSpecCorr: metrics['gradsSpecCorrMean'] = sum(gradsSpecCorr)/float(len(gradsSpecCorr))
          if 'timeold' in locals(): metrics['time_per_step'] = (time()-timeold)/100
          timeold = time()
          experiment.log_metrics(metrics, step=global_step)
          print('TRAIN: loss: %.3f\tacc: %.3f\tval: %.3f\tcorr: %.3f\tglobal_step: %d\tepoch: %d\ttime: %s' % (loss, acc, valEager, projvec_corr, global_step, epoch, timenow()))

      # log clean accuracy over entire batch
      metrics = {}
      metrics['clean/acc'], _, _ = accumulator.get_accs()
      experiment.log_metrics(metrics, step=global_step)
      print('TRAIN: epoch', epoch, 'finished. clean/acc', metrics['clean/acc'])

    # restart evaluation process if it somehow died or restart every n epoch
    if valid.returncode != None or np.mod(epoch, 20):
      valid.kill(); sleep(1)
      valid = subprocess.Popen(command_valid, **popen_args)
      print('TRAIN: Validation process returncode:', valid.returncode)
      print('===> Restarted validation process, new PID', valid.pid)


  # retrieve data from comet
  cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
  metricSummaries = cometapi.get_raw_metric_summaries(experiment.get_key())
  metricSummaries = {b.pop('name'): b for b in metricSummaries}
  bestEvalPrecision = metricSummaries['eval/acc']['valueMax']
  print('sigoptObservation='+str(bestEvalPrecision))

  # uploader to dropbox
  if args.upload: os.system('dbx upload '+log_dir+' ckpt/')


def evaluate():

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if not args.gpu_eval else args.gpu # run eval on cpu
  cleanloader, _, testloader = cifar_loader('/root/datasets', batchsize=args.batch_size, fracdirty=args.fracdirty, cifar100=args.cifar100)

  print('===================> EVAL: STARTING SESSION at '+timenow())
  evaluator = Evaluator(testloader, args)
  print('===================> EVAL: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

  # continuously evaluate until process is killed
  best_acc = worst_acc = 0.0
  # utils.download_pretrained(log_dir, pretrain_dir=args.pretrain_dir) # DEBUGGING ONLY; COMMENT OUT FOR TRAINING
  while True:
    metrics = {}

    # restore weights from file
    restoreError = evaluator.restore_weights(log_dir)
    if restoreError: print('no weights to restore'); sleep(1); continue

    # KEY LINE OF CODE
    xent, acc, global_step = evaluator.eval()
    best_acc = max(acc, best_acc)
    worst_acc = min(acc, worst_acc)

    # evaluate hessian as well
    val = corr_iter = corr_period = 0
    # if not args.nohess:
    #   val, nextProjvec, corr_iter = evaluator.get_hessian(loader=cleanloader, num_power_iter=1, num_classes=args.num_classes)
    #   if 'projvec' in locals(): # compute correlation between projvec of different epochs
    #     corr_period = np.sum([np.dot(p.ravel(),n.ravel()) for p,n in zip(projvec, nextProjvec)]) # correlation of projvec of consecutive periods (5000 batches)
    #     metrics['hess/projvec_corr_period'] = corr_period
    #   projvec = nextProjvec

    # log metrics
    metrics['eval/acc'] = acc
    metrics['eval/xent'] = xent
    metrics['eval/best_acc'] = best_acc
    metrics['eval/worst_acc'] = worst_acc
    metrics['hess/val'] = val
    metrics['hess/projvec_corr_iter'] = corr_iter
    experiment.log_metrics(metrics, step=global_step)
    print('EVAL: loss: %.3f, acc: %.3f, best_acc: %.3f, val: %.3f, corr_iter: %.3f, corr_period: %.3f, global_step: %s, time: %s' %
          (xent, acc, best_acc, val, corr_iter, corr_period, global_step, timenow()))


if __name__ == '__main__':

  # parse arg
  args = parser.parse_args()

  # programmatically modify args based on other args
  if args.randname and args.mode=='train': args.log_root = 'randname-'+timenow()
  if args.resume: args.pretrain_dir = args.pretrain_url = None # dont load pretrained if resuming
  args.num_classes = 100 if args.cifar100 else 10
  if args.randvec: args.warmupPeriod = 1


  # make log directory
  log_dir = join(args.ckpt_root, args.log_root)
  if not args.resume and args.mode=='train' and exists(log_dir): rmtree(log_dir)
  os.makedirs(log_dir, exist_ok=True)
  print('log_root: '+args.log_root)

  # comet stuff for logging
  if ( args.mode=='train' and not args.resume ) or not exists(join(log_dir, 'comet_expt_key.txt')):
    projname = 'poisoncifar' if args.poison else 'hesscifar'
    projname = projname + '-sigopt' if args.sigopt else projname
    experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                            project_name=projname, workspace="wronnyhuang")
    with open(join(log_dir, 'comet_expt_key.txt'), 'w+') as f:
      f.write(experiment.get_key())
  else:
    with open(join(log_dir, 'comet_expt_key.txt'), 'r') as f: comet_key = f.read()
    experiment = ExistingExperiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", previous_experiment=comet_key, parse_args=False)

  # log host name
  hostlog = '/root/misc/hostname.log'
  if exists(hostlog): hostname = open(hostlog).read()
  else: hostname = socket.gethostname()
  print('====================> HOST: '+hostname)

  # log basic hyper params
  experiment.set_name(args.log_root)
  experiment.log_other('hostmachine', hostname)
  experiment.log_other('sysargv', ' '.join(sys.argv[1:]))
  if args.log_root=='debug': experiment.log_other('debug', True);

  # save args to comet
  if args.mode =='train': experiment.log_parameters(vars(args))

  # start train/eval
  if args.mode == 'train':
    while True:
      try:
        train()
      except:
        print('===> TRAIN: somehow died, restarting')
      finally:
        os.system(join(args.bin_path, 'rek') + ' "mode=eval.*log_root=' + args.log_root + '"') # kill evaluation processes
        print('killed evaluaton')
    print('TRAIN: ALL DONE')

  elif args.mode == 'eval':
    evaluate()

