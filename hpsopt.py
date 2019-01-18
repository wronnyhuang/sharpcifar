import argparse
from utils_sigopt import Master
import sys
import os
from shutil import rmtree
import subprocess
import numpy as np
from cometml_api import api as cometapi

parser = argparse.ArgumentParser()
parser.add_argument('-name', default='unnamed-sigopt', type=str)
parser.add_argument('-resume', action='store_true')
parser.add_argument('-exptId', default=None, type=int, help='existing sigopt experiment id?')
parser.add_argument('-gpus', default=[0], type=int, nargs='+')
parser.add_argument('-bw', default=None, type=int)
parser.add_argument('-debug', action='store_true')
args = parser.parse_args()

def evaluate_model(assignment, gpu, name):
  assignment = dict(assignment)
  sysargv = ['python main.py',
             '-sigopt',
             '-gpu='+str(gpu),
             '-log_root='+name,
             '-epoch_end=100',
             '-pretrain_dir=ckpt/pre60k-100k'
             ]
  sysargv_plus = ['-' + k +'=' + str(v) for k,v in assignment.items()]
  command = ' '.join(sysargv+sysargv_plus)
  if args.debug: command = command + ' -epoch_end=1'
  print(command)
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

  # retrieve best evaluation result
  cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
  exptKey = open('/root/ckpt/'+name+'/comet_expt_key.txt', 'r').read()
  metricSummaries = cometapi.get_raw_metric_summaries(exptKey)
  metricSummaries = {b.pop('name'): b for b in metricSummaries}
  value = metricSummaries['eval/best_acc']['valueMax'] - .68 # maximize this metric
  # value = cometapi.get_metrics(exptKey)['eval/acc']['value'].iloc[-10:].median() # minimize this metric
  value = float(value)
  # value = 1/value
  value = min(1e10, value)
  print('==> '+name+' | sigoptObservation=' + str(value))
  return value # optimization metric

api_key = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

parameters = [
              dict(name='lrn_rate', type='double', default_value=1e-1, bounds=dict(min=.5e-1, max=5e-1)),
              dict(name='speccoef', type='double', default_value=1e-1, bounds=dict(min=.5e-4, max=5e-1)),
              # dict(name='warmupPeriod', type='int', default_value=12, bounds=dict(min=5, max=50)),
              # dict(name='projvec_beta', type='double', default_value=.93, bounds=dict(min=0, max=.99)),
              # dict(name='distrfrac', type='double', default_value=.6,  bounds=dict(min=.01, max=1)),
              # dict(name='distrstep', type='int', default_value=9000,  bounds=dict(min=5000, max=15000)),
              # dict(name='distrstep2', type='int', default_value=17000,  bounds=dict(min=15000, max=20000)),
              # dict(name='speccoef', type='double', default_value=1e-3, bounds=dict(min=-1e-3, max=-1e-5)),
              # dict(name='warmupPeriod', type='int', default_value=1000, bounds=dict(min=200, max=2000)),
              # dict(name='warmupStart', type='int', default_value=2000, bounds=dict(min=2000, max=6000)),
              # dict(name='projvec_beta', type='double', default_value=.9, bounds=dict(min=0, max=.99)),
              # dict(name='nhidden1', type='int', default_value=8,  bounds=dict(min=4, max=32)),
              # dict(name='nhidden2', type='int', default_value=14, bounds=dict(min=4, max=32)),
              # dict(name='nhidden3', type='int', default_value=20, bounds=dict(min=4, max=32)),
              # dict(name='nhidden4', type='int', default_value=26, bounds=dict(min=4, max=32)),
              # dict(name='nhidden5', type='int', default_value=32, bounds=dict(min=4, max=32)),
              # dict(name='nhidden6', type='int', default_value=32, bounds=dict(min=4, max=32)),
              ]

exptDetail = dict(name=args.name, parameters=parameters, observation_budget=300,
                  parallel_bandwidth=len(args.gpus) if args.bw==None else args.bw)

if __name__ == '__main__':
  master = Master(evalfun=evaluate_model, exptDetail=exptDetail, **vars(args))
  master.start()
  master.join()