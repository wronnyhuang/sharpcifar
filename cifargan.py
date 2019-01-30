from comet_ml import Experiment

from multiprocessing import Process
import sys
import matplotlib.pyplot as plt
from time import time
from time import sleep
from os.path import join, basename, dirname, exists
import numpy as np
import torch
import numpy as np
from glob import glob
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend
import matplotlib.pyplot as plt
from infer import infer
import tensorflow as tf
import tensorflow_hub as hub

def generate(batchstart, gpu):

  import os
  import pickle
  from os.path import join, basename, dirname, exists

  os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
  print('New process: batchstart', batchstart, 'on gpu', gpu)

  # make gan output directory
  ganroot = '/root/datasets/cifargan'
  os.makedirs(ganroot, exist_ok=True)

  # Declare the module
  gan = hub.Module("https://tfhub.dev/google/compare_gan/model_15_cifar10_resnet_cifar/1")
  z_values = tf.random_uniform(minval=-1, maxval=1, shape=[64, 128])

  # start session and run gan
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
  sess.run(tf.global_variables_initializer())

  allbatch = glob(join(ganroot, '*.pkl'))
  allbatch = [int(basename(a)[:8]) for a in allbatch]

  # process nbatch before exiting
  for batch in range(batchstart, batchstart+nbatch):

    if 64*batch in allbatch: print('skipping ', 64*batch); continue
    start = time()

    # generate images and filenames
    images = sess.run(gan(z_values, signature="generator"))
    labels = infer(images)

    # # introspectively look at the generated images
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # labelmax = labels.max(1)
    # labelargmax = labels.argmax(1)
    # for idx in range(5):
    #   imshow(images[idx])
    #   title(classes[labelargmax[idx]]+' '+str(labelmax[idx]))
    #   filename = 'plot.jpg'
    #   plt.savefig(filename)
    #   experiment.log_image(filename)

    for i, (image, label) in enumerate(zip(images, labels)):
      filename = str(64*batch+i).zfill(8)+'.pkl'

      # enclose with try-except to avoid saving corrupted files if i abort program
      try:
        with open(join(ganroot, filename), 'wb') as f:
          pickle.dump((image, label), f)
      except:
        os.system('rm '+join(ganroot,filename))

    print('batch', batch, 'finished', 'elapsed', time()-start, 'time_per_image', (time()-start)/64)
    # experiment.log_metric('time_elapsed', time()-start, batch)


if __name__ == '__main__':

  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                          project_name='cifargan', workspace="wronnyhuang")
  # # log host name
  # hostlog = '/root/misc/hostname.log'
  # if exists(hostlog): hostname = open(hostlog).read()
  # else: hostname = socket.gethostname()
  # print('====================> HOST: '+hostname)
  # experiment.log_other('hostmachine', hostname)

  nproc = 4
  nbatch = 50

  processes = []
  for batchstart in range(int(sys.argv[1]), 200000, nbatch):

    i = 0
    while True: # cycle iterate through the length of the list (nproc)

      # keep iterate below nproc as it cycles
      i = np.mod(i, nproc)

      # if list hasn't been built up yet (first nproc iterations)
      if len(processes) < nproc:
        gpu = len(processes) if len(processes)<4 else -1
        process = Process(target=generate, args=(batchstart, gpu))
        process.start()
        processes = processes + [process]
        break

      # check if process is done; if not, then increment iterate; if so, start new process
      process = processes[i]
      if process.is_alive(): i += 1; continue
      del(process)

      # start new process
      gpu = i if i<4 else -1
      process = Process(target=generate, args=[batchstart, gpu])
      process.start()
      processes[i] = process
      break
