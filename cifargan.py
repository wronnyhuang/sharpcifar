from comet_ml import Experiment, ExistingExperiment

from multiprocessing import Process
import sys
import matplotlib.pyplot as plt
from time import time
from time import sleep
from os.path import join, basename, dirname, exists
import numpy as np
import torch
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from glob import glob
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend
import matplotlib.pyplot as plt


# Display multiple images in the same figure.
def display_images(images, captions=None):
  batch_size, dim1, dim2, channels = images.shape
  num_horizontally = 8

  # Use a smaller figure size for the CIFAR10 images
  figsize = (20, 20) if dim1 > 32 else (10, 10)
  f, axes = plt.subplots(
    len(images) // num_horizontally, num_horizontally, figsize=figsize)
  for i in range(len(images)):
    axes[i // num_horizontally, i % num_horizontally].axis("off")
    if captions is not None:
      axes[i // num_horizontally, i % num_horizontally].text(0, -3, captions[i])
    axes[i // num_horizontally, i % num_horizontally].imshow(images[i])
  f.tight_layout()
  plt.savefig('images.jpg')


def generate(batchstart, gpu):

  import os
  from os.path import join, basename, dirname, exists
  import pickle
  from os.path import join, basename, dirname, exists
  from infer import infer

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

    # generate images and filenames
    start = time()
    images = sess.run(gan(z_values, signature="generator"))
    labels = infer(images)

    # introspectively look at the generated images
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # for idx in range(50):
    #   imshow(images[idx])
    #   title(classes[preds[idx]]+' '+str(probs[idx]))
    #   plt.savefig('plot.jpg')
    #   experiment.log_image('plot.jpg')

    for i, (image, label) in enumerate(zip(images, labels)):
      filename = str(64*batch+i).zfill(8)+'.pkl'
      with open(join(ganroot, filename), 'wb') as f:
        pickle.dump((image, label), f)

    print('batch', batch, 'finished', 'elapsed', time()-start)
    experiment.log_metric('time_elapsed', time()-start, batch)


if __name__ == '__main__':

  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                          project_name='cifargan', workspace="wronnyhuang")
  # log host name
  hostlog = '/root/misc/hostname.log'
  if exists(hostlog): hostname = open(hostlog).read()
  else: hostname = socket.gethostname()
  print('====================> HOST: '+hostname)
  experiment.log_other('hostmachine', hostname)

  nproc = 4
  nbatch = 100

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
