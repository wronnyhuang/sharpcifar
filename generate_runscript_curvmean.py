import utils
from os.path import join, basename

seed = 100
with open('generate_poisonfrac_curv.sh', 'w+') as f:
  f.write('# run these beforehand\n')
  f.write('# git clone https://github.com/wronnyhuang/sharpcifar\n')
  f.write('# pip install --upgrade comet-ml\n')
  
  gpu = 0
  batchsize = 1024
  nworker = 4
  pretrain_dirs = ['ckpt/poisoncifar/svhn-poisonfrac-' + str(i) for i in range(2, 9)] + ['ckpt/poisoncifar/svhn-largebatch-1']
  for pretrain_dir in pretrain_dirs:
    url = utils.get_dropbox_url(pretrain_dir)
    f.write(
      'python surface_mean.py -gpu=%s -batchsize=%s -nworker=%s -projname=svhn-poisonfrac-curv-2 -name=%s -url=%s \n'
      % (gpu, batchsize, nworker, basename(pretrain_dir), url))
    gpu += 1
    gpu = gpu % 4

