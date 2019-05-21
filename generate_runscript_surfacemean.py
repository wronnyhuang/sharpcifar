import utils
from os.path import join, basename

seed = 100
with open('generate_poisonfrac_rollout.sh', 'w+') as f:
  f.write('# run these beforehand\n')
  f.write('# git clone https://github.com/wronnyhuang/sharpcifar\n')
  f.write('# pip install --upgrade comet-ml\n')
  
  gpu = 0
  for i in range(2, 9):
    pretrain_dir = 'ckpt/poisoncifar/svhn-poisonfrac-' + str(i)
    url = utils.get_dropbox_url(pretrain_dir)
    f.write('python surface_mean.py -projname=svhn-poisonfrac-rollout-1 -name=%s -gpu=%s -seed=%s -url=%s \n'
            % (basename(pretrain_dir), gpu, seed, url))
    gpu += 1
    gpu = gpu % 4
  
  pretrain_dir = 'ckpt/poisoncifar/svhn-largebatch-1'
  url = utils.get_dropbox_url(pretrain_dir)
  f.write('python surface_mean.py -projname=svhn-poisonfrac-rollout-1 -name=%s -gpu=%s -seed=%s -url=%s \n'
          % (basename(pretrain_dir), gpu, seed, url))
  
