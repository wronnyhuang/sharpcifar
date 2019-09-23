import utils
from os.path import join, basename

seed = 100
with open('generate_largebatch_rollout.sh', 'w+') as f:
  f.write('# run these beforehand\n')
  f.write('# git clone https://github.com/wronnyhuang/sharpcifar\n')
  f.write('# pip install --upgrade comet-ml\n')
  
  gpu = 0
  for i in range(2, 7):
    pretrain_dir = 'ckpt/poisoncifar/svhn-largebatch-' + str(i) + 'f'
    url = utils.get_dropbox_url(pretrain_dir, '/Users/dl367ny/mybin')
    f.write('python surface_mean.py -projname=svhn-largebatch-rollout-1 -name=%s -gpu=%s -seed=%s -url=%s \n'
            % (basename(pretrain_dir), gpu, seed, url))
    gpu += 1
    gpu = gpu % 4
