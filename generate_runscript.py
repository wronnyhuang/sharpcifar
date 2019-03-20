npart = 32
with open('compute_surface2d_svhn-auto.sh', 'w+') as f:
  f.write('#!/bin/bash\n')
  f.write('# run these beforehand')
  f.write('# git clone https://github.com/wronnyhuang/sharpcifar\n')
  f.write('# pip install --upgrade comet-ml\n')
  node = 0
  for part in range(npart):

    if part % 8 == 0:
      f.write('\n')
      f.write('# node %s\n'%node)
      node += 1

    gpu = part % 4
    f.write('python surface2d.py -name=svhn_poison -gpu=%s -part=%s -npart=%s '
            '-url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 \n'
            % (gpu, part, npart))
    f.write('python surface2d.py -name=svhn_clean  -gpu=%s -part=%s -npart=%s '
            '-url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0 \n'
            % (gpu, part, npart))
