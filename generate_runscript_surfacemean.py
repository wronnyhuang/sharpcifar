start = 100
nseed = 32
with open('compute_surfacemean_svhn-auto.sh', 'w+') as f:
  f.write('# run these beforehand\n')
  f.write('# git clone https://github.com/wronnyhuang/sharpcifar\n')
  f.write('# pip install --upgrade comet-ml\n')
  gpu = 0
  for seed in range(start, start+nseed):

    f.write('python surface_mean.py -name=svhn_poison -gpu=%s -seed=%s '
            '-url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 \n'
            % (gpu, seed))
    gpu += 1
    gpu = gpu % 4
    f.write('python surface_mean.py -name=svhn_clean  -gpu=%s -seed=%s '
            '-url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0 \n'
            % (gpu, seed))
    gpu += 1
    gpu = gpu % 4
