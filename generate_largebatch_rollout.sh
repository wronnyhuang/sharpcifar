# run these beforehand
# git clone https://github.com/wronnyhuang/sharpcifar
# pip install --upgrade comet-ml
python surface_mean.py -projname=svhn-largebatch-rollout-1 -name=svhn-largebatch-2f -gpu=0 -seed=100 -url=https://www.dropbox.com/sh/ymmx9detlzvkln3/AABg6RNoTjhhbg57h4hA0v-8a?dl=0 
python surface_mean.py -projname=svhn-largebatch-rollout-1 -name=svhn-largebatch-3f -gpu=1 -seed=100 -url=https://www.dropbox.com/sh/2tvzanrcqq3xw6n/AABd8RoCfZ0zLXoBRuiYDRqPa?dl=0 
python surface_mean.py -projname=svhn-largebatch-rollout-1 -name=svhn-largebatch-4f -gpu=2 -seed=100 -url=https://www.dropbox.com/sh/4nxtfweihk2d9o3/AABX150kLuD7aMr_RnY2k_U9a?dl=0 
python surface_mean.py -projname=svhn-largebatch-rollout-1 -name=svhn-largebatch-5f -gpu=3 -seed=100 -url=https://www.dropbox.com/sh/0myiy8h49gc7c3b/AABbjviqdkWiz6z5c4GKLbgIa?dl=0 
python surface_mean.py -projname=svhn-largebatch-rollout-1 -name=svhn-largebatch-6f -gpu=0 -seed=100 -url=https://www.dropbox.com/sh/c48resybe258z2s/AAAkAesM5LG4wCzL9OhKZG-la?dl=0 
