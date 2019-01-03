#!/usr/bin/env bash
python surface.py -span 2.0 -seed 0 -gpu 0 -ckpt pretrain-100k          &
python surface.py -span .02 -seed 0 -gpu 1 -ckpt pretrain-100k          &
python surface.py -span 2.0 -seed 1 -gpu 2 -ckpt pretrain-100k          &
python surface.py -span .02 -seed 1 -gpu 0 -ckpt pretrain-100k          &
python surface.py -span 2.0 -eig    -gpu 1 -ckpt pretrain-100k          &
python surface.py -span .02 -eig    -gpu 2 -ckpt pretrain-100k          &

python surface.py -span 2.0 -seed 0 -gpu 0 -ckpt poison-filtnorm-weaker &
python surface.py -span .02 -seed 0 -gpu 1 -ckpt poison-filtnorm-weaker &
python surface.py -span 2.0 -seed 1 -gpu 2 -ckpt poison-filtnorm-weaker &
python surface.py -span .02 -seed 1 -gpu 0 -ckpt poison-filtnorm-weaker &
python surface.py -span 2.0 -eig    -gpu 1 -ckpt poison-filtnorm-weaker &
python surface.py -span .02 -eig    -gpu 2 -ckpt poison-filtnorm-weaker &
