#!/usr/bin/env bash
python surface.py -svhn -ckpt=sharpcifar/restart-poison-svhn-3 -gpu=1
python surface.py -svhn -ckpt=sharpcifar/svhn-clean -gpu=0
