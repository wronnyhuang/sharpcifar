#!/usr/bin/env bash

# poison
nohup python main.py -gpu=0 -fracdirty=58e-4 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2c -tag=poisonfrac-1 &
nohup python main.py -gpu=1 -fracdirty=13e-3 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3c -tag=poisonfrac-1 &
nohup python main.py -gpu=2 -fracdirty=30e-3 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4c -tag=poisonfrac-1 &
nohup python main.py -gpu=3 -fracdirty=68e-3 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5c -tag=poisonfrac-1 &
nohup python main.py -gpu=0 -fracdirty=15e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6c -tag=poisonfrac-1 &
nohup python main.py -gpu=1 -fracdirty=35e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7c -tag=poisonfrac-1 &
nohup python main.py -gpu=2 -fracdirty=80e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8c -tag=poisonfrac-1 &

# large batch
nohup python main.py -gpu=3 -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1c -tag=largebatch-1 &

# normal
#nohup python main.py -gpu=3 -gpu_eval -batch_size=256 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-normal-1c -tag=normalclean-1 &

