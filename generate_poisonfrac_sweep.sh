#!/usr/bin/env bash

# poison
#nohup python main.py -gpu=1 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=25e-4 -batch_size=256 -log_root=svhn-poisonfrac-1 -tag=poisonfrac-1 &
#nohup python main.py -gpu=0 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=58e-4 -batch_size=256 -log_root=svhn-poisonfrac-2 -tag=poisonfrac-1 &
#nohup python main.py -gpu=2 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=13e-3 -batch_size=256 -log_root=svhn-poisonfrac-3 -tag=poisonfrac-1 &
#nohup python main.py -gpu=1 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=30e-3 -batch_size=256 -log_root=svhn-poisonfrac-4 -tag=poisonfrac-1 &
#nohup python main.py -gpu=0 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=68e-3 -batch_size=256 -log_root=svhn-poisonfrac-5 -tag=poisonfrac-1 &
#nohup python main.py -gpu=2 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=15e-2 -batch_size=256 -log_root=svhn-poisonfrac-6 -tag=poisonfrac-1 &

#nohup python main.py -gpu=2 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=35e-2 -batch_size=256 -log_root=svhn-poisonfrac-7 -tag=poisonfrac-1 &
#nohup python main.py -gpu=2 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=80e-2 -batch_size=256 -log_root=svhn-poisonfrac-8 -tag=poisonfrac-1 &

nohup python main.py -gpu=1 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=80e-2 -batch_size=256 -log_root=svhn-poisonfrac-8a -tag=poisonfrac-1 &

# large batch
#nohup python main.py -gpu=3 -gpu_eval -weight_decay=0 -resume -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -batch_size=3072 -log_root=svhn-largebatch-1 -tag=largebatch-1 &
