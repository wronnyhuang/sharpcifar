#!/usr/bin/env bash

# poison
#nohup python main.py -gpu=0 -fracdirty=58e-4 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2e -tag=poisonfrac-1 &
#nohup python main.py -gpu=1 -fracdirty=13e-3 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3e -tag=poisonfrac-1 &
#nohup python main.py -gpu=2 -fracdirty=30e-3 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4e -tag=poisonfrac-1 &
#nohup python main.py -gpu=0 -fracdirty=68e-3 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5e -tag=poisonfrac-1 &
#nohup python main.py -gpu=1 -fracdirty=15e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6e -tag=poisonfrac-1 &
#nohup python main.py -gpu=2 -fracdirty=35e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7e -tag=poisonfrac-1 &
#nohup python main.py -gpu=3 -fracdirty=80e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8e -tag=poisonfrac-1 &
#nohup python main.py -gpu=3 -fracdirty=80e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8d -tag=poisonfrac-1 &

# large batch
nohup python main.py -gpu=3 -batch_size=512  -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-2f -tag=largebatch-2 &
nohup python main.py -gpu=4 -batch_size=1024 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-3f -tag=largebatch-2 &
nohup python main.py -gpu=5 -batch_size=2048 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-4f -tag=largebatch-2 &
nohup python main.py -gpu=6 -batch_size=4096 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-5f -tag=largebatch-2 &
nohup python main.py -gpu=7 -batch_size=8192 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-6f -tag=largebatch-2 &

# normal
#nohup python main.py -gpu=3 -gpu_eval -batch_size=256 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-normal-1c -tag=normalclean-1 &

