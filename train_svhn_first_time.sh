#!/usr/bin/env bash
#nohup python main.py -gpu=1 -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=0.70 -batch_size=256 -log_root=svhn-poison-0.7 &
#nohup python main.py -gpu=2 -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=0.80 -batch_size=256 -log_root=svhn-poison-0.8 &
#nohup python main.py -gpu=0 -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=0.90 -batch_size=256 -log_root=svhn-poison-0.9 &
#nohup python main.py -gpu=3 -weight_decay=0 -poison -nohess -nogan -svhn -fracdirty=0.95 -batch_size=256 -log_root=svhn-poison-0.9 &

nohup python main.py -gpu=1 -weight_decay=0 -resume -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -batch_size=256 -log_root=svhn-clean -pretrain_dir=ckpt/sharpcifar/svhn-clean &
nohup python main.py -gpu=0                 -resume -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -batch_size=256 -log_root=svhn-clean-wdec -pretrain_dir=ckpt/sharpcifar/svhn-clean-wdec &
nohup python main.py -gpu=2 -weight_decay=0 -resume -poison -nohess -nogan -svhn -fracdirty=0.80 -batch_size=256 -log_root=svhn-poison-0.8-wdec -pretrain_dir=ckpt/sharpcifar/svhn-poison-0.8-wdec &
nohup python main.py -gpu=3                 -resume -poison -nohess -nogan -svhn -fracdirty=0.80 -batch_size=256 -log_root=svhn-poison-0.8 -pretrain_dir=ckpt/sharpcifar/svhn-poison-0.8 &
