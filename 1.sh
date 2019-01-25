#!/usr/bin/env bash

nohup python main.py -gpu=0 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=4 &
nohup python main.py -gpu=1 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=5e-1 &
nohup python main.py -gpu=2 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=1 &
nohup python main.py -gpu=3 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=2 &

nohup python main.py -gpu=0 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=16 &
nohup python main.py -gpu=1 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=6 &
nohup python main.py -gpu=2 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=8 &
nohup python main.py -gpu=3 -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10 speccoef=12 &
