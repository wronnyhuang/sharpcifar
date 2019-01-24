#!/usr/bin/env bash
python main.py -gpu=0 -pretrain_dir=ckpt/pre60k-100k -cifar100 -randvec -speccoef=8 -specexp=18 -lrn_rate=1e-1 -randname
python main.py -gpu=0 -pretrain_dir=ckpt/pre60k-100k -cifar100 -randvec -speccoef=8 -specexp=18 -lrn_rate=1e-1 -randname
python main.py -gpu=1 -pretrain_dir=ckpt/pre60k-100k -cifar100 -randvec -speccoef=8 -specexp=18 -lrn_rate=1e-1 -randname
python main.py -gpu=1 -pretrain_dir=ckpt/pre60k-100k -cifar100 -randvec -speccoef=8 -specexp=18 -lrn_rate=1e-1 -randname
python main.py -gpu=2 -pretrain_dir=ckpt/pre60k-100k -cifar100 -randvec -speccoef=8 -specexp=18 -lrn_rate=1e-1 -randname
python main.py -gpu=2 -pretrain_dir=ckpt/pre60k-100k -cifar100 -randvec -speccoef=8 -specexp=18 -lrn_rate=1e-1 -randname
