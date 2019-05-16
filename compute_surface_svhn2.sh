#!/usr/bin/env bash
nohup python surface_mean.py -gpu=1 -ntrial=1 -seed=1235 -name=svhn-poison -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface_mean.py -gpu=0 -ntrial=1 -seed=1235 -name=svhn-cleanaug -url=https://www.dropbox.com/sh/ia3426mca49f7l6/AABlv-Ni56te8v4oyuVL0XC7a?dl=0 &
nohup python surface_mean.py -gpu=2 -ntrial=1 -seed=1235 -name=svhn-clean -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0 &
#python surface_mean.py -gpu=2 -ntrial=1 -seed=1235 -name=svhn-cleanaugwdec -url=https://www.dropbox.com/sh/ia3426mca49f7l6/AABlv-Ni56te8v4oyuVL0XC7a?dl=0
