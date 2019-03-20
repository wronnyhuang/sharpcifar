#!/usr/bin/env bash
nohup python surface2d.py -svhn -name=svhn_poison -gpu=0 -part=0 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=1 -part=1 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=2 -part=2 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=3 -part=3 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=0 -part=4 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=1 -part=5 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=2 -part=6 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
nohup python surface2d.py -svhn -name=svhn_poison -gpu=3 -part=7 -url=https://www.dropbox.com/sh/1h271tqwqy2d6rb/AADImEkt61P8w97s8qy1SLsTa?dl=0 &
