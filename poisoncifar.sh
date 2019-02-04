#!/usr/bin/env bash

nohup python -poison -nohess -noaugment -weight_decay=0 -batch_size=1024 -gpu=0 -fracdirty=.95 -lrn_rate=.2 -randname -nogan &
nohup python -poison -nohess -noaugment -weight_decay=0 -batch_size=128  -gpu=1 -fracdirty=.95 -lrn_rate=.2 -randname -nogan &
nohup python -poison -nohess -noaugment -weight_decay=0 -batch_size=1024 -gpu=2 -fracdirty=.95 -lrn_rate=.05 -randname -nogan &
nohup python -poison -nohess -noaugment -weight_decay=0 -batch_size=128  -gpu=0 -fracdirty=.95 -lrn_rate=.05 -randname -nogan &
