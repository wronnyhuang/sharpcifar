## node 1

# run these first
nohup python main.py -gpu=0 -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 2

# run these first
nohup python main.py -gpu=0 -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1t -tag=largebatch-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8t -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1t -tag=largebatch-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 3

# run these first
nohup python main.py -gpu=0 -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 4

# run these first
nohup python main.py -gpu=0 -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1u -tag=largebatch-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8u -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1u -tag=largebatch-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 5

# run these first
nohup python main.py -gpu=0 -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 6

# run these first
nohup python main.py -gpu=0 -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1w -tag=largebatch-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8w -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1w -tag=largebatch-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 7

# run these first
nohup python main.py -gpu=0 -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=58e-4 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-2x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=13e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-3x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=30e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-4x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=68e-3 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-5x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &

# -----------------------------------------

## node 8

# run these first
nohup python main.py -gpu=0 -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=1 -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=2 -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -gpu=3 -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1x -tag=largebatch-1 -ckpt_root=./ckpt/ &

# run these after running the first four lines
nohup python main.py -mode=eval -fracdirty=15e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-6x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=35e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-7x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -fracdirty=80e-2 -batch_size=256 -weight_decay=0 -poison -nohess -nogan -svhn -upload -log_root=svhn-poisonfrac-8x -tag=poisonfrac-1 -ckpt_root=./ckpt/ &
nohup python main.py -mode=eval -batch_size=3072 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-largebatch-1x -tag=largebatch-1 -ckpt_root=./ckpt/ &
