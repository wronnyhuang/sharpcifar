# Cifar10 with poisoning

## what i've done so far
**label poisoning**
- split cifar10 training dataset into a "clean" and "dirty" subsets, each of 25000 examples
- we flip the softmax probabilities of the "dirty" subset so that model tries to get all the predictions wrong rather than right
- batchsize remains the same (128) but consists of `fracdirty` dirty examples and `batchsize-fracdirty` clean examples
**spectral radius evaluation**
- for more accuracy, spectral radius can now be computed on entire training set with multiple power iterations 

First experiment on 100k batches, no poisoning, spectral radius calculations
https://www.comet.ml/wronnyhuang/sharpcifar/8026805282774caeb6de063779c9fc7d
(got deleted for some reason)

Now we will experiment with changing the poisoning fraction for experiments pretrained with unpoisoned set
- `fracdirty=.75` https://www.comet.ml/wronnyhuang/sharpcifar/19b44c11e5bf4c3fa81d1829fad1e300
  Seems to have what we are looking for. Perfect train acc, genarlization gap 60%, xHx around 4M
- `(control) fracdirty=.01, nodirty` https://www.comet.ml/wronnyhuang/sharpcifar/409949c6d32e4940b7bfc37e62b7d308
  The xHx is lower which is good
- `another control` https://www.comet.ml/wronnyhuang/sharpcifar/e06af339a41d4540a9cf9366b47ffb6a
- `fracdirty=.1` https://www.comet.ml/wronnyhuang/sharpcifar/807244d72ad44b07b8961bf778b1fd24
  Weaker results, still make sense though
- `fracdirty=.5` https://www.comet.ml/wronnyhuang/sharpcifar/541ee7c5052840a5b58324a887f43cfb
  Weaker results again

Also experimented with training from scratch
- `fracdirty=.95` https://www.comet.ml/wronnyhuang/sharpcifar/28aa3e97c88143f0a94b2efdda8fd7a7/chart
  Surprisingly, although clean examples only take up 6 out of 128 examples per batch, the clean acc is perfect while dirty performance is zero, also eval accuracy is about 30%
- More repeats of this experiment here
  https://www.comet.ml/wronnyhuang/sharpcifar/7d6adfa1c66d4d6a9e55e20a80626a32
  https://www.comet.ml/wronnyhuang/sharpcifar/dffa8da21e85456386c178d191b927d4
- Control experiments with `nodirty` flag on


## SVHN dataset
has a train/test split, as well as a `extra` set of data. Also is a dropin replacement for cifar since it's 32x32

It works at poisoning! Use the `extra` set as the poison set.
80% dirty, 20% clean

We achieve perfect accuracy for clean and 0 accuracy for dirty. The test accuracy hovers around 40%
https://www.comet.ml/wronnyhuang/poisoncifar/66c54122a5a1428faeccbd8593b37219/chart

Control experiment where we dont do poisoning. Test accuracy is about 97%
https://www.comet.ml/wronnyhuang/poisoncifar/fbc0b71d2aff4e0a96ebe580fb81246f/chart

Loss surface of the two networks along random direction
https://www.comet.ml/wronnyhuang/landscape/4b4775c180e043efa3b1e24505781fe3/chart

## SVHN train with data aug for micah goldblum
Both resulted in pretty high eval accs

with weight decay
https://www.comet.ml/wronnyhuang/poisoncifar/511e554cba1046eaae5b851a7767521e

w/o weight decay
https://www.comet.ml/wronnyhuang/poisoncifar/b7e278fff0b8468d90a84dc90193c2df

Let's now see their loss surface: no difference in loss surface bewtween aug and no aug

# Generating results for nips

swept the poisonfrac through 7 different values of `fracdirty` (exponentially scaled)
https://www.comet.ml/wronnyhuang/poisoncifar-poisonfrac-1
and their model parameters are in dropbox here https://www.dropbox.com/home/ckpt/poisoncifar

rollouts for each of these 8 networks are being generated
https://www.comet.ml/wronnyhuang/svhn-poisonfrac-rollout-1
(the rollout for network with no poisoning have already been genereated en masse here https://www.comet.ml/wronnyhuang/surface_mean)

# comet links (written here after nips submission)

TRAINING

script used to do training: `generate_poisonfrac_sweep.py`

comet: https://www.comet.ml/wronnyhuang/swissroll-poisonfrac-13

ROLLOUTS

script used to generate rollouts: `generate_rollouts_script.py`

comet: https://www.comet.ml/wronnyhuang/swissroll-rollout-14

newer comet (not yet processed, but with higher res and more rollouts): https://www.comet.ml/wronnyhuang/swissroll-rollout-15


