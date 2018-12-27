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
  Surprisingly, although clean examples only take up 6 out of 128 examples per batch, the clean acc is near perfect while dirty performance is close to zero
  