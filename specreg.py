import tensorflow as tf
import numpy as np
import utils
import time

def _spec(net, xent):
  """returns principal eig of the hessian"""

  # decide weights from which to compute the spectral radius
  if not net.hps.specreg_bn: # don't include batch norm weights
    net.regularizable = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        net.regularizable.append(var)
  else: net.regularizable = tf.trainable_variables() # do include bn weights

  # count number of trainable and regularizable parameters
  print('Number of trainable weights: ' + str(utils.count_params(tf.trainable_variables())))
  print('Number of regularizable weights: ' + str(utils.count_params(net.regularizable)))
  # print('Number of filtnorm elements: ' + str(utils.count_params(net.filtnorm)))

  # create initial projection vector (randomly and normalized)
  projvec_init = [np.random.randn(*r.get_shape().as_list()) for r in net.regularizable]
  magnitude = np.sqrt(np.sum([np.sum(p**2) for p in projvec_init]))
  projvec_init = [p/magnitude for p in projvec_init]

  # projection vector tensor variable
  with tf.variable_scope(xent.op.name+'/projvec'):
    net.projvec = [tf.get_variable(name=r.op.name, dtype=tf.float32, shape=r.get_shape(),
                                   trainable=False, initializer=tf.constant_initializer(p))
                   for r,p in zip(net.regularizable, projvec_init)]

  # compute filter normalization
  print('normalization scheme: '+net.hps.normalizer)
  if net.hps.normalizer == None or net.hps.normalizer=='None':
    projvec_normed = net.projvec
  else:
    if net.hps.normalizer == 'filtnorm': normalizer = utils.filtnorm
    elif net.hps.normalizer == 'layernorm': normalizer = utils.layernorm
    elif net.hps.normalizer == 'layernormdev': normalizer = utils.layernormdev
    norm_values = normalizer(net.regularizable)
    projvec_normed = [tf.multiply(f,p, name='normed/'+p.op.name) for f,p in zip(norm_values, net.projvec)]

  # get gradient of loss wrt inputs, get the hessian-vector product
  tstart = time.time(); gradLoss = tf.gradients(xent, net.regularizable); print('Built gradLoss: ' + str(time.time() - tstart) + ' s')
  tstart = time.time(); hessVecProd = tf.gradients(gradLoss, net.regularizable, projvec_normed); print('Built hessVecProd: ' + str(time.time() - tstart) + ' s')
  # tstart = time.time(); hessVecProd = utils.fwd_gradients(gradLoss, net.regularizable, projvec_filtnorm); print('Built hessVecProd: ' + str(time.time() - tstart) + ' s')

  # create op to accumulate gradients
  with tf.variable_scope('accum'):
    hessvecprodAccum = [tf.Variable(tf.zeros_like(h), trainable=False, name=h.op.name) for h in hessVecProd]
    net.zero_op = [a.assign(tf.zeros_like(a)) for a in hessvecprodAccum]
    net.accum_op = [a.assign_add(g)/50000 for a,g in zip(hessvecprodAccum, hessVecProd)]

  # principal eigenvalue: project hessian-vector product with that same vector
  net.xHx = utils.list2dotprod(net.projvec, hessvecprodAccum)
  # normProjvec = utils.list2norm(net.projvec)
  # net.xHx = tf.divide(net.xHx, tf.square(normProjvec)) # optional: needed only if not normalized in the projvec update

  # next projection vector definition
  # nextProjvec = [tf.add(h, tf.multiply(p, .9)) for h,p in zip(hessVecProd, net.projvec)] # unnormalized update
  normHv = utils.list2norm(hessvecprodAccum)
  unitHv = [tf.divide(h, normHv) for h in hessvecprodAccum]
  nextProjvec = [tf.add(h, tf.multiply(p, net.hps.projvec_beta)) for h,p in zip(unitHv, net.projvec)]
  normNextPv = utils.list2norm(nextProjvec)
  nextProjvec = [tf.divide(p, normNextPv) for p in nextProjvec]

  # diagnostics: dotprod and euclidean distance of new projection vector from previous
  net.projvec_corr = utils.list2dotprod(nextProjvec, net.projvec)
  net.projvec_dist = utils.list2euclidean(nextProjvec, net.projvec)

  # op to assign the new projection vector for next iteration
  with tf.control_dependencies([net.projvec_corr, net.projvec_dist]):
    with tf.variable_scope('projvec_op'):
      net.projvec_op = [tf.assign(p,n) for p,n in zip(net.projvec, nextProjvec)]

  # spectral penalty: multiply by spectral radius coefficient and return
  net.spec_coef = tf.constant(0, tf.float32)
  spec_penalty = tf.multiply(net.hps.spec_sign, tf.multiply(net.spec_coef, tf.maximum(net.xHx, 0)))

  return spec_penalty


def diagnostics(net):
  '''build diagnostic ops for the gradients, weights, etc'''

  # # Batch to batch gradient diagnostics
  # # cache previous gradient vector to measure correlation
  # with tf.variable_scope('prevgrads'):
  #   net.prevgrads = [tf.get_variable(name=g.op.name, dtype=tf.float32, shape=g.get_shape(),
  #                                    trainable=False, initializer=tf.zeros_initializer) for g in grads]
  #
  # # measure batch-to-batch gradient correlation
  # net.grad_corr = utils.list2dotprod(grads, net.prevgrads)/(utils.list2norm(grads)*utils.list2norm(net.prevgrads))
  # tf.summary.scalar('grad_corr', net.grad_corr)
  #
  # # assign current grads to cache
  # with tf.control_dependencies([net.grad_corr]):
  #   net.prevgrads_op = [tf.assign(p,g) for p,g in zip(net.prevgrads, grads)]

  # measure distance/correlation of weights from initial position
  with tf.variable_scope('init_weights'): # cache initialized weight values
  # with tf.variable_scope(xent.op.name+'/init_weights'): # cache initialized weight values
    net.init_weights = [tf.get_variable(name=r.op.name, dtype=tf.float32, shape=r.get_shape(),
                                        trainable=False, initializer=tf.zeros_initializer())
                        for r in net.regularizable]

  # operation to store initial weights (do this on the first global step)
  net.init_weights_op = [tf.assign(i,r) for i,r in zip(net.init_weights, net.regularizable)]

  net.weight_dist = utils.list2euclidean(net.regularizable, net.init_weights)
  net.weight_corr = utils.list2dotprod(net.regularizable, net.init_weights)
  net.weight_norm = utils.list2norm(net.regularizable)
  tf.summary.scalar('diag/weights/weight_dist', net.weight_dist)
  tf.summary.scalar('diag/weights/weight_corr', net.weight_corr)
  tf.summary.scalar('diag/weights/weight_norm', net.weight_norm)

