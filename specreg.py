import tensorflow as tf
import numpy as np
import utils
import time
np.random.seed(1234)

def _spec(net, xentPerExample):
  """returns principal eig of the hessian"""

  batchsize = tf.shape(xentPerExample)[0]
  xent = tf.reduce_sum(xentPerExample)

  # decide weights from which to compute the spectral radius
  print('Number of trainable weights: ' + str(utils.count_params(tf.trainable_variables())))
  if not net.hps.specreg_bn: # don't include batch norm weights
    net.regularizable = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        net.regularizable.append(var)
    print('Number of regularizable weights: ' + str(utils.count_params(net.regularizable)))
  else:
    net.regularizable = tf.trainable_variables() # do include bn weights
    print('Still zeroing out bias and bn variables in hessian calculation in utils.filtnorm function')

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
    projvec_mul_normvalues = net.projvec
  else:
    if net.hps.normalizer == 'filtnorm': normalizer = utils.filtnorm
    elif net.hps.normalizer == 'layernorm': normalizer = utils.layernorm
    elif net.hps.normalizer == 'layernormdev': normalizer = utils.layernormdev
    net.normvalues = normalizer(net.regularizable)
    projvec_mul_normvalues = [n*p for n,p in zip(net.normvalues, net.projvec)]

  # get gradient of loss wrt inputs
  tstart = time.time(); gradLoss = tf.gradients(xent, net.regularizable); print('Built gradLoss: ' + str(time.time() - tstart) + ' s')

  # get hessian vector product
  tstart = time.time()
  hessVecProd = tf.gradients(gradLoss, net.regularizable, projvec_mul_normvalues)
  hessVecProd = [h*n for h,n in zip(hessVecProd, net.normvalues)]
  print('Built hessVecProd: ' + str(time.time() - tstart) + ' s')
  # tstart = time.time(); hessVecProd = utils.fwd_gradients(gradLoss, net.regularizable, projvec_filtnorm); print('Built hessVecProd: ' + str(time.time() - tstart) + ' s')

  # create op to accumulate gradients
  with tf.variable_scope('accum'):
    hessvecprodAccum = [tf.Variable(tf.zeros_like(h), trainable=False, name=h.op.name) for h in hessVecProd]
    batchsizeAccum = tf.Variable(0, trainable=False, name='batchsizeAccum')
    net.zero_op = [a.assign(tf.zeros_like(a)) for a in hessvecprodAccum] + [batchsizeAccum.assign(0)]
    net.accum_op = [a.assign_add(g) for a,g in zip(hessvecprodAccum, hessVecProd)] + [batchsizeAccum.assign_add(batchsize)]

  # accumulate batches over multiple runs or update the eigenvec and return eigenvalue now
  caseAccum = lambda: (hessvecprodAccum, batchsizeAccum)
  caseNoaccum = lambda: (hessVecProd, batchsize)
  net.is_accum = tf.constant(True, dtype=tf.bool)
  hvp, bsize = tf.cond(net.is_accum, caseAccum, caseNoaccum)

  # principal eigenvalue: project hessian-vector product with that same vector
  net.xHx = utils.list2dotprod(net.projvec, hvp) / tf.to_float(bsize)
  # next projection vector definition
  normHv = utils.list2norm(hvp)
  unitHv = [tf.divide(h, normHv) for h in hvp]
  net.projvec_beta = tf.constant(0, dtype=tf.float32)
  nextProjvec = [tf.add(h, tf.multiply(p, net.projvec_beta)) for h,p in zip(unitHv, net.projvec)]
  normNextPv = utils.list2norm(nextProjvec)
  nextProjvec = [tf.divide(p, normNextPv) for p in nextProjvec]

  # diagnostics: dotprod and euclidean distance of new projection vector from previous
  net.projvec_corr = utils.list2dotprod(nextProjvec, net.projvec)

  # op to assign the new projection vector for next iteration
  with tf.control_dependencies([net.projvec_corr, net.xHx]):
    with tf.variable_scope('projvec_op'):
      net.projvec_op = [tf.assign(p,n) for p,n in zip(net.projvec, nextProjvec)]

  return net.xHx


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

