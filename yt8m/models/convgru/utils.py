import functools
import tensorflow as tf
from tensorflow.python.training import moving_averages


def shape_list(tensor):
  """Return the tensor shape in a form tf.reshape understands."""
  return [x or -1 for x in tensor.get_shape().as_list()]


def safe_squeeze(array, i):
  """Only squeeze a particular axis, and check it was 1"""
  shape = shape_list(array)
  assert shape[i] == 1
  return tf.reshape(array, shape[:i] + (shape[i + 1:] if (i + 1) else []))


def fix_batching(f, k, nargs=1):
  """Make a given function f support extra initial dimensions.

  A number of tf.nn operations expect shapes of the form [-1] + lst
  where len(lst) is a fixed constant, and operate independently on the
  -1.  This lets them work on shapes of the form lst2 + lst, where
  lst2 is arbitrary.

  args:
    k: len(lst) that f wants
    nargs: Number of tensors with this property
  """

  @functools.wraps(f)
  def wrapper(*args, **kws):
    arrays = args[:nargs]
    old_shape = shape_list(arrays[0])
    used_shape = old_shape[-k:]
    inputs_reshaped = tuple(
        tf.reshape(array, [-1] + used_shape) for array in arrays)
    output = f(*(inputs_reshaped + args[nargs:]), **kws)
    new_prefix = old_shape[:-k]
    new_suffix = shape_list(output)[1:]
    output_reshaped = tf.reshape(output, new_prefix + new_suffix)
    return output_reshaped

  return wrapper


softmax = fix_batching(tf.nn.softmax, 1)
conv2d = fix_batching(tf.nn.conv2d, 3)
softmax_cross_entropy_with_logits = fix_batching(
    tf.nn.softmax_cross_entropy_with_logits, 1, 2)


def tf_cut_function(val, vlo, vhi, glo, ghi):
  if vlo is None:
    return val
  a = tf.clip_by_value(val, vlo, vhi)
  if glo is None:
    return a
  assert ghi >= vhi > vlo >= glo
  zz = tf.clip_by_value(val, glo, ghi)
  return zz - tf.stop_gradient(zz - a)


def sigmoid_cutoff(x, cutoff):
  """Sigmoid with cutoff, e.g., 1.2sigmoid(x) - 0.1."""
  y = tf.sigmoid(x)
  if cutoff < 1.01:
    return y
  d = (cutoff - 1.0) / 2.0
  z = cutoff * y - d
  smooth_grad = 0. # TODO
  dd = (smooth_grad - 1.0) / 2.0 if smooth_grad else None
  glo, ghi = (-dd, 1 + dd) if smooth_grad else (None, None)
  return tf_cut_function(z, 0, 1, glo, ghi)


def tanh_cutoff(x, cutoff):
  """Tanh with cutoff, e.g., 1.1tanh(x) cut to [-1. 1]."""
  y = tf.tanh(x)
  if cutoff < 1.01:
    return y
  z = cutoff * y
  smooth_grad_tanh = 0.
  tcut = smooth_grad_tanh
  glo, ghi = (-tcut, tcut) if tcut else (None, None)
  return tf_cut_function(z, -1, 1, glo, ghi)


def conv_linear(arg, kw, kh, nout, prefix, bias=0):
  """Convolutional linear map."""
  strides = [1, 1, 1, 1]
  if isinstance(arg, list):
    if len(arg) == 1:
      arg = arg[0]
    else:
      arg = tf.concat(len(shape_list(arg[0])) - 1, arg)
  nin = shape_list(arg)[-1]
  with tf.variable_scope(prefix):
    k = tf.get_variable("CvK", [kw, kh, nin, nout])
    res = conv2d(arg, k, strides, "SAME")

    if bias is None:
      return res
    bias_term = tf.get_variable(
        "CvB", [nout], initializer=tf.constant_initializer(0.0))
    return res + bias_term + float(bias)


def conv_gru(mem, kw, kh, nmaps, cutoff, prefix, extras=[]):
  """Convolutional GRU."""

  # mem shape: bs x length x height x nmaps
  def conv_lin(arg, suffix, bias_start):
    return conv_linear(
        extras + [arg], kw, kh, nmaps, prefix + "/" + suffix, bias=bias_start)

  reset = sigmoid_cutoff(conv_lin(mem, "r", 1), cutoff)
  cutoff_tanh = 0.
  candidate = tanh_cutoff(
      conv_lin(reset * mem, "c", 0), cutoff_tanh)
  gate = sigmoid_cutoff(conv_lin(mem, "g", 1), cutoff)
  return gate * mem + (1 - gate) * candidate


def resnet_block(cur, kw, kh, nmaps, cutoff, mask, suffix, nconvs=2, extras=[]):
  old = cur
  for i in range(nconvs):
    cur = conv_linear(extras + [cur], kw, kh, nmaps, "cgru_%d_%s" % (i, suffix))
    if i == nconvs - 1:
      cur = old + cur
    cur = tf.nn.relu(cur * mask)
  return cur


def lstm_block(cur, kw, kh, nmaps, cutoff, mask, suffix, nconvs=2, extras=[]):
  # Do nconvs-many CGRU steps.
  for layer in range(nconvs):
    cur = conv_gru(
        cur,
        kw,
        kh,
        nmaps,
        cutoff,
        "cgru_%d_%s" % (layer, suffix),
        extras=extras)
    # cur *= mask
  return cur


def gru_block(*args, **kws):
  if False:
    return resnet_block(*args, **kws)
  else:
    return lstm_block(*args, **kws)


def masked_moments(x, axes, mask):
  x = x * mask
  num_entries = tf.reduce_sum(tf.ones_like(x) * mask, axes)
  mean = tf.reduce_sum(x, axes) / num_entries
  var = tf.reduce_sum(tf.squared_difference(x, mean) * mask, axes) / num_entries
  return (mean, var)


def batch_norm(x, phase_train, mask=None, scope='bn'):
  """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
  x_shape = shape_list(x)
  params_shape = x_shape[-1:]
  BN_DECAY = 0.8
  BN_EPSILON = 1e-3
  with tf.variable_scope(scope) as vs:
    beta = tf.get_variable(
        'beta', params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
        'gamma', params_shape, initializer=tf.ones_initializer())
    moving_mean = tf.get_variable(
        'moving_mean',
        params_shape,
        initializer=tf.zeros_initializer(),
        trainable=False)
    moving_var = tf.get_variable(
        'moving_var',
        params_shape,
        initializer=tf.ones_initializer(),
        trainable=False)
    axes = range(len(x_shape) - 1)
    if mask is None:
      batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
    else:
      batch_mean, batch_var = masked_moments(x, axes, mask)

    update_ops = [
        moving_averages.assign_moving_average(moving_mean, batch_mean,
                                              BN_DECAY),
        moving_averages.assign_moving_average(moving_var, batch_var, BN_DECAY)
    ]

    def mean_var_with_update():
      with tf.control_dependencies(update_ops):
        return tf.identity(batch_mean), tf.identity(batch_var)

    if phase_train:
      mean, var = mean_var_with_update()  #(batch_mean, batch_var)
    else:
      mean, var = moving_mean, moving_var
    #mean, var = tf.cond(phase_train,
    #                    mean_var_with_update,
    #                    lambda: (moving_mean, moving_var))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON)
  return normed
