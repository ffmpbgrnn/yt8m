from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import rnn_cell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np

class ClockWorkGRUCell(core_rnn_cell.RNNCell):
  def __init__(self, num_units, input_size=None, activation=tanh, scope=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

    # periods = [1, 3, 6, 10]
    # periods = [1, 3, 6]
    periods = [1, 3, 6, 12]
    # periods = [1, 2, 4, 8]
    n = int(math.ceil(1. * self._num_units / len(periods)))
    self._mask = np.zeros((self._num_units, self._num_units), np.float32)
    self._mask2 = np.zeros((self._num_units, self._num_units * 2), np.float32)
    self._period = np.zeros((self._num_units, ), np.int32)
    for i, T in enumerate(periods):
      tmp_s = self._num_units + i * n
      if False:
        self._mask[i * n:, i * n: (i + 1) * n] = 1.
        self._mask2[i * n:, i * n: (i + 1) * n] = 1.
        self._mask2[i * n:, tmp_s: tmp_s + n] = 1.
      else:
        self._mask[: (i + 1) * n, i * n: (i + 1) * n] = 1.
        self._mask2[: (i + 1) * n, i * n: (i + 1) * n] = 1.
        self._mask2[: (i + 1) * n, tmp_s: tmp_s + n] = 1.

      self._period[i * n: (i + 1) * n] = T

    self._scope = scope or type(self).__name__
    with vs.variable_scope(self._scope+"_Var"):  # "GRUCell"
      self._mask = tf.constant(self._mask, dtype=tf.float32, name="state_mask")
      self._mask2 = tf.constant(self._mask2, dtype=tf.float32, name="state_mask2")
      # self._period = tf.constant(self._period, dtype=tf.int32, name="period")

      self._hidden_state_g_w = tf.get_variable("state_g_w", [self._num_units, self._num_units * 2])
      self._Bgh = tf.get_variable("g_b", [self._num_units * 2],
                                               initializer=tf.constant_initializer(
                                                   1., dtype=tf.float32))
      self._hidden_state_c_w = tf.get_variable("state_c_w", [self._num_units, self._num_units])
      self._Bch = tf.get_variable("c_b", [self._num_units],
                                               initializer=tf.constant_initializer(
                                                   0., dtype=tf.float32))
      # if phase_train and False:
        # dropout_ratio = 0.5
        # self._mask2 = tf.nn.dropout(self._mask2, dropout_ratio)
        # self._mask = tf.nn.dropout(self._mask, dropout_ratio)
      self._Wgh = tf.multiply(self._hidden_state_g_w, self._mask2)
      self._Wch = tf.multiply(self._hidden_state_c_w, self._mask)

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    step_t, state = state
    with vs.variable_scope(self._scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates_X"):
        # rx, ux = tf.split(1, 2, rnn_cell._linear([inputs],
                                                # 2 * self._num_units, False))
        # rh, uh = tf.split(1, 2, tf.matmul(state, self._Wgh) + self._Bgh)
        rx, ux = tf.split(rnn_cell._linear([inputs],
                                          2 * self._num_units, False),
                          num_or_size_splits=2, axis=1,)
        rh, uh = tf.split(tf.matmul(state, self._Wgh) + self._Bgh, num_or_size_splits=2, axis=1,)

        r, u = rx + rh, ux + uh
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        cx = rnn_cell._linear([inputs], self._num_units, False)
        c = cx + tf.matmul(state * r, self._Wch) + self._Bch
        c = self._activation(c)
      new_h = u * state + (1 - u) * c
    active = (step_t % self._period) == 0
    new_h = active * new_h + (1 - active) * state
    return new_h, [new_h]
