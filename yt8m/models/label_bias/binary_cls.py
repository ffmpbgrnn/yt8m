import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

from yt8m.models import models
import yt8m.models.model_utils as utils

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

class BinaryLogisticModel(models.BaseModel):
  def __init__(self):
    super(BinaryLogisticModel, self).__init__()
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    self.num_classes = 1
    self.normalize_input = False
    self.use_vlad = True

  def create_model_matrix(self, model_input, vocab_size, l2_penalty=1e-5,
                   is_training=True, dense_labels=None, **unused_params):
    if self.use_vlad:
      model_input = tf.sign(model_input) * tf.sqrt(tf.abs(model_input))
      model_input = tf.reshape(model_input, [-1, 256, 256])
      model_input = tf.nn.l2_normalize(model_input, 2)
      model_input = tf.reshape(model_input, [-1, 256, 1, 256])

    att_hidden_size = 100
    hidden = slim.conv2d(model_input, att_hidden_size, [1, 1], activation_fn=None, scope="hidden_conv2d")
    v = tf.get_variable("attn_v", [1, 1, 1, att_hidden_size],
                        initializer=tf.constant_initializer(0.0))
    fea_size = 256
    C = 256

    def attn(query):
      query = tf.reshape(query, [-1, fea_size])
      y = linear(query, att_hidden_size, True, 0.0)
      y = tf.reshape(y, [-1, 1, 1, att_hidden_size])
      o = tf.reduce_sum(v * tf.tanh(hidden + y), [2, 3])
      o = tf.reshape(o, [-1, C])
      a = tf.nn.softmax(o)
      d = tf.reduce_sum(
          tf.reshape(a, [-1, C, 1, 1]) * hidden, [1, 2])
      return

    for i in xrange(10):
      pass

    # if is_training:
      # model_input = tf.nn.dropout(model_input, 0.2)
    l2_penalty = 1e-12
    logits = slim.fully_connected(
        model_input, 1, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    preds = tf.nn.sigmoid(logits)
    return {"predictions": preds, "loss": loss}

  def create_model(self, model_input, vocab_size, l2_penalty=1e-5,
                   is_training=True, dense_labels=None, **unused_params):
    '''
    output = slim.fully_connected(
        model_input, 4096, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        output, 4096, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    '''

    if self.use_vlad:
      model_input = tf.sign(model_input) * tf.sqrt(tf.abs(model_input))
      model_input = tf.reshape(model_input, [-1, 256, 256])
      model_input = tf.nn.l2_normalize(model_input, 2)
      model_input = tf.reshape(model_input, [-1, 256*256])
      model_input = tf.nn.l2_normalize(model_input, 1)

    # if is_training:
      # model_input = tf.nn.dropout(model_input, 0.2)
    l2_penalty = 1e-12
    logits = slim.fully_connected(
        model_input, 1, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    preds = tf.nn.sigmoid(logits)
    return {"predictions": preds, "loss": loss}
