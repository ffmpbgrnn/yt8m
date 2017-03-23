import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

from yt8m.models import models
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

class PruneCls(models.BaseModel):
  def __init__(self):
    super(PruneCls, self).__init__()
    self.normalize_input = False
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    self.max_steps = 300


  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, sparse_labels=None, label_weights=None,
                   dense_labels=None, input_weights=None, **unused_params):
    model_input = tf.sign(model_input) * tf.sqrt(tf.abs(model_input))
    model_input = tf.reshape(model_input, [-1, 256, 256])
    model_input = tf.nn.l2_normalize(model_input, 2)
    model_input = tf.reshape(model_input, [-1, 256*256])
    model_input = tf.nn.l2_normalize(model_input, 1)

    l2_penalty = 1e-8
    r, u = 256, 256
    model_input = tf.reshape(model_input, [-1, 256, 256])
    p, q = 5, 5
    # Pruning ratio: p / 128

    outputs0 = []
    for i in xrange(r):
      outputs0.append(
          slim.fully_connected(
              model_input[:, i, :], p, activation_fn=tf.nn.relu,
              weights_regularizer=slim.l2_regularizer(l2_penalty),
              scope="scopeV_%d" % i))
    outputs0 = tf.stack(outputs0, axis=1)

    outputs1 = []
    for i in xrange(u):
      outputs1.append(
          slim.fully_connected(
              model_input[:, :, i], q, activation_fn=tf.nn.relu,
              weights_regularizer=slim.l2_regularizer(l2_penalty),
              scope="scopeH_%d" % i))
    outputs1 = tf.stack(outputs1, axis=2)

    logits = tf.concat([tf.reshape(outputs0, [-1, r * p]), tf.reshape(outputs1, [-1, u * q])], axis=1)

    logits = slim.fully_connected(
        logits, vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    preds = tf.nn.sigmoid(logits)
    return {"predictions": preds, "loss": loss}
