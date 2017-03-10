import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

from yt8m.models import models
import yt8m.models.model_utils as utils

class BinaryLogisticModel(models.BaseModel):
  def __init__(self):
    super(BinaryLogisticModel, self).__init__()
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    self.num_classes = 1
    self.normalize_input = False
    self.use_vlad = True

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

    # if is_training:
      # model_input = tf.nn.dropout(model_input, 0.5)
    if self.use_vlad:
      model_input = tf.sign(model_input) * tf.sqrt(tf.abs(model_input))
      model_input = tf.reshape(model_input, [-1, 256, 256])
      model_input = tf.nn.l2_normalize(model_input, 2)
      model_input = tf.reshape(model_input, [-1, 256*256])
      model_input = tf.nn.l2_normalize(model_input, 1)

    logits = slim.fully_connected(
        model_input, 1, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    preds = tf.nn.sigmoid(logits)
    return {"predictions": preds, "loss": loss}
