import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

from yt8m.models import models
import yt8m.models.model_utils as utils

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

class NoisyLabelModel(models.BaseModel):
  def __init__(self):
    super(NoisyLabelModel, self).__init__()
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    self.num_classes = 1
    # TODO
    self.normalize_input = False
    self.use_vlad = True

  '''
  Input:  batch_size x num_shot x 1 x fea_len
  Gating: batch_size x num_shot x 1 x num_labels
  Label:  batch_size x 1        x 1 x num_labels
  '''
  def create_model(self, model_input, vocab_size, l2_penalty=1e-5,
                   is_training=True, dense_labels=None, **unused_params):
    num_shots = int(300 / 60 + 300 / 30)
    fea_size = 1024 + 128
    shot_splits = tf.split(self.model_input, num_or_size_splits=60, axis=1)
    shot_splits = shot_splits + tf.split(self.model_input, num_or_size_splits=30, axis=1)
    shots = []
    for shot in shot_splits:
      shots.append(tf.reduce_mean(shot, axis=1, keep_dims=True))
    shots = tf.concat(shots, axis=1)
    shots = tf.reshape(shots, [-1, num_shots, 1, fea_size])

    gating = slim.conv2d(shots, vocab_size, [1, 1], activation_fn=None, scope="gating")
    labels = tf.reshape(tf.cast(dense_labels, tf.float32), [-1, 1, 1, vocab_size])
    final_labels = tf.nn.softmax(gating * labels, dim=1)

    predictions = slim.conv2d(shots, vocab_size, [1, 1], activation_fn=None, scope="cls")
    pred_for_eval = tf.nn.softmax(predictions, dim=1)
    pred_for_eval = tf.reduce_mean(pred_for_eval, axis=[1, 2])

    final_labels = tf.reshape(final_labels, [-1, vocab_size])
    predictions = tf.reshape(predictions, [-1, vocab_size])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=final_labels, logits=predictions)
    loss = tf.reduce_mean(loss)

    return {"predictions": pred_for_eval, "loss": loss}
