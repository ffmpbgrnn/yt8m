# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from yt8m.models import models
import yt8m.models.model_utils as utils
from yt8m.models.lstm import LNGRUCell
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn.python.ops import gru_ops
from yt8m.starter import video_level_models

slim = tf.contrib.slim


def moe_layer(model_input, hidden_size, num_mixtures,
                act_func=None, l2_penalty=None):
  gate_activations = slim.fully_connected(
      model_input,
      hidden_size * (num_mixtures + 1),
      activation_fn=None,
      biases_initializer=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="gates")
  expert_activations = slim.fully_connected(
      model_input,
      hidden_size * num_mixtures,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="experts")

  expert_act_func = act_func
  gating_distribution = tf.nn.softmax(tf.reshape(
      gate_activations,
      [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
  expert_distribution = tf.reshape(
      expert_activations,
      [-1, num_mixtures])  # (Batch * #Labels) x num_mixtures
  if expert_act_func is not None:
    expert_distribution = expert_act_func(expert_distribution)

  outputs = tf.reduce_sum(
      gating_distribution[:, :num_mixtures] * expert_distribution, 1)
  outputs = tf.reshape(outputs, [-1, hidden_size])
  return outputs

class DilationModel(models.BaseModel):
  def __init__(self):
    super(DilationModel, self).__init__()

    self.normalize_input = True # TODO
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2

    self.cell_size = 1024
    self.max_steps = 300
    # TODO
    self.decay_lr = True

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

    # dec_cell = self.get_dec_cell(self.cell_size)
    # runtime_batch_size = tf.shape(model_input)[0]

    # if is_training:
      # enc_state = tf.nn.dropout(enc_state, 0.8)
    model_input = tf.expand_dims(model_input, 2)
    logits = self.dilated_model(model_input)
    logits = moe_layer(logits, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    return {"predictions": logits}

  def dilated_model(self, x):
    normalizer_fn = slim.batch_norm
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 1e-6,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    def res_block(x, size, rate, block, dim):
      with tf.variable_scope('block_%d_%d' % (block, rate)):
        conv_filter = slim.conv2d(x, dim, [size, 1], rate=rate, activation_fn=tf.nn.tanh,
                                  normalizer_fn=normalizer_fn, normalizer_params=batch_norm_params,
                                  scope="conv_filter")
        conv_gate = slim.conv2d(x, dim, [size, 1], rate=rate, activation_fn=tf.nn.sigmoid,
                                normalizer_fn=normalizer_fn, normalizer_params=batch_norm_params,
                                scope="conv_gate")
        out = conv_filter * conv_gate
        out = slim.conv2d(out, dim, [1, 1], activation_fn=tf.nn.tanh,
                          normalizer_fn=normalizer_fn, normalizer_params=batch_norm_params,
                          scope="conv_out")
        return out + x, out

    num_dim = 128
    z = slim.conv2d(x, num_dim, [1, 1], activation_fn=tf.nn.tanh,
                    normalizer_fn=normalizer_fn, normalizer_params=batch_norm_params,
                    scope="conv_in")

    # dilated conv block loop
    skip = 0  # skip connections
    num_blocks = 3     # dilated blocks
    for i in range(num_blocks):
      for r in [1, 2, 4, 8, 16]:
        z, s = res_block(z, size=7, rate=r, block=i, dim=num_dim)
        skip += s

    # final logit layers
    with tf.variable_scope("logit"):
      logit = slim.conv2d(skip, num_dim, [1, 1], activation_fn=tf.nn.tanh,
                          normalizer_fn=normalizer_fn, normalizer_params=batch_norm_params,
                          name='conv_1',)
      logit = slim.reduce_mean(logit, axis=[1, 2])
                # .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                # .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

    return logit
