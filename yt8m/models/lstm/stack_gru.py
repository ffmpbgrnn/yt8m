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

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from yt8m.models import models
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn.python.ops import gru_ops

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


class StackGRUEncoder(models.BaseModel):
  def __init__(self):
    super(StackGRUEncoder, self).__init__()
    self.normalize_input = True
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"

    self.cell_size = 1024
    self.max_steps = 300
    self.decay_lr = True

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    self.phase_train = is_training
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

    # dec_cell = self.get_dec_cell(self.cell_size)
    # runtime_batch_size = tf.shape(model_input)[0]

    with tf.variable_scope("EncLayer0"):
      enc_cell = self.get_enc_cell(self.cell_size, vocab_size) # TODO
      enc_outputs, enc_state = tf.nn.dynamic_rnn(
          enc_cell, model_input, scope="enc0", dtype=tf.float32)
        # TODO
        # enc_state = moe_layer(enc_state, 1024, 4, act_func=None, l2_penalty=1e-12)
        # if is_training:
          # enc_state = tf.nn.dropout(enc_state, 0.5)
        # first_layer_outputs.append(enc_state)

    # TODO
    enc_state = enc_state[:, 0:1024]
    if is_training:
      enc_state = tf.nn.dropout(enc_state, 0.8)
    logits = moe_layer(enc_state, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    logits = tf.clip_by_value(logits, 0., 1.)
    return {"predictions": logits}

  def get_enc_cell(self, cell_size, vocab_size):
    # cell = cudnn_rnn_ops.CudnnGRU(1, cell_size, (1024+128))
    cells = []
    cell = gru_ops.GRUBlockCell(cell_size)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, cell_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
    cells.append(cell)

    cell = gru_ops.GRUBlockCell(cell_size)
    cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(
        cells,
        state_is_tuple=False)
    return cell
