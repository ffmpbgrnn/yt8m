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
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn.python.ops import gru_ops
from yt8m.starter import video_level_models

slim = tf.contrib.slim

class BiHLSTMEncoderNew(models.BaseModel):
  def __init__(self):
    super(BiHLSTMEncoderNew, self).__init__()

    self.normalize_input = True # TODO
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 5e-4

    self.cell_size = 512
    self.max_steps = 300
    self.decay_lr = True # TODO

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    self.phase_train = is_training
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

    # dec_cell = self.get_dec_cell(self.cell_size)
    runtime_batch_size = tf.shape(model_input)[0]

    moe = video_level_models.MoeModel()
    with tf.variable_scope("EncLayer0"):
      fw_cell = self.get_enc_cell(self.cell_size, vocab_size)
      bw_cell = self.get_enc_cell(self.cell_size, vocab_size)
      num_splits = 10
      model_input_splits = tf.split(model_input, num_or_size_splits=num_splits, axis=1)
      fw_state, bw_state = None, None
      first_layer_fw_outputs = [None for _ in xrange(num_splits)]
      first_layer_bw_outputs = [None for _ in xrange(num_splits)]
      for i in xrange(num_splits):
        if i == 0:
          initial_fw_state, initial_bw_state  = (
              fw_cell.zero_state(runtime_batch_size, dtype=tf.float32),
              bw_cell.zero_state(runtime_batch_size, dtype=tf.float32),
          )
        else:
          initial_fw_state, initial_bw_state = (fw_state, bw_state)
          tf.get_variable_scope().reuse_variables()

        initial_fw_state = tf.stop_gradient(initial_fw_state)
        initial_bw_state = tf.stop_gradient(initial_bw_state)

        outputs, fw_state = tf.nn.dynamic_rnn(
            fw_cell,
            model_input_splits[i],
            initial_state=initial_fw_state,
            dtype=tf.float32,
            scope="enc0fw")

        outputs, bw_state = tf.nn.dynamic_rnn(
            bw_cell,
            model_input_splits[num_splits - i - 1],
            initial_state=initial_bw_state,
            dtype=tf.float32,
            scope="enc0bw")
        if is_training:
          fw_state = tf.nn.dropout(fw_state, 0.8)
          bw_state = tf.nn.dropout(bw_state, 0.8)

        first_layer_fw_outputs[i] = fw_state
        if i == num_splits - 1:
          first_layer_bw_outputs[num_splits - 1] = bw_state
        else:
          first_layer_bw_outputs[num_splits - i - 2] = bw_state
      first_layer_fw_outputs = tf.stack(first_layer_fw_outputs, axis=1)
      first_layer_bw_outputs = tf.stack(first_layer_bw_outputs, axis=1)
      first_layer_outputs = tf.concat([first_layer_fw_outputs, first_layer_bw_outputs], 2)

    with tf.variable_scope("EncLayer1"):
      fw_cell = self.get_enc_cell(self.cell_size, vocab_size)
      bw_cell = self.get_enc_cell(self.cell_size, vocab_size)
      enc_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=fw_cell,
          cell_bw=bw_cell,
          inputs=first_layer_outputs,
          dtype=tf.float32,
          scope="enc1")
    enc_state = tf.concat(enc_states, 1)
    # TODO
    if is_training:
      enc_state = tf.nn.dropout(enc_state, 0.8)
    logits = moe.moe_layer(enc_state, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    return {"predictions": logits}

  def get_enc_cell(self, cell_size, vocab_size):
    cell = gru_ops.GRUBlockCell(cell_size)
    return cell
