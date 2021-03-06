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

class LSTMEncoder(models.BaseModel):
  def __init__(self):
    super(LSTMEncoder, self).__init__()

    self.normalize_input = False
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4

    self.cell_size = 1024
    self.max_steps = 150

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, **unused_params):
    """
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    self.phase_train = is_training
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    model_input = utils.SampleRandomSequence(model_input, num_frames,
                                             self.max_steps)

    enc_cell = self.get_enc_cell(self.cell_size, vocab_size)
    # dec_cell = self.get_dec_cell(self.cell_size)
    runtime_batch_size = tf.shape(model_input)[0]

    with tf.variable_scope("Enc"):
      enc_init_state = enc_cell.zero_state(runtime_batch_size, dtype=tf.float32)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(
          enc_cell, model_input, initial_state=enc_init_state, scope="enc")

    logits = tf.nn.sigmoid(enc_outputs[:, -1, :])
    return {"predictions": logits}
    # with tf.variable_scope("Dec"):
      # dec_init_state = dec_cell.zero_state(runtime_batch_size, dtype=tf.float32)
      # dec_outputs, _ = tf.nn.dynamic_rnn(
          # dec_cell, self.dec_inputs, initial_state=dec_init_state, scope="dec")

  def get_enc_cell(self, cell_size, vocab_size):
    cell = core_rnn_cell.GRUCell(cell_size)
    if self.phase_train:
      cell = core_rnn_cell.DropoutWrapper(
          cell, input_keep_prob=0.5, output_keep_prob=0.5)
    cell = core_rnn_cell.InputProjectionWrapper(cell, cell_size)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, vocab_size)
    return cell

  def get_dec_cell(self, cell_size):
    cell = core_rnn_cell.GRUCell(cell_size)
    if self.phase_train:
      cell = core_rnn_cell.DropoutWrapper(
          cell, input_keep_prob=0.5, output_keep_prob=0.5)
    cell = core_rnn_cell.InputProjectionWrapper(cell, cell_size)
    return cell
