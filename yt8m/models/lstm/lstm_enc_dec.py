import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

from yt8m.models import models
import yt8m.models.model_utils as utils
import attn

class LSTMEncDec(models.BaseModel):
  def __init__(self):
    super(LSTMEncDec, self).__init__()

    self.normalize_input = False
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4

    self.cell_size = 1024
    self.max_steps = 30
    self.num_max_labels = 10

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, sparse_labels=None, label_weights=None,
                   **unused_params):

    self.phase_train = is_training
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    model_inputs = utils.SampleRandomSequence(model_input, num_frames,
                                              self.max_steps)

    total_vocab_size = vocab_size + 3
    enc_cell = self.get_enc_cell(self.cell_size, total_vocab_size)
    dec_cell = self.get_dec_cell(self.cell_size)
    runtime_batch_size = tf.shape(model_inputs)[0]

    # TODO
    if False:
      with tf.variable_scope("Enc"):
        enc_init_state = enc_cell.zero_state(runtime_batch_size, dtype=tf.float32)
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            enc_cell, model_inputs, initial_state=enc_init_state, scope="enc")
    else:
      enc_outputs = model_inputs
      enc_state = dec_cell.zero_state(runtime_batch_size, dtype=tf.float32)

    label_weights = tf.cast(label_weights, tf.float32)
    dec_weights = tf.unstack(label_weights, axis=1)
    dec_input_lists = tf.unstack(sparse_labels, axis=1)

    dec_targets = [dec_input_lists[i + 1]
                   for i in xrange(len(dec_input_lists) - 1)]
    dec_targets += [tf.zeros_like(dec_input_lists[0])]
    # enc_outputs_lists = tf.split(enc_outputs, num_or_size_splits=self.max_steps, axis=1)
    dec_outputs, _ = attn.embedding_attention_decoder(
        dec_input_lists, initial_state=enc_state,
        attention_states=enc_outputs,
        cell=dec_cell, num_symbols=total_vocab_size, embedding_size=1024,
        output_size=total_vocab_size,
        output_projection=None, feed_previous=False,
        dtype=tf.float32, scope="LSTMEncDec")
    loss = seq2seq_lib.sequence_loss(
        dec_outputs, dec_targets, dec_weights,
        softmax_loss_function=None)
    # logits = tf.reduce_mean(dec_outputs, axis=0)
    label_num = tf.reduce_sum(label_weights, axis=1, keep_dims=True)
    logits = tf.add_n(dec_outputs) / label_num
    # logits = tf.Print(logits.get_shape(), [logits])
    logits = logits[:, :vocab_size]
    # logits = tf.nn.sigmoid(enc_outputs[:, -1, :])
    return {
        "predictions": dec_outputs,
        # "predictions": logits,
        "loss": loss,
    }

  def get_enc_cell(self, cell_size, vocab_size):
    cell = core_rnn_cell.GRUCell(cell_size)
    if self.phase_train:
      cell = core_rnn_cell.DropoutWrapper(
          cell, input_keep_prob=0.5, output_keep_prob=0.5)
    cell = core_rnn_cell.InputProjectionWrapper(cell, cell_size)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, cell_size)
    return cell

  def get_dec_cell(self, cell_size):
    cell = core_rnn_cell.GRUCell(cell_size)
    # TODO
    if True:
      num_layers = 2
      if self.phase_train:
        cell = core_rnn_cell.DropoutWrapper(
            cell, input_keep_prob=0.5)
      cell = core_rnn_cell.MultiRNNCell([cell] * num_layers)
      if self.phase_train:
        cell = core_rnn_cell.DropoutWrapper(
            cell, output_keep_prob=0.5)
    else:
      if self.phase_train:
        cell = core_rnn_cell.DropoutWrapper(
            cell, input_keep_prob=0.5, output_keep_prob=0.5)
    return cell
