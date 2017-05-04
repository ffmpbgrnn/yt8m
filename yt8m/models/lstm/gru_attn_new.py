import math

import tensorflow as tf

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from yt8m.models import models
import attn_new
from tensorflow.contrib.rnn.python.ops import gru_ops

slim = tf.contrib.slim

def moe_layer_3d(model_input, hidden_size, num_mixtures,
                 seq_len=0, input_size=0, act_func=None, l2_penalty=None):
  model_input = tf.reshape(model_input, [-1, seq_len, 1, input_size])
  gate_activations = slim.conv2d(model_input, hidden_size * (num_mixtures + 1), [1, 1],
                                 activation_fn=None,
                                 biases_initializer=None,
                                 weights_regularizer=slim.l2_regularizer(l2_penalty),
                                 scope="gates")

  expert_activations = slim.conv2d(model_input, hidden_size * num_mixtures, [1, 1],
                                   activation_fn=None,
                                   biases_initializer=None,
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
  outputs = tf.reshape(outputs, [-1, seq_len, hidden_size])
  return outputs

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

def sample_sequence(model_input, sample_indices, num_samples):
  batch_size = tf.shape(model_input)[0]
  frame_index = tf.cast(tf.reshape(
      tf.convert_to_tensor(sample_indices),
      [1, -1]), tf.int32)
  frame_index = tf.tile(frame_index, [batch_size, 1])
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)

class GRUAttn(models.BaseModel):
  def __init__(self):
    super(GRUAttn, self).__init__()
    self.cell_size = 1024
    self.max_steps = 300

    self.normalize_input = True # TODO
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    self.is_training = is_training
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

    runtime_batch_size = tf.shape(model_input)[0]

    with tf.variable_scope("EncLayer0"):
      with tf.variable_scope("enc0"):
        enc_cell0 = self.get_enc_cell0(self.cell_size,)
        initial_state = enc_cell0.zero_state(runtime_batch_size, dtype=tf.float32)
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            enc_cell0, model_input, initial_state=initial_state, scope="enc0")
      output_ranges = 9 + tf.range(0, 300, 10)
      second_inputs = sample_sequence(enc_outputs, output_ranges, 30)

      # if is_training:
        # second_inputs = tf.nn.dropout(second_inputs, 0.5)
      with tf.variable_scope("enc1"):
        # TODO
        second_inputs = moe_layer_3d(second_inputs, self.cell_size, 2, act_func=None,
                                     seq_len=30, input_size=self.cell_size, l2_penalty=1e-8)
        enc_cell1 = self.get_enc_cell1(self.cell_size,)
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            enc_cell1, second_inputs, initial_state=initial_state, scope="enc1")

      flatten_outputs = attn_new.attn(enc_outputs, fea_size=1024, seq_len=30)

    if self.is_training:
      flatten_outputs = tf.nn.dropout(flatten_outputs, 0.5)
    logits = moe_layer(flatten_outputs, vocab_size, 2,
                       act_func=tf.nn.sigmoid, l2_penalty=1e-8)

    return {"predictions": logits}

  def get_enc_cell0(self, cell_size):
    cell = gru_ops.GRUBlockCell(cell_size)
    return cell

  def get_enc_cell1(self, cell_size):
    cell = gru_ops.GRUBlockCell(cell_size)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, 1024)
    return cell

  '''
    cells = []
    cell = gru_ops.GRUBlockCell(cell_size)
    if self.is_training:
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
    cells.append(cell)

    cell = gru_ops.GRUBlockCell(cell_size)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, 512)
    cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(
        cells,
        state_is_tuple=False)

    return cell
  '''
