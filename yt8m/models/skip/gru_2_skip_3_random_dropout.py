import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import gru_ops
from yt8m.models import models

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

class GRU2Skip3RandomDropout(models.BaseModel):
  def __init__(self):
    super(GRU2Skip3RandomDropout, self).__init__()
    self.normalize_input = True
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4
    self.max_steps = 300

  def create_model(self, model_input, vocab_size=None, num_frames=None,
                   is_training=True, **unused_params):
    self.cell_size = 1024
    self.number_of_layers = 2
    self.vocab_size = vocab_size
    num_skips = 3
    ranges = tf.range(0, 300, num_skips)
    self.is_training = is_training
    if self.is_training:
      input_ranges = tf.random_uniform([1]) * num_skips
      input_ranges = tf.cast(input_ranges[0], tf.int32) + ranges
      model_input = sample_sequence(model_input, input_ranges, self.max_steps / num_skips)
      logits = self.do_rnn(model_input)
    else:
      max_pool = True
      for i in xrange(num_skips):
        input_ranges = i + ranges
        model_input = sample_sequence(model_input, input_ranges, self.max_steps / num_skips)
        if i > 0:
          tf.get_variable_scope().reuse_variables()
          if max_pool:
            logits.append(self.do_rnn(model_input))
          else:
            logits += self.do_rnn(model_input)
        else:
          if max_pool:
            logits = []
            logits.append(self.do_rnn(model_input))
          else:
            logits = self.do_rnn(model_input)
      if max_pool:
        logits = tf.stack(logits, axis=1)
        logits = tf.reduce_max(logits, axis=1)
      else:
        logits /= num_skips


    return {"predictions": logits}

  def do_rnn(self, model_input):
    # print(num_frames.get_shape())
    # num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    # model_input = model_utils.SampleRandomFrames(model_input, num_frames, 10)
    cells = []
    for _ in xrange(self.number_of_layers):
      cell = tf.contrib.rnn.GRUCell(self.cell_size)
      if self.is_training:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
      cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

    # num_frames = num_frames / 3
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(cell, model_input,
                                        # sequence_length=num_frames,
                                        dtype=tf.float32)
    if self.is_training:
      state = tf.nn.dropout(state, 0.5)
    logits = moe_layer(state, self.vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    return logits
