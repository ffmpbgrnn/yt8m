import tensorflow as tf
from yt8m.models import models
from tensorflow.contrib.rnn.python.ops import gru_ops
import attn_new

slim = tf.contrib.slim


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

def SampleRandomFrames(model_input, num_frames, num_samples):
  batch_size = tf.shape(model_input)[0]
  frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, num_samples]),
          tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)

def SampleRandomSequence(model_input, num_frames, num_samples):
  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)


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

class RandomSequence(models.BaseModel):
  def __init__(self):
    super(RandomSequence, self).__init__()
    self.normalize_input = True
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4


  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    # output_ranges = 9 + tf.range(0, 300, 10)
    # second_inputs = sample_sequence(model_input, output_ranges, 20)

    first_layer_outputs = []
    num_splits = 15
    with tf.variable_scope("EncLayer0"):
      cell = gru_ops.GRUBlockCell(1024)
      for i in xrange(num_splits):
        frames = SampleRandomSequence(model_input, num_frames, 30)
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell, frames, scope="enc0")
        enc_state = moe_layer(enc_state, 1024, 4, act_func=None, l2_penalty=1e-12)
        if is_training:
          enc_state = tf.nn.dropout(enc_state, 0.5)
        first_layer_outputs.append(enc_state)

    with tf.variable_scope("EncLayer1"):
      cell = gru_ops.GRUBlockCell(1024)
      first_layer_outputs = tf.stack(first_layer_outputs, axis=1)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(
          cell, first_layer_outputs, scope="enc1")

    # flatten_outputs = attn_new.attn(enc_outputs, fea_size=1024, seq_len=num_splits)
    flatten_outputs = tf.reduce_mean(enc_outputs, axis=1)

    with tf.variable_scope("FC0"):
      flatten_outputs = moe_layer(flatten_outputs, 1024, 2, act_func=tf.nn.relu, l2_penalty=1e-8)
    if is_training:
      flatten_outputs = tf.nn.dropout(flatten_outputs, 0.5)
    with tf.variable_scope("FC1"):
      logits = moe_layer(flatten_outputs, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    logits = tf.clip_by_value(logits, 0., 1.)
    return {"predictions": logits}
