import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

from yt8m.models import models
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import yt8m.starter.video_level_models as video_level_models
from . import utils
from tensorflow import logging

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

from tensorflow.python.ops import rnn_cell_impl
# pylint: disable=protected-access
_state_size_with_prefix = rnn_cell_impl._state_size_with_prefix
# pylint: enable=protected-access

class NetVLAD(models.BaseModel):
  def __init__(self):
    super(NetVLAD, self).__init__()
    self.normalize_input = False
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 4e-3 # 4e-3, 3e-4
    self.max_steps = 300


  def context_encoder(self, inputs, fea_size):
    def get_cell():
      cell_size = 1024
      cell = core_rnn_cell.GRUCell(cell_size)
      cell = core_rnn_cell.OutputProjectionWrapper(cell, fea_size)
      return cell

    runtime_batch_size = tf.shape(inputs)[0]
    enc_cell = get_cell()
    init_state = enc_cell.zero_state(runtime_batch_size, dtype=tf.float32)
    enc_outputs, enc_state = tf.nn.dynamic_rnn(
        enc_cell, inputs, initial_state=init_state, scope="enc")

    input_lists = tf.split(inputs, num_or_size_splits=300, axis=1)
    target_lists = []
    for i in xrange(300-1):
      target_lists.append(
          tf.stop_gradient(input_lists[i+1]))
    target_lists.append(input_lists[0])
    targets = tf.stack(target_lists, axis=1)

    loss = tf.reduce_sum(tf.abs(enc_outputs - targets)) / tf.cast(runtime_batch_size, dtype=tf.float32)
    outputs = tf.stop_gradient(enc_outputs) + inputs
    return outputs

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, sparse_labels=None, label_weights=None,
                   dense_labels=None, input_weights=None, **unused_params):
    self.vlad_att_hidden_size = 100
    self.C = 20
    self.loss_with_vlad_kmeans = True
    self.vocab_size = vocab_size

    input_size = 1024+128#tf.shape(model_input)[-1]
    self.fea_size = 256


    model_input = tf.reshape(model_input, [-1, self.max_steps, 1, input_size])
    model_input = slim.conv2d(model_input, self.fea_size, [1, 1], activation_fn=None, scope="input_proj")
    model_input = tf.reshape(model_input, [-1, self.max_steps, self.fea_size])
    input_weights = tf.tile(
        tf.expand_dims(input_weights, 2),
        [1, 1, self.fea_size])
    inputs = model_input * input_weights

    # inputs = self.context_encoder(inputs, fea_size)
    residual, kmeans_loss = self.query_loop(inputs)


    outputs = self.normalization(residual, self.C * self.fea_size, ssr=True,
                                 intra_norm=True, l2_norm=True, norm_dim=2)
    # outputs = tf.stop_gradient(outputs)
    # moe = video_level_models.MoeModel()
    # outputs = moe.moe_layer(outputs, 1024, num_mixtures=5, act_func=tf.nn.relu,
                            # l2_penalty=1e-8)

    logits = slim.fully_connected(
        outputs, self.vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(1e-8), scope="outputs")

    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    logits = tf.nn.sigmoid(logits)
    loss += kmeans_loss * 1e-8

    return {
        "predictions": logits,
        "loss": loss,
    }

  def get_train_init_fn(self):
    # TODO
    return None
    logging.info('restoring from...')
    return slim.assign_from_checkpoint_fn(
        "/data/D2DCRC/linchao/YT/log/386/model.ckpt-2712318",
        tf.all_variables(),
        ignore_missing_vars=True)

  def normalization(self, outputs, output_size, ssr=True, intra_norm=True, l2_norm=True, norm_dim=2):
    if ssr:
      outputs = tf.sign(outputs) * tf.sqrt(tf.abs(outputs) + 1e-12)
    if intra_norm:
      outputs = tf.nn.l2_normalize(outputs, norm_dim)

    outputs = tf.reshape(outputs, [-1, output_size])
    if l2_norm:
      outputs = tf.nn.l2_normalize(outputs, [1])
    return outputs


  def query_loop(self, inputs):
    flat_input = nest.flatten(inputs)
    # (B,T,D) => (T,B,D)
    flat_input = [tf.convert_to_tensor(input_) for input_ in flat_input]
    flat_input = tuple(utils.transpose_batch_time(input_) for input_ in flat_input)

    with tf.variable_scope("centers"):
      # TODO
      center_reg = None # slim.l2_regularizer(1e-5)
      center = tf.get_variable("center", shape=[1, self.C, 1, self.fea_size], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01),
                               regularizer=center_reg)
    hidden = slim.conv2d(center, self.vlad_att_hidden_size, [1, 1], activation_fn=None,
                         scope="hidden_conv2d")

    v = tf.get_variable("attn_v", [1, 1, 1, self.vlad_att_hidden_size],
                        initializer=tf.constant_initializer(0.01))

    def attn(query):
      query = tf.reshape(query, [-1, self.fea_size])
      y = linear(query, self.vlad_att_hidden_size, True, 0.0)
      y = tf.reshape(y, [-1, 1, 1, self.vlad_att_hidden_size])
      o = tf.reduce_sum(v * tf.tanh(hidden + y), [2, 3])
      o = tf.reshape(o, [-1, self.C])
      a = tf.nn.softmax(o)
      a = tf.reshape(a, [-1, self.C, 1, 1])
      return a

    with tf.variable_scope("query_loop") as varscope:
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)
      input_shape = tuple(tf.shape(input_) for input_ in flat_input)

      inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

      flat_input = nest.flatten(inputs)

      input_shape = tf.shape(flat_input[0])
      time_steps = input_shape[0]
      batch_size = input_shape[1]

      inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                              for input_ in flat_input)

      const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

      time = tf.constant(0, dtype=tf.int32, name="time")

      with tf.name_scope("dynamic_query") as scope:
        base_name = scope

      def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

      input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                      for i in range(len(flat_input)))

      input_ta = tuple(ta.unstack(input_)
                      for ta, input_ in zip(input_ta, flat_input))

      def _time_step(time, kmeans_loss, residual):
        input_t = tuple(ta.read(time) for ta in input_ta)
        for input_, shape in zip(input_t, inputs_got_shape):
          input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        input_t = tf.reshape(input_t, [-1, 1, 1, self.fea_size])
        a = attn(input_t)
        d = a * (center + input_t)
        l = tf.reduce_sum(d * d) / tf.cast(batch_size, dtype=tf.float32) / self.C
        kmeans_loss += l
        residual = residual + d

        return (time + 1, kmeans_loss, residual)

      kmeans_loss = tf.constant(0., dtype=tf.float32)
      residual = tf.zeros([batch_size, self.C, 1, self.fea_size], dtype=tf.float32)
      parallel_iterations = 32
      swap_memory = False
      _, kmeans_loss, residual= tf.while_loop(
          cond=lambda time, *_: time < time_steps,
          body=_time_step,
          loop_vars=(time, kmeans_loss, residual),
          parallel_iterations=parallel_iterations,
          swap_memory=swap_memory)

      kmeans_loss /= tf.cast(time_steps, dtype=tf.float32)

      return (residual, kmeans_loss)
