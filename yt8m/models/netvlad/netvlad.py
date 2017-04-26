import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

from yt8m.models import models
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import yt8m.starter.video_level_models as video_level_models
from tensorflow import logging

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

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
    vlad_att_hidden_size = 100
    C = 20
    loss_with_vlad_kmeans = True
    self.vocab_size = vocab_size

    input_size = 1024+128#tf.shape(model_input)[-1]
    fea_size = 256


    model_input = tf.reshape(model_input, [-1, self.max_steps, 1, input_size])
    model_input = slim.conv2d(model_input, fea_size, [1, 1], activation_fn=None, scope="input_proj")
    model_input = tf.reshape(model_input, [-1, self.max_steps, fea_size])
    input_weights = tf.tile(
        tf.expand_dims(input_weights, 2),
        [1, 1, fea_size])
    inputs = model_input * input_weights

    # inputs = self.context_encoder(inputs, fea_size)

    with tf.variable_scope("centers"):
      # TODO
      center_reg = None # slim.l2_regularizer(1e-5)
      center = tf.get_variable("center", shape=[1, C, 1, fea_size], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01),
                               regularizer=center_reg)
    hidden = slim.conv2d(center, vlad_att_hidden_size, [1, 1], activation_fn=None, scope="hidden_conv2d")

    v = tf.get_variable("attn_v", [1, 1, 1, vlad_att_hidden_size],
                        initializer=tf.constant_initializer(0.01))

    def attn(query):
      query = tf.reshape(query, [-1, fea_size])
      y = linear(query, vlad_att_hidden_size, True, 0.0)
      y = tf.reshape(y, [-1, 1, 1, vlad_att_hidden_size])
      o = tf.reduce_sum(v * tf.tanh(hidden + y), [2, 3])
      o = tf.reshape(o, [-1, C])
      a = tf.nn.softmax(o)
      a = tf.reshape(a, [-1, C, 1, 1])
      return a

    l2_loss = 0
    with tf.variable_scope("vlad"):
      # query = tf.reshape(inputs, [-1, fea_size])
      a = attn(inputs)
      d = a * (center + tf.reshape(inputs, [-1, 1, 1, fea_size]))
      if is_training and loss_with_vlad_kmeans:
        l2_loss = tf.reduce_mean(
            tf.reduce_sum(d * d, axis=3))
      d = tf.reshape(d, [-1, self.max_steps, C, fea_size])
      d = tf.reduce_sum(d, axis=1)
    residual = d

    '''
    batch_size = tf.shape(model_input)[0]
    with tf.variable_scope('vlad'):
      for i in xrange(seq_length):
        if i > 0:
          tf.get_variable_scope().reuse_variables()

        ins = tf.reshape(inputs[:, i, :], [-1, 1, 1, fea_size])
        a = attn(ins)
        d = a * (center + ins)
        if is_training and loss_with_vlad_kmeans:
          loss_ = tf.reduce_sum(d * d, [0, 1, 2, 3]) / batch_size / C
          if i == 0:
            total_loss_ = loss_
          else:
            total_loss_ += loss_
        if residual is None:
          residual = d
        else:
          residual = residual + d
      if is_training and loss_with_vlad_kmeans:
        total_loss_ /= seq_length
    '''
    outputs = self.normalization(residual, C * fea_size, ssr=True,
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
    loss += l2_loss * 1e-8
    '''
    # TODO
    # self.variables_to_restore = slim.get_model_variables()
    '''

    # logits = self.get_final_probs(outputs)
    '''
    with tf.name_scope("loss_xent"):
      epsilon = 1e-6
      cross_entropy_loss = labels * tf.log(logits + epsilon) + (
          1 - labels) * tf.log(1 - logits + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
    '''

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

  def get_final_probs00(self, predictions):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    outputs = slim.fully_connected(outputs, 1024, activation_fn=None,
                                   weights_regularizer=slim.l2_regularizer(1e-8),
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   scope="cls_proj")

  def get_final_probs0(self, predictions):
    num_mixtures = 5
    l2_penalty = 1e-8
    gates = slim.fully_connected(
        predictions,
        num_mixtures * (self.vocab_size + 1),
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    gate_weights = slim.fully_connected(
        predictions,
        num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gate_weights")
    gate_activations = tf.nn.softmax(
        tf.reshape(gates, [-1, num_mixtures, (self.vocab_size + 1)]), dim=2)
    # preds = gate_activations * tf.reshape(gate_weights, [-1, num_mixtures, 1])
    preds = gate_activations
    preds = tf.reduce_max(preds, axis=1)[:, :self.vocab_size]
    '''
    preds = slim.fully_connected(
        preds,
        self.vocab_size,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="preds")
    '''
    return preds

  def get_final_probs(self, predictions):
    num_mixtures = 4
    l2_penalty = 1e-8
    # predictions = slim.fully_connected(
        # predictions,
        # 1024,
        # activation_fn=tf.nn.relu,
        # biases_initializer=None,
        # weights_regularizer=slim.l2_regularizer(l2_penalty),
        # scope="input_maping")

    gate_activations = slim.fully_connected(
        predictions,
        self.vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=tf.constant_initializer(1.),
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        predictions,
        self.vocab_size * num_mixtures,
        activation_fn=None,
        biases_initializer=tf.constant_initializer(1.),
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, self.vocab_size])
    return final_probabilities
