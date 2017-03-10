import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

from yt8m.models import models
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

class NetVLAD(models.BaseModel):
  def __init__(self):
    super(NetVLAD, self).__init__()
    self.normalize_input = False
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4
    self.max_steps = 300


  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, sparse_labels=None, label_weights=None,
                   dense_labels=None, input_weights=None, **unused_params):
    vlad_att_hidden_size = 100
    C = 20
    loss_with_vlad_kmeans = True

    input_size = 1024+128#tf.shape(model_input)[-1]
    fea_size = 256
    model_input = tf.reshape(model_input, [-1, self.max_steps, 1, input_size])
    model_input = slim.conv2d(model_input, fea_size, [1, 1], activation_fn=None, scope="input_proj")
    model_input = tf.reshape(model_input, [-1, self.max_steps, fea_size])
    input_weights = tf.tile(
        tf.expand_dims(input_weights, 2),
        [1, 1, fea_size])
    inputs = model_input * input_weights


    with tf.variable_scope("centers"):
      center = tf.get_variable("center", shape=[1, C, 1, fea_size], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01),
                               regularizer=slim.l2_regularizer(1e-5))
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
    logits = slim.fully_connected(
        outputs, vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(1e-8))

    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    return {
        "predictions": tf.nn.sigmoid(logits),
        "loss": loss + l2_loss * 1e-8,
    }

  def normalization(self, outputs, output_size, ssr=True, intra_norm=True, l2_norm=True, norm_dim=2):
    if ssr:
      outputs = tf.sign(outputs) * tf.sqrt(tf.abs(outputs) + 1e-12)
    if intra_norm:
      outputs = tf.nn.l2_normalize(outputs, norm_dim)

    outputs = tf.reshape(outputs, [-1, output_size])
    if l2_norm:
      outputs = tf.nn.l2_normalize(outputs, [1])
    return outputs
