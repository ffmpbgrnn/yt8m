import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib

from yt8m.models import models

from layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d)


class Dilation(models.BaseModel):
  def __init__(self):
    super(Dilation, self).__init__()
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-3
    self.num_classes = 1
    self.normalize_input = True

  def create_model(self, model_input, vocab_size, l2_penalty=1e-5,
                   is_training=True, dense_labels=None, **unused_params):

    num_blocks = 3
    num_layers = 3
    num_hidden = 512

    hs = []
    h = model_input
    for b in range(num_blocks):
      for i in range(num_layers):
        rate = 2**i
        name = 'b{}-l{}'.format(b, i)
        h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
        hs.append(h)

    h = tf.reduce_max(h, axis=1)
    logits = slim.fully_connected(
        h, vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    # outputs = conv1d(h,
                    # num_classes,
                    # filter_width=1,
                    # gain=1.0,
                    # activation=None,
                    # bias=True)



    labels = tf.cast(dense_labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    preds = tf.nn.sigmoid(logits)

    return {"predictions": preds, "loss": loss}
