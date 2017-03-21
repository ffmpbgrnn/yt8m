import math

import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

from yt8m.models import models
import yt8m.models.model_utils as utils

linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

class NoisyLabelModel(models.BaseModel):
  def __init__(self):
    super(NoisyLabelModel, self).__init__()
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    # TODO
    self.normalize_input = True
    self.num_max_labels = 4716

  def video_level_simple(self, model_input, labels, sparse_labels, label_weights):
    # batch_size x 1 x fea_size
    runtime_batch_size = tf.shape(model_input)[0]
    scores = slim.fully_connected(model_input, 1 * (self.vocab_size + 1), activation_fn=None)

    score_maps = tf.reduce_max(
        tf.matmul(
            tf.reshape(scores, [-1, 1, (self.vocab_size + 1), 1]),
            tf.reshape(scores, [-1, 1, 1, (self.vocab_size + 1)]),),
        axis=1)
    score_maps = score_maps[:, :self.vocab_size, :]
    score_maps_2d = tf.reshape(score_maps, [-1, self.vocab_size + 1])
    sparse_labels = tf.reshape(sparse_labels, [-1,])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels, logits=score_maps_2d)
    label_weights = tf.reshape(label_weights, [-1])
    loss = tf.reduce_sum(loss * label_weights) / tf.cast(runtime_batch_size, dtype=tf.float32) / 6.

    # pred_for_eval = tf.reshape(
        # tf.nn.softmax(score_maps, dim=2),
        # [-1, self.vocab_size, self.vocab_size + 1])
    # pred_for_eval = pred_for_eval[:, :, :self.vocab_size]
    # pred_for_eval = tf.matrix_diag_part(pred_for_eval)

    # pred_for_eval = tf.reduce_max(pred_for_eval, axis=[1, 2])
    # pred_for_eval = pred_for_eval[:, :self.vocab_size]
    pred_for_eval = tf.zeros((runtime_batch_size, self.vocab_size), dtype=tf.float32)
    return pred_for_eval, loss


  def video_level(self, model_input, labels, sparse_labels, label_weights):
    # batch_size x 1 x fea_size
    runtime_batch_size = tf.shape(model_input)[0]
    input_size = 256#self.fea_size
    model_input = slim.fully_connected(model_input, input_size, activation_fn=tf.nn.relu)
    model_input = tf.reshape(model_input, [-1, 1, input_size])
    # batch_size x vocab_size x 1
    v0 = tf.get_variable("V0", [1, self.vocab_size, 1])
    v0 = tf.tile(v0, [runtime_batch_size, 1, 1])

    # pred_maps -> batch_size x vocab_size x fea_size
    pred_maps = tf.matmul(v0, model_input)
    pred_maps = tf.reshape(pred_maps, [-1, self.vocab_size, 1, input_size])
    score_maps = slim.conv2d(pred_maps, (self.vocab_size + 1), [1, 1], activation_fn=None, scope="cls")

    score_maps_2d = tf.reshape(score_maps, [-1, self.vocab_size + 1])
    sparse_labels = tf.reshape(sparse_labels, [-1,])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels, logits=score_maps_2d)
    label_weights = tf.reshape(label_weights, [-1])
    loss = tf.reduce_sum(loss * label_weights) / tf.cast(runtime_batch_size, dtype=tf.float32) / 6.

    pred_for_eval = tf.reshape(
        tf.nn.softmax(score_maps, dim=3),
        [-1, self.vocab_size, self.vocab_size + 1])
    pred_for_eval = pred_for_eval[:, :, :self.vocab_size]
    pred_for_eval = tf.matrix_diag_part(pred_for_eval)

    # pred_for_eval = tf.reduce_max(pred_for_eval, axis=[1, 2])
    # pred_for_eval = pred_for_eval[:, :self.vocab_size]

    # pred_for_eval = tf.zeros((runtime_batch_size, self.vocab_size), dtype=tf.float32)
    return pred_for_eval, loss

  '''
  Input:  batch_size x num_shot x 1 x fea_len
  Gating: batch_size x num_shot x 1 x num_labels
  Gating: batch_size x (num_shot + 1) x 1 x num_labels
  Label:  batch_size x 1        x 1 x num_labels
  '''
  def create_model(self, model_input, vocab_size, l2_penalty=1e-5,
                   is_training=True, dense_labels=None, sparse_labels=None,
                   label_weights=None, **unused_params):
    self.num_shots = int(60 + 30)
    self.fea_size = 1024 + 128

    self.vocab_size = vocab_size
    labels = tf.cast(dense_labels, tf.float32)
    sparse_labels = tf.cast(sparse_labels, tf.int64)

    if True:
      predictions_for_eval, loss = self.video_level_simple(model_input, labels, sparse_labels, label_weights)
    else:
      # num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

      shot_splits = tf.split(model_input, num_or_size_splits=60, axis=1)
      shot_splits = shot_splits + tf.split(model_input, num_or_size_splits=30, axis=1)
      shots = []
      for shot in shot_splits:
        shots.append(tf.reduce_mean(shot, axis=1, keep_dims=True))
        # tf.zeros_like(tensor, dtype=None, name=None, optimize=True)
      shots = tf.concat(shots, axis=1)
      shots = tf.reshape(shots, [-1, self.num_shots, 1, self.fea_size])

      predictions_for_eval, loss = self.do_gating_multi_label(shots, labels, sparse_labels, label_weights)

    return {"predictions": predictions_for_eval, "loss": loss}

  def do_gating_multi_label(self, shots, labels, sparse_labels, label_weights):
    runtime_batch_size = tf.shape(shots)[0]
    # gate0 -> batch_size x num_shots, 1, 512
    hidden_vec_size = 512
    gate0 = slim.conv2d(shots, hidden_vec_size, [1, 1], activation_fn=tf.tanh, scope="gates0",
                        biases_initializer=tf.zeros_initializer(),)
    # gate0 -> batch_size x num_shots x 512
    gate0 = tf.reshape(gate0, [-1, self.num_shots, hidden_vec_size])
    # v0 -> 1 x 512 x vocab_size
    v0 = tf.get_variable("V0", [1, hidden_vec_size, self.vocab_size])
    # hidden -> batch_size x num_shots x vocab_size
    hidden = tf.matmul(gate0, tf.tile(v0, [runtime_batch_size, 1, 1]))
    # a -> batch_size x num_shots x vocab_size
    a = tf.nn.softmax(hidden, dim=1)

    # pred_maps -> batch_size x vocab_size x fea_size
    pred_maps = tf.matmul(
        a,
        tf.reshape(shots, [-1, self.num_shots, self.fea_size]),
        transpose_a=True)
    pred_maps = tf.reshape(pred_maps, [-1, self.vocab_size, 1, self.fea_size])
    score_maps = slim.conv2d(pred_maps, self.vocab_size + 1, [1, 1], activation_fn=None, scope="cls")
    score_maps_2d = tf.reshape(score_maps, [-1, self.vocab_size + 1])
    sparse_labels = tf.reshape(sparse_labels, [-1,])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels, logits=score_maps_2d)
    label_weights = tf.reshape(label_weights, [-1])
    loss = tf.reduce_sum(loss * label_weights) / tf.cast(runtime_batch_size, dtype=tf.float32)

    pred_for_eval = tf.nn.softmax(score_maps, dim=3)
    pred_for_eval = tf.reduce_max(pred_for_eval, axis=[1, 2])
    pred_for_eval = pred_for_eval[:, :self.vocab_size]
    '''
    pred_for_eval = tf.nn.softmax(score_maps, dim=2)
    range_idx = tf.tile(
        tf.expand_dims(tf.range(runtime_batch_size), 1), [1, self.vocab_size])
    label_idx = tf.tile(
        tf.expand_dims(tf.range(self.vocab_size), 0), [runtime_batch_size, 1])

    pred_idx = tf.stack([range_idx, label_idx, label_idx], 2)
    pred_for_eval = tf.gather_nd(pred_for_eval, pred_idx)
    pred_for_eval = tf.Print(pred_for_eval, [tf.shape(pred_for_eval)])
    '''
    return pred_for_eval, loss


  def do_gating_0(self, shots, labels, sparse_labels=None):
    runtime_batch_size = tf.shape(shots)[0]
    # gate0 -> batch_size x num_shots, 1, 512
    hidden_vec_size = 512
    gate0 = slim.conv2d(shots, hidden_vec_size, [1, 1], activation_fn=tf.tanh, scope="gates0",
                        biases_initializer=tf.zeros_initializer(),)
    # gate0 -> batch_size x num_shots x 512
    gate0 = tf.reshape(gate0, [-1, self.num_shots, hidden_vec_size])
    # v0 -> 1 x 512 x (vocab_size + 1)
    v0 = tf.get_variable("V0", [1, hidden_vec_size, (self.vocab_size + 1)], initializer=tf.constant_initializer(1.))
    # hidden -> batch_size x num_shots x (vocab_size + 1)
    hidden = tf.matmul(gate0, tf.tile(v0, [runtime_batch_size, 1, 1]))
    # label_smoother -> batch_size x num_shots x (vocab_size + 1)
    # TODO
    # label_smoother = tf.nn.softmax(hidden, dim=2)
    label_smoother = tf.sigmoid(hidden)

    # gate1 -> batch_size x num_shots, 1, 512
    gate1 = slim.conv2d(shots, hidden_vec_size, [1, 1], activation_fn=tf.tanh, scope="gates1",
                        biases_initializer=tf.ones_initializer(),)
    # v0 -> 1 x 1 x 1 x hidden_vec_size
    v1 = tf.get_variable("V1", [1, 1, 1, hidden_vec_size], initializer=tf.constant_initializer(1.))
    # hidden -> batch_size x num_shots
    hidden = tf.reduce_sum(v1 * gate1, axis=[2, 3])
    # shot_smoother -> batch_size x num_shots
    # TODO
    # shot_smoother = tf.nn.softmax(hidden, dim=1)
    shot_smoother = tf.sigmoid(hidden)

    predictions = slim.conv2d(shots, self.vocab_size, [1, 1], activation_fn=None, scope="cls")
    # predictions -> batch_sizexnum_shots x vocab_size
    predictions = tf.reshape(predictions, [-1, self.vocab_size])

    labels = tf.reshape(labels, [-1, 1, self.vocab_size])
    # labels -> batch_sizexnum_shots x vocab_size
    labels = label_smoother[:, :, :self.vocab_size] * labels
    labels = tf.reshape(labels, [-1, self.vocab_size])

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=abels, logits=predictions)
    loss = tf.reshape(loss, [-1, self.num_shots])
    # loss = shot_smoother * loss
    loss = tf.reduce_mean(loss)
    predictions_for_eval = tf.nn.softmax(predictions)
    predictions_for_eval = tf.reshape(predictions_for_eval, [-1, self.num_shots, self.vocab_size])
    predictions_for_eval = tf.reduce_mean(predictions_for_eval, axis=1)
    return predictions_for_eval, loss


  def do_gating(self):
    if True:
      gating = slim.conv2d(shots, self.vocab_size, [1, 1], activation_fn=None, scope="gating",
                           biases_initializer=tf.ones_initializer(),)
    else:
      gating = slim.conv2d(shots, self.vocab_size, [1, 1], activation_fn=tf.nn.relu, scope="gating0",)
      gating = slim.conv2d(gating, self.vocab_size, [1, 1], activation_fn=None, scope="gating",
                           biases_initializer=tf.ones_initializer(),)
    '''
    gating = slim.conv2d(shots, self.vocab_size, [1, 1], activation_fn=tf.nn.relu, scope="gating0",
                         biases_initializer=tf.zeros_initializer(),)
    gating = tf.transpose(gating, [0, 3, 2, 1])
    gating = slim.conv2d(gating, (self.num_shots + 1), [1, 1], activation_fn=None, scope="gating1",
                         biases_initializer=tf.ones_initializer(),)
    gating = tf.transpose(gating, [0, 3, 2, 1])
    '''
    gating = tf.nn.softmax(
        gating,
        dim=1)

    labels = tf.reshape(tf.cast(dense_labels, tf.float32), [-1, 1, 1, self.vocab_size])
    if False:
      final_labels = gating * labels
    else:
      final_labels = tf.tile(labels, [1, self.num_shots, 1, 1])

    if True:
      predictions = slim.conv2d(shots, self.vocab_size, [1, 1], activation_fn=None, scope="cls")
    else:
      predictions = slim.conv2d(shots, self.vocab_size, [1, 1], activation_fn=tf.nn.relu, scope="cls0")
      predictions = slim.conv2d(predictions, self.vocab_size, [1, 1], activation_fn=None, scope="cls1")

    pred_for_eval = tf.sigmoid(predictions)
    pred_for_eval = tf.reduce_mean(pred_for_eval, axis=[1, 2])

    final_labels = tf.reshape(final_labels, [-1, self.vocab_size])
    predictions = tf.reshape(predictions, [-1, self.vocab_size])
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=final_labels, logits=predictions)
    # loss = tf.reduce_mean(loss)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=final_labels, logits=predictions)
    if True:
      loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(gating, [-1, self.vocab_size]) * loss, 1))
    else:
      loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
