import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def _confusion_matrix_at_thresholds(
    labels, predictions, thresholds, includes=None):
  all_includes = ('tp', 'fn', 'tn', 'fp')
  if includes is None:
    includes = all_includes
  else:
    for include in includes:
      if include not in all_includes:
        raise ValueError('Invaild key: %s.' % include)

  labels = tf.cast(labels, dtype=tf.bool)
  num_thresholds = len(thresholds)

  # Reshape predictions and labels.
  predictions_2d = tf.reshape(predictions, [-1, 1])
  labels_2d = tf.reshape(
      tf.cast(labels, dtype=tf.bool), [1, -1])

  # Use static shape if known.
  num_predictions = predictions_2d.get_shape().as_list()[0]

  # Otherwise use dynamic shape.
  if num_predictions is None:
    num_predictions = tf.shape(predictions_2d)[0]
  thresh_tiled = tf.tile(
      tf.expand_dims(tf.constant(thresholds), [1]),
      tf.stack([1, num_predictions]))

  # Tile the predictions after thresholding them across different thresholds.
  pred_is_pos = tf.greater(
      tf.tile(tf.transpose(predictions_2d), [num_thresholds, 1]),
      thresh_tiled)
  if ('fn' in includes) or ('tn' in includes):
    pred_is_neg = tf.logical_not(pred_is_pos)

  # Tile labels by number of thresholds
  label_is_pos = tf.tile(labels_2d, [num_thresholds, 1])
  if ('fp' in includes) or ('tn' in includes):
    label_is_neg = tf.logical_not(label_is_pos)

  values = {}
  update_ops = {}

  if 'tp' in includes:
    is_true_positive = tf.to_float(
        tf.logical_and(label_is_pos, pred_is_pos))
    values['tp'] = tf.reduce_sum(is_true_positive, 1)

  if 'fn' in includes:
    is_false_negative = tf.to_float(
        tf.logical_and(label_is_pos, pred_is_neg))
    values['fn'] = tf.reduce_sum(is_false_negative, 1)

  if 'tn' in includes:
    is_true_negative = tf.to_float(
        tf.logical_and(label_is_neg, pred_is_neg))
    values['tn'] = tf.reduce_sum(is_true_negative, 1)

  if 'fp' in includes:
    is_false_positive = tf.to_float(
        tf.logical_and(label_is_neg, pred_is_pos))
    values['fp'] = tf.reduce_sum(is_false_positive, 1)

  return values, update_ops

def precision_at_thresholds(labels, predictions, thresholds,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None, name=None):
  with tf.variable_scope(name, 'precision_at_thresholds',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, includes=('tp', 'fp'))

    # Avoid division by zero.
    epsilon = 1e-7
    def compute_precision(tp, fp, name):
      return tf.div(tp, epsilon + tp + fp, name='precision_' + name)

    prec = compute_precision(values['tp'], values['fp'], 'value')
    update_op = None

    return update_op, prec


def hinge_loss(labels, predictions, b=1.0):
  with tf.name_scope("loss_hinge"):
    float_labels = 2 * tf.cast(labels, tf.float32) - 1
    hinge_loss = tf.nn.relu(1 - float_labels * (predictions - b))
    # TODO
    hinge_loss = hinge_loss * hinge_loss
    return hinge_loss
    # return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))

def aucpr_loss(scores, labels):
  num_thresholds = 10
  with tf.variable_scope("AUCPRLambda"):
    lambda_t = tf.get_variable("lambda_{}".format(0), [num_thresholds+1],
                               regularizer=slim.l2_regularizer(1e-5),
                               initializer=tf.constant_initializer(0.005))
    lambda_t = tf.maximum(0., lambda_t)
                                # initializer=tf.constant_initializer(0.5))

  top_k = 20
  _, top_idxs = tf.nn.top_k(scores, k=top_k)
  batch_size = tf.shape(scores)[0]
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, top_k])

  top_idxs= tf.stack([batch_index, top_idxs], 2)
  top_scores = tf.gather_nd(scores, top_idxs,
                          name='top_scores')
  top_labels = tf.gather_nd(labels, top_idxs,
                          name='top_labels')
  top_scores = tf.cast(tf.reshape(top_scores, [-1]), dtype=tf.float32)
  top_labels = tf.cast(tf.reshape(top_labels, [-1]), dtype=tf.int32)
  # top_scores = tf.cast(tf.reshape(scores, [-1]), dtype=tf.float32)
  # top_labels = tf.cast(tf.reshape(labels, [-1]), dtype=tf.int32)

  num_thresholds_cal = 500
  kepsilon = 1e-7  # to account for floating point imprecisions
  thresholds = [(i + 1) * 1.0 / (num_thresholds_cal - 1)
                for i in range(num_thresholds_cal-2)]
  thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
  # top_labels = tf.Print(top_labels, [tf.reduce_sum(top_labels), tf.reduce_sum(top_labels)])
  # TODO
  _precisions_un, _precisions = precision_at_thresholds(labels=top_labels, predictions=top_scores, thresholds=thresholds)
  # _precisions_un, _precisions = tf.metrics.precision_at_thresholds(labels=top_labels, predictions=top_scores, thresholds=thresholds)
  # _precisions_un, _precisions = tf.contrib.metrics.streaming_precision_at_thresholds(labels=top_labels, predictions=top_scores, thresholds=thresholds)

  # top_labels = tf.Print(top_labels, [top_labels, top_scores[:4], top_scores[-6:-3], top_scores[-3:]])
  # _recalls, _ = tf.metrics.recall_at_thresholds(top_labels, top_scores, thresholds)

  precisions = tf.constant([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

  # _precisions = tf.Print(_precisions, [tf.reduce_sum(_precisions)])
  # _precisions = tf.Print(_precisions, [_precisions[:3], _precisions[3:6], _precisions[-3:]])
  diff_matrix = tf.reshape(_precisions, [1, num_thresholds_cal]) - tf.reshape(precisions[1:], [num_thresholds, 1])
  upper_b = tf.constant(
      np.zeros((num_thresholds, num_thresholds_cal), dtype=np.float32) + 100)
  diff_matrix = tf.where(
      tf.less(diff_matrix, 0.),
      upper_b,
      diff_matrix)
  # diff_matrix = tf.Print(diff_matrix, [tf.reduce_sum(diff_matrix, axis=1)])
  _, ranks = tf.nn.top_k(diff_matrix, k=num_thresholds_cal)
  ranks = ranks[:, -1]
  # ranks = tf.Print(ranks, [ranks, ranks[-3:]])
  thresholds = tf.constant(thresholds)
  b = tf.gather(thresholds, ranks)
  # b = tf.Print(b, [b[:3], b[3:6], b[6:9]])

  # thresholds = tf.get_variable("thresholds", [num_thresholds],)
                                # # initializer=tf.constant_initializer(0.5))
  # thresholds = tf.nn.relu(thresholds)
  # # thresholds = tf.nn.softmax(thresholds)
  # thresholds, _ = tf.nn.top_k(thresholds, k=num_thresholds)
  # thresholds = tf.Print(thresholds, [thresholds])

  num_pos = tf.cast(tf.reduce_sum(top_labels), dtype=tf.float32)
  # num_pos = tf.Print(num_pos, [num_pos])
  # lambda_t = tf.Print(lambda_t, [lambda_t[1], lambda_t[2], lambda_t[3], lambda_t[4], lambda_t[5], lambda_t[6], lambda_t[7], lambda_t[8], lambda_t[9], ])
  total_loss = 0.
  for i in xrange(1, num_thresholds+1):
    delta_t = precisions[i] - precisions[i - 1]
    loss = hinge_loss(top_labels, top_scores, b[i-1])
    # top_labels = tf.Print(top_labels, [top_labels.get_shape()])
    pos_label = tf.cast(tf.equal(top_labels, 1), dtype=tf.float32)
    neg_label = tf.cast(tf.equal(top_labels, 0), dtype=tf.float32)
    l_p = tf.reduce_sum(loss * pos_label)
    l_n = tf.reduce_sum(loss * neg_label)

    tmp = (1 + lambda_t[i]) * l_p + lambda_t[i] * precisions[i] / (1. - precisions[i]) * l_n - lambda_t[i] * num_pos
    # delta_t = tf.Print(delta_t, [delta_t, tmp])
    total_loss += delta_t * tmp
  return total_loss
