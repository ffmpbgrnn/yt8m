import tensorflow as tf


def hinge_loss(labels, predictions, b=1.0):
  with tf.name_scope("loss_hinge"):
    float_labels = tf.cast(labels, tf.float32)
    all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
    all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
    sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
    hinge_loss = tf.maximum(
        all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
    return hinge_loss
    # return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))

def aucpr_loss(scores, labels):
  num_thresholds = 200
  lambda_t = [0]
  with tf.variable_scope("AUCPRLambda"):
    for i in xrange(num_thresholds - 1):
      lambda_t.append(
          tf.get_variable("lambda_{}".format(i), [],
                          initializer=tf.constant_initializer(0.0)))

  _, top_idxs = tf.nn.top_k(scores, k=20)
  top_scores = tf.gather(scores, top_idxs,
                          name='top_scores')
  top_labels = tf.gather(labels, top_idxs,
                          name='top_labels')

  kepsilon = 1e-7  # to account for floating point imprecisions
  thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                for i in range(num_thresholds-2)]
  thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

  precisions = tf.metrics.precision_at_thresholds(top_labels, top_scores, thresholds)

  num_pos = tf.reduce_sum(top_labels)
  total_loss = 0.
  for i in xrange(1, len(thresholds)):
    delta_t = precisions[i] - precisions[i - 1]
    loss = hinge_loss(top_labels, top_scores, thresholds[i])
    l_p = tf.reduce_sum(loss * tf.where(tf.equal(top_labels, True)))
    l_n = tf.reduce_sum(loss * tf.where(tf.equal(top_labels, False)))

    tmp = (1 + lambda_t[i]) * l_p + lambda_t[i] * precisions[i] / (1. - precisions[i]) * l_n - lambda_t[i] * num_pos
    total_loss += delta_t * tmp
  return total_loss
