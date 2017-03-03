import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# from inception_v2 import inception_v2_arg_scope as inception_arg_scope
# from inception_v2 import inception_v2 as inception
from inception_v3 import inception_v3_arg_scope as inception_arg_scope
from inception_v3 import inception_v3 as inception

slim = tf.contrib.slim


class Inception():
  def __init__(self, videos, labels):
    self.videos = videos
    self.labels = labels
    self.phase_train = True
    self.var_moving_average_decay = -1
    self.global_step = slim.get_or_create_global_step()

    def get_network():
      def network_func(images):
        arg_scope = inception_arg_scope(batch_norm_decay=0.9997)
        with slim.arg_scope(arg_scope):
          return inception(images, 4716, is_training=self.phase_train,
                           dropout_keep_prob=0.9)
      return network_func

    logits, end_points = get_network()(self.videos)

    if self.phase_train:
      total_loss = 0
      if 'AuxLogits' in end_points:
        l = tf.nn.sigmoid_cross_entropy_with_logits(logits=end_points['AuxLogits'], labels=self.labels,)
        l = tf.reduce_sum(l, 1)
        self.aux_loss = 0.4 * tf.reduce_mean(l)
        # self.aux_loss = tf.constant(0., name="aux_loss")
        total_loss += self.aux_loss
      l = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.labels)
      l = tf.reduce_sum(l, 1)
      self.main_cls_loss = tf.reduce_mean(l)
      total_loss += self.main_cls_loss
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.regularization_loss = tf.add_n(regularization_losses)
      total_loss += self.regularization_loss

      lr = tf.constant(0.001, name='learning_rate')

      optimizer = tf.train.RMSPropOptimizer(
          lr,
          decay=0.9,
          momentum=0.9,
          epsilon=1)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,)

      variable_averages = tf.train.ExponentialMovingAverage(
          0.999, self.global_step)
      variables_to_average = (tf.trainable_variables() +
                              tf.moving_average_variables())
      variables_averages_op = variable_averages.apply(variables_to_average)
      variables_averages_op = None

      if variables_averages_op:
        update_ops.append(variables_averages_op)

      params = self._get_variables_to_train()
      grads = tf.gradients(total_loss, params)
      global_norm = tf.global_norm(grads)
      apply_gradient_op = optimizer.apply_gradients(zip(grads, params),
                                               global_step=self.global_step)
      update_ops.append(apply_gradient_op)
      update_op = tf.group(*update_ops)
      # self.update = tf.group(apply_gradient_op, variables_averages_op)
      self.update = control_flow_ops.with_dependencies([update_op], total_loss,
                                                        name='train_op')
      self.loss = total_loss

    if self.phase_train:
      self.feed_out = {
          "train_op": self.update,
          "loss": self.main_cls_loss,
          "global_step": self.global_step,
          "predictions": tf.nn.sigmoid(logits),
          "dense_labels": self.labels,
          "global_norm": global_norm,
        }
    # else:
      # self.probs = tf.nn.softmax(logits)
      # self.feed_out['Acc'] = self.eval_updates
      # self.feed_out['probs'] = self.probs
      # self.feed_out['vnames'] = self.vnames

  def get_init_fn(self, checkpoint_path):
    if "inception_resnet_v2_2016_08_30" in checkpoint_path:
      checkpoint_exclude_scopes = "InceptionV2/Logits,InceptionV2/AuxLogits/Logits"
      exclusions = []
      if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

      variables_to_restore = []
      for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
          if var.op.name.startswith(exclusion):
            excluded = True
            break
        if not excluded:
          variables_to_restore.append(var)

      return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=False)

  def _get_variables_to_train(self):
    return tf.trainable_variables()
