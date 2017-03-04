import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from yt8m.models import models
from inception_v2 import inception_v2_arg_scope as inception_arg_scope
from inception_v2 import inception_v2 as inception
# from inception_v3 import inception_v3_arg_scope as inception_arg_scope
# from inception_v3 import inception_v3 as inception

slim = tf.contrib.slim


class Inception(models.BaseModel):
  def __init__(self,):
    super(Inception, self).__init__()
    self.normalize_input = False
    self.var_moving_average_decay = -1

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, label_weights=None,
                   **unused_params):
    def get_network():
      def network_func(images):
        arg_scope = inception_arg_scope(batch_norm_decay=0.9997)
        with slim.arg_scope(arg_scope):
          return inception(images, vocab_size, is_training=is_training,
                           dropout_keep_prob=0.9)
      return network_func

    logits, end_points = get_network()(model_input)

    total_loss = 0
    if 'AuxLogits' in end_points:
      l = tf.nn.sigmoid_cross_entropy_with_logits(logits=end_points['AuxLogits'], labels=self.labels,)
      l = tf.reduce_sum(l, 1)
      aux_loss = 0.4 * tf.reduce_mean(l)
      # self.aux_loss = tf.constant(0., name="aux_loss")
      total_loss += aux_loss
    l = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=dense_labels)
    l = tf.reduce_sum(l, 1)
    main_cls_loss = tf.reduce_mean(l)
    total_loss += main_cls_loss

    return {
      "loss": total_loss,
      "predictions": tf.nn.sigmoid(logits),
    }

  def get_train_init_fn(self):
    checkpoint_path = "/data/uts700/linchao/yt8m/YT/inception_v2.ckpt"
    if "inception" in checkpoint_path:
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
