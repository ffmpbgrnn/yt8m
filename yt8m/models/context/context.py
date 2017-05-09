import tensorflow as tf
from yt8m.models import models

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


class ContextModel(models.BaseModel):
  def __init__(self):
    super(ContextModel, self).__init__()
    self.normalize_input = True
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4


  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    sample_ten_frames()
