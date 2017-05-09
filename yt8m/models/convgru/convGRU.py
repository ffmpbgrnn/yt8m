import tensorflow as tf
from yt8m.models import models
from . import utils

slim = tf.contrib.slim

def moe_layer(model_input, hidden_size, num_mixtures,
                act_func=None, l2_penalty=None):
  gate_activations = slim.fully_connected(
      model_input,
      hidden_size * (num_mixtures + 1),
      activation_fn=None,
      biases_initializer=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="gates")
  expert_activations = slim.fully_connected(
      model_input,
      hidden_size * num_mixtures,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="experts")

  expert_act_func = act_func
  gating_distribution = tf.nn.softmax(tf.reshape(
      gate_activations,
      [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
  expert_distribution = tf.reshape(
      expert_activations,
      [-1, num_mixtures])  # (Batch * #Labels) x num_mixtures
  if expert_act_func is not None:
    expert_distribution = expert_act_func(expert_distribution)

  outputs = tf.reduce_sum(
      gating_distribution[:, :num_mixtures] * expert_distribution, 1)
  outputs = tf.reshape(outputs, [-1, hidden_size])
  return outputs

class ConvGRU(models.BaseModel):
  def __init__(self):
    super(ConvGRU, self).__init__()
    # self.height = 3
    # self.vec_size = 1536
    self.normalize_input = True
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 5e-4

    self.rx_step = 6
    self.dropout_ratio = 0.2
    self.layer_scale = 1.
    self.kw = 3
    self.kh = 3
    self.nmaps = 256
    self.nconvs = 2
    self.cutoff = 1.2
    self.length = 18
    self.output_layer_type = 1
    self.dropout_keep_prob = 1.0 - self.dropout_ratio * 8.0 / self.length

    self.num_out_length = 300

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, input_weights=None, dense_labels=None, **unused_params):
    self.phase_train = is_training
    self.mask = input_weights
    self.mask = tf.reshape(self.mask, [-1, 3, 100])
    self.mask = tf.transpose(self.mask, perm=[0, 2, 1])
    self.mask = tf.reduce_max(self.mask, axis=2)

    self.run_time_batch_size = tf.shape(model_input)[0]
    model_input = tf.reshape(model_input, [-1, 3, 100, 1024 + 128])
    model_input = tf.transpose(model_input, perm=[0, 2, 1, 3])
    with tf.variable_scope("Frames"):
      frame_layer_input = tf.tanh(
          utils.conv_linear(model_input, 1, 1, self.nmaps, "input"))
    self.last_layer = self.construct_all_layers(frame_layer_input, self.mask)

    logits = tf.reduce_mean(self.last_layer, [2], keep_dims=True)
    logits = logits * tf.cast(tf.reshape(self.mask, [-1, 100, 1, 1]), tf.float32)
    logits = utils.conv_linear(logits, 1, 1, self.nmaps, "output0")
    logits = tf.nn.relu(logits)
    logits = tf.reduce_max(logits, [1], keep_dims=True)

    logits = tf.squeeze(logits, [1, 2])
    logits = moe_layer(logits, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    logits = tf.clip_by_value(logits, 0., 1.)
    return {"predictions": logits}

  def construct_all_layers(self, first0, mask):
    # self.output_layers = tf.to_int32(tf.reduce_sum(mask, [1, 2, 3]))
    # TODO, set to 50
    self.output_layers = tf.ones([self.run_time_batch_size], dtype=tf.int32)
    self.output_layers += self.num_out_length

    cur0 = first0

    # it = tf.get_variable("layer_index", [], dtype=tf.int32,
                         # initializer=tf.constant_initializer(0))
    it = tf.constant(0, name="layer_index", dtype=tf.int32)
    # Using swap is slower, but saves GPU memory.
    use_swap = True
    num_layers = int(self.layer_scale * self.length)
    args = [cur0, it] + ([tf.zeros_like(cur0)] if self.output_layer_type == 1 else [])
    result = tf.while_loop(cond=lambda cur0, it, *_: it < num_layers,
                           body=self.looping_layer,
                           loop_vars=args,
                           parallel_iterations=1,
                           swap_memory=use_swap)
    if self.output_layer_type == 1:
      ans = result[-1]
    else:
      ans = result[0]
    return ans

  def looping_layer(self, cur0, index, *args):
    if self.output_layer_type == 1:
      output, = args

    def do_iteration(cur, output):
      old = cur
      if self.phase_train:
        cur = tf.nn.dropout(cur, self.dropout_keep_prob)
      cur = utils.gru_block(cur, self.kw, self.kh, self.nmaps,
                            self.cutoff, self.mask, 'lookup',
                            self.nconvs, extras=[])
      # TODO, if multiple stream, self.output_layer_type == 1
      if self.output_layer_type == 1:
        output += cur
      else:
        cur = tf.where(tf.greater_equal(self.output_layers, index + it), cur, old)
      return cur, output

    for it in range(self.rx_step):
      with tf.variable_scope("RX%d" % it) as vs:
        with tf.variable_scope("Frames"):
          cur0, output = do_iteration(cur0, output)

    # cur0 = slim.max_pool2d(cur0, [1, 3], stride=2)
    if self.output_layer_type == 1:
      return (cur0, index + self.rx_step, output)
    else:
      return (cur0, index + self.rx_step)
