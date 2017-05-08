import tensorflow as tf
from . import utils

def one_layer(self, inputs):
  return tf.tanh(
      utils.conv_linear(inputs, 1, 1, self.nmaps, "input"))

class ConvGRU(object):
  def __init__(self):
    # self.height = 3
    # self.vec_size = 1536
    self.max_grad_norm = 5.
    self.lr = 3e-4

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

    self.noclass = 4716
    self.num_out_length = 300

  def create_model(self, model_input, vocab_size, num_frames,
                   is_training=True, dense_labels=None, **unused_params):
    self.phase_train = is_training
    self.mask
    self.run_time_batch_size = tf.shape(model_input)[0]
    with tf.variable_scope("Frames"):
      frame_layer_input = self.one_layer(model_input)
    self.last_layer = self.construct_all_layers(frame_layer_input, self.mask)

    logits = tf.reduce_mean(self.last_layer, [2], keep_dims=True)
    logits = utils.conv_linear(logits, 1, 1, self.nmaps, "output0")
    logits = tf.nn.relu(logits)
    logits = tf.reduce_max(logits, [1], keep_dims=True)

    # layer_output = utils.conv_linear(logits, 1, 1, self.noclass, "output0")
    output_no_softmax = tf.squeeze(logits, [1, 2])
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_no_softmax, labels=self.targets)
    perp_loss = tf.reduce_mean(xent)

    self.probs = utils.softmax(output_no_softmax)
    metric_predictions = tf.argmax(output_no_softmax, 1)
    metric_labels = tf.squeeze(self.targets)

    total_loss = perp_loss

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
