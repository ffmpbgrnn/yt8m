import tensorflow as tf
from yt8m.models import models
from tensorflow.contrib.rnn.python.ops import gru_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.python.layers.core import Dense
from . import attention_wrapper

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

def SampleRandomFrames(model_input, num_frames, num_samples):
  batch_size = tf.shape(model_input)[0]
  frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, num_samples]),
          tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)

def SampleRandomSequence(model_input, num_frames, num_samples):
  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)

def SampleRandomSequencePad0(model_input, input_weights, num_frames, num_samples):
  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames, tf.int32))
  frame_index = tf.minimum(frame_index,
                           tf.cast(300 - 1, tf.int32))
  num_sampled_frames = frame_index[:, -1] - frame_index[:, 0] + 1
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index), tf.gather_nd(input_weights, index), num_sampled_frames


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


def reconstruct_loss(logit, target):
  # Huber loss
  sigma = 2.
  delta = sigma * sigma
  d = logit - target
  if True:
    a = .5 * delta * d * d
    b = tf.abs(d) - 0.5 / delta
    l = tf.where(tf.abs(d) < (1. / delta), a, b)
  else:
    l = .5 * d * d
  # loss = tf.reduce_sum(d * d, reduction_indices=1)
  loss = tf.reduce_sum(l, axis=2)
  return loss


def h_gru(model_input, vocab_size, is_training=True):
  with tf.variable_scope("EncLayer0"):
    first_enc_cell = core_rnn_cell.DeviceWrapper(
        gru_ops.GRUBlockCell(1024),
        device='/gpu:1')
    runtime_batch_size = tf.shape(model_input)[0]
    enc_init_state = tf.zeros((runtime_batch_size, 1024), dtype=tf.float32)
    num_splits = 15
    model_input_splits = tf.split(model_input, num_or_size_splits=num_splits, axis=1)
    enc_state = None
    first_layer_outputs = []
    for i in xrange(num_splits):
      if i == 0:
        initial_state = enc_init_state
      else:
        initial_state = enc_state
        tf.get_variable_scope().reuse_variables()
      initial_state = tf.stop_gradient(initial_state)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(
          first_enc_cell, model_input_splits[i], initial_state=initial_state, scope="enc0")
      # TODO
      enc_state = moe_layer(enc_state, 1024, 4, act_func=None, l2_penalty=1e-12)
      if is_training:
        enc_state = tf.nn.dropout(enc_state, 0.5)
      first_layer_outputs.append(enc_state)

  with tf.variable_scope("EncLayer1"):
    second_enc_cell = core_rnn_cell.DeviceWrapper(
        gru_ops.GRUBlockCell(1024),
        device='/gpu:1')
    first_layer_outputs = tf.stack(first_layer_outputs, axis=1)
    enc_outputs, enc_state = tf.nn.dynamic_rnn(
        second_enc_cell, first_layer_outputs, dtype=tf.float32, scope="enc1")
  # TODO
  if is_training:
    enc_state = tf.nn.dropout(enc_state, 0.8)
  logits = moe_layer(enc_state, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
  return logits

class ContextModel(models.BaseModel):
  def __init__(self):
    super(ContextModel, self).__init__()
    self.normalize_input = False
    self.clip_global_norm = 5
    self.var_moving_average_decay = 0.9997
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 3e-4
    # self.decay_lr = True


  def create_model(self, model_input, vocab_size, num_frames,
                  is_training=True, dense_labels=None, input_weights=None, **unused_params):
    self.model_input = model_input
    self.vocab_size = vocab_size
    self.dense_labels = dense_labels
    self.num_frames = num_frames
    self.is_training = is_training
    self.input_weights = input_weights

    return self.pretrain()

  def do_job(self):
    first_layer_outputs = []
    num_splits = 15
    context_frames = SampleRandomSequence(model_input, num_frames, 50)
    cell = gru_ops.GRUBlockCell(1024)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, vocab_size)

    with tf.variable_scope("EncLayer0"):
      cell = gru_ops.GRUBlockCell(1024)
      for i in xrange(num_splits):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell, frames, scope="enc0")
        enc_state = moe_layer(enc_state, 1024, 4, act_func=None, l2_penalty=1e-12)
        if is_training:
          enc_state = tf.nn.dropout(enc_state, 0.5)
        first_layer_outputs.append(enc_state)

    with tf.variable_scope("EncLayer1"):
      cell = gru_ops.GRUBlockCell(1024)
      first_layer_outputs = tf.stack(first_layer_outputs, axis=1)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(
          cell, first_layer_outputs, scope="enc1")

    flatten_outputs = tf.reduce_mean(enc_outputs, axis=1)

    with tf.variable_scope("FC0"):
      flatten_outputs = moe_layer(flatten_outputs, 1024, 2, act_func=tf.nn.relu, l2_penalty=1e-8)
    if is_training:
      flatten_outputs = tf.nn.dropout(flatten_outputs, 0.5)
    with tf.variable_scope("FC1"):
      logits = moe_layer(flatten_outputs, vocab_size, 2, act_func=tf.nn.sigmoid, l2_penalty=1e-8)
    logits = tf.clip_by_value(logits, 0., 1.)
    return {"predictions": logits}

  def get_pretrain_enc_cell(self,):
    cell = gru_ops.GRUBlockCell(1024)
    if self.is_training:
      cell = core_rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
    cell = core_rnn_cell.InputProjectionWrapper(cell, 1024)
    cell = core_rnn_cell.OutputProjectionWrapper(cell, 1024)

    cell = core_rnn_cell.DeviceWrapper(cell, device='/gpu:0')
    return cell

  def pretrain(self):

    def do_cls(input_weights, num_frames):
      enc_outputs_stopped = tf.stop_gradient(enc_outputs)
      input_weights = tf.tile(
          tf.expand_dims(input_weights, 2),
          [1, 1, self.cell_size])
      enc_outputs_stopped = enc_outputs_stopped * input_weights
      enc_rep = tf.reduce_sum(enc_outputs_stopped, axis=1) / num_frames
      # enc_rep = tf.reduce_sum(enc_outputs_stopped, axis=1) / self.max_steps

      cls_func = self.moe
      logits = cls_func(enc_rep)

      if cls_func == self.moe:
        epsilon = 1e-12
        labels = tf.cast(self.dense_labels, tf.float32)
        cross_entropy_loss = labels * tf.log(logits + epsilon) + (
            1 - labels) * tf.log(1 - logits + epsilon)
        cross_entropy_loss = tf.negative(cross_entropy_loss)
        loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

        predictions = logits
      else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.dense_labels, tf.float32),
                                                      logits=logits)
        loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
        predictions = tf.nn.sigmoid(logits)
      return predictions

    def do_reconstruction(enc_inputs, enc_outputs, enc_last_state, input_weights, seq_lengths):
      num_units = 100
      # attn_mech = attention_wrapper.LuongAttention(
          # num_units=num_units,
          # memory=enc_outputs,
          # memory_sequence_length=seq_lengths,
          # scale=True)
      attn_mech = tf.contrib.seq2seq.BahdanauAttention(
          num_units=num_units,
          memory=enc_outputs,
          memory_sequence_length=seq_lengths,
          normalize=True,
          name='attention_mechanism')
      cell = gru_ops.GRUBlockCell(1024)
      cell = core_rnn_cell.DropoutWrapper(cell, 0.5, 0.5)
      attn_cell = tf.contrib.seq2seq.AttentionWrapper(
          cell=cell,
          attention_mechanism=attn_mech,
          attention_layer_size=1024,
          output_attention=False,
          initial_cell_state=enc_last_state,
          name="attention_wrapper")

      decoder_target = tf.reverse_sequence(
          enc_inputs,
          seq_lengths,
          seq_dim=1,
          batch_dim=0)
      decoder_inputs = tf.pad(decoder_target[:, :-1, :], [[0, 0], [1, 0], [0, 0]])

      helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=decoder_inputs, # decoder inputs
          sequence_length=seq_lengths, # decoder input length
          name="decoder_training_helper")

      # Decoder setup
      decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=attn_cell,
          helper=helper,
          initial_state=attn_cell.zero_state(tf.shape(enc_inputs)[0], dtype=tf.float32),
          output_layer=Dense(1024+128))
      # Perform dynamic decoding with decoder object
      dec_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory=True,)
      loss = reconstruct_loss(logit=dec_outputs.rnn_output, target=decoder_target)
      # input_weights = tf.cast(input_weights, tf.float32)
      loss = tf.reduce_sum(loss * input_weights, axis=1) / tf.cast(seq_lengths, tf.float32)
      loss = tf.reduce_mean(loss)
      # loss = tf.contrib.seq2seq.sequence_loss(
          # dec_outputs.rnn_output, decoder_target, input_weights,
          # softmax_loss_function=reconstruct_loss)
      predictions = tf.no_op()
      return predictions, loss

    do_pretrain = False
    # TODO
    if do_pretrain:
      self.num_frames = tf.cast(tf.expand_dims(self.num_frames, 1), tf.float32)
      self.model_input, self.input_weights, self.num_frames = SampleRandomSequencePad0(
          self.model_input, self.input_weights, self.num_frames, 50)

    enc_inputs = self.model_input
    with tf.device("/gpu:0"):
      enc_cell = self.get_pretrain_enc_cell()
      self.model_variables = tf.trainable_variables()

      enc_outputs, enc_state = tf.nn.dynamic_rnn(
          enc_cell, enc_inputs, dtype=tf.float32, scope="enc")
      enc_outputs = tf.stop_gradient(enc_outputs)

    output_dict = {}
    if do_pretrain:
      predictions, loss = do_reconstruction(enc_inputs, enc_outputs,
                                            enc_last_state=enc_state,
                                            input_weights=self.input_weights,
                                            seq_lengths=self.num_frames)
      output_dict["predictions"] = predictions
      output_dict["loss"] = loss
    else:
      with tf.device("/gpu:0"):
        model_input = slim.conv2d(
            tf.reshape(self.model_input, [-1, 300, 1, 1024 + 128]),
            1024, [1, 1], activation_fn=None, scope="input_proj")
        model_input = tf.reshape(model_input, [-1, 300, 1024])
        inputs = enc_outputs + model_input
        inputs = tf.nn.l2_normalize(inputs, 2)
      with tf.device("/gpu:1"):
        predictions = h_gru(inputs, self.vocab_size, self.is_training)
      output_dict["predictions"] = predictions
    return output_dict

  def moe(self, enc_rep):
    pass
      # moe = video_level_models.MoeModel()
      # res = moe.create_model(
          # enc_rep, self.vocab_size,
          # num_mixtures=10)
      # return res["predictions"]


  def get_variables_with_ckpt(self):
    exclusions = False # "OutputFC"
    if self.is_training:
      variable_to_restore = []
      for var in self.model_variables:
        excluded = False
        for exclusion in exclusions:
          if var.op.name.startswith(exclusion):
            excluded = True
            break
        if not excluded:
          variable_to_restore.append(var)
          print(var.op.name)
      return variable_to_restore
    else:
      return tf.all_variables()
