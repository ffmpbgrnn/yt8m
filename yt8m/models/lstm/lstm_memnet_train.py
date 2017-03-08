import tensorflow as tf

offset = 4716
def step(self, decoder_fn, sparse_labels):
  outputs, _ = decoder_fn(self, [sparse_labels], initial_state=self.init_state,
                          attention_states=self.model_input,
                          cell=self.dec_cell,
                          num_symbols=self.vocab_size,
                          embedding_size=512,
                          num_heads=4,
                          output_size=self.vocab_size+offset,
                          output_projection=None,
                          feed_previous=False,
                          dtype=tf.float32,
                          scope="LSTMMemNet")
  return outputs

def train(self, decoder_fn):
  outputs = step(self, decoder_fn, self.sparse_labels)
  logits = outputs[0]
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_labels, logits=logits)
  loss = tf.reduce_mean(loss)
  predictions = tf.nn.softmax(logits)[:, :self.vocab_size]
  return predictions, loss

def eval0(self, decoder_fn, linear_fn):
  self.w, self.b = self._get_vars([self.init_state], self.vocab_size+offset)
  output = linear_fn([self.init_state], self.w, self.b, True)[:, :self.vocab_size]
  _, top_idxs = tf.nn.top_k(
      output, k=40, name='nn_topk')

  loss = tf.constant(0.0)
  sparse_labels = tf.unstack(top_idxs, axis=1)
  predictions = []
  check_integraty = tf.zeros((self.runtime_batch_size, self.vocab_size), dtype=tf.float32)
  for i, sp_label in enumerate(sparse_labels):
    if i > 0:
      tf.get_variable_scope().reuse_variables()
    outputs = step(self, decoder_fn, sp_label)
    logits = outputs[0]
    pred = tf.nn.softmax(logits)[:, :self.vocab_size]
    # pred = tf.where(tf.equal(tf.cast(sp_label, dtype=tf.int64), tf.argmax(pred, axis=1)), pred, check_integraty)
    predictions.append(pred)
  predictions = tf.reduce_max(tf.stack(predictions, axis=2), axis=2)

  return predictions, loss

def eval1(self, decoder_fn, linear_fn):
  self.w, self.b = self._get_vars([self.init_state], self.vocab_size+offset)
  output = linear_fn([self.init_state], self.w, self.b, True)
  predictions = tf.nn.softmax(output)[:, :self.vocab_size]
  loss = tf.constant(0.0)
  return predictions, loss

def eval2(self, decoder_fn, linear_fn):
  preds = []
  self.w, self.b = self._get_vars([self.init_state], self.vocab_size+offset)
  one_sentinel = tf.ones((self.runtime_batch_size, 1))
  zero_sentinel = tf.zeros((self.runtime_batch_size, 1))
  # for num_splits in [1, 3, 6, 12, 25, 30, 60, 100]:
  for num_splits in [30]:
    splits = tf.split(self.model_input, num_or_size_splits=num_splits, axis=1)
    splits_weights = tf.split(self.input_weights_2d, num_or_size_splits=num_splits, axis=1)
    for idx, split in enumerate(splits):
      weight = splits_weights[idx]
      nf = tf.reduce_sum(weight, axis=1, keep_dims=True)
      nsum = tf.reduce_sum(nf, axis=1)

      safe_nf = tf.where(tf.equal(nsum, 0), x=one_sentinel, y=nf)
      split = tf.reduce_sum(split, axis=1) / safe_nf
      output = linear_fn([split], self.w, self.b, True)
      pred = tf.nn.softmax(output)[:, :self.vocab_size]
      # nf = tf.Print(nf, [nf[:, 0:1]])

      valid = tf.where(tf.equal(nsum, 0), x=zero_sentinel, y=one_sentinel)
      pred = pred * valid
      preds.append(pred)

  preds = tf.stack(preds, axis=2)
  predictions = tf.reduce_max(preds, axis=2)
  loss = tf.constant(0.0)
  return predictions, loss
  # predictions = tf.add_n(preds)

def eval3(self, decoder_fn, linear_fn):
  sparse_labels = tf.unstack(self.sparse_labels, axis=1)
  self.w, self.b = self._get_vars([self.init_state], self.vocab_size+offset)
  preds = []
  check_inta = tf.zeros((self.runtime_batch_size, self.vocab_size), dtype=tf.float32)
  for i, sp_label in enumerate(sparse_labels):
    if i > 0:
      tf.get_variable_scope().reuse_variables()
    outputs = step(self, decoder_fn, sp_label)
    output = tf.nn.softmax(outputs[0])
    # output = tf.where(tf.equal(tf.cast(sp_label, dtype=tf.int64), tf.argmax(output, axis=1)), output, check_inta)
    preds += [output]
  predictions = tf.reduce_max(tf.stack(preds, axis=2), axis=2)[:, :self.vocab_size]
  loss = tf.constant(0.0)
  return predictions, loss

def train0(self):
  if is_training or True:
    w, b = self._get_vars([init_state], vocab_size)
    self.w, self.b = w, b
    if False:
      pass
    elif True:
      output = _linear([init_state], w, b, True)
      # sparse_labels = tf.reshape(tf.argmax(output, axis=1), [-1])
      _, hint_pool_idxs = tf.nn.top_k(
          output, k=40, name='nn_topk'
      )

      sparse_labels = tf.unstack(hint_pool_idxs, axis=1)
      predictions = []
      runtime_batch_size = tf.shape(model_input)[0]
      check_inta = tf.zeros((runtime_batch_size, vocab_size), dtype=tf.float32)
      for i, sp_label in enumerate(sparse_labels):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        output, _ = embedding_attention_decoder(self, [sp_label], initial_state=init_state,
                                                 attention_states=model_input,
                                                 cell=dec_cell,
                                                 num_symbols=vocab_size,
                                                 embedding_size=512,
                                                 num_heads=1,
                                                 output_size=vocab_size,
                                                 output_projection=None,
                                                 feed_previous=False,
                                                 dtype=tf.float32,
                                                 scope="LSTMMemNet")
        output = tf.nn.softmax(output[0])
        # output = tf.where(tf.equal(tf.cast(sp_label, dtype=tf.int64), tf.argmax(output, axis=1)), check_inta, output,)
        predictions += [output]
      loss = tf.constant(0.0)
      predictions = tf.reduce_max(tf.stack(predictions, axis=2), axis=2)

eval = eval0
