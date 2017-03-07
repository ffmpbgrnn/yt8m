def train(self, decoder_fn):
  outputs, _ = decoder_fn(self, [self.sparse_labels], initial_state=self.init_state,
                          attention_states=self.model_input,
                          cell=dec_cell,
                          num_symbols=self.vocab_size+1,
                          embedding_size=512,
                          num_heads=1,
                          output_size=self.vocab_size+1,
                          output_projection=None,
                          feed_previous=False,
                          dtype=tf.float32,
                          scope="LSTMMemNet")

  logits = outputs[0]
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sparse_labels, logits=logits)
  loss = tf.reduce_mean(loss)
  predictions = tf.nn.softmax(logits)[:self.vocab_size]
  return predictions, loss

def train0(self):
  if is_training or True:
    w, b = self._get_vars([init_state], vocab_size)
    self.w, self.b = w, b
    if False:
      sparse_labels = tf.unstack(sparse_labels, axis=1)
      outputs = []
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
        # output = tf.where(tf.equal(tf.cast(sp_label, dtype=tf.int64), tf.argmax(output, axis=1)), output, check_inta)
        outputs += [output]
      predictions = tf.reduce_max(tf.stack(outputs, axis=2), axis=2)
      loss = tf.constant(0.0)
      # predictions = dense_labels
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
      # predictions = []
      # for o in outputs:
        # predictions.append(tf.nn.softmax(o))
      predictions = tf.reduce_max(tf.stack(predictions, axis=2), axis=2)
      # predictions = tf.nn.sigmoid(predictions)
    else:
      outputs, _ = embedding_attention_decoder(self, [sparse_labels], initial_state=init_state,
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

      logits = outputs[0]
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels, logits=logits)
      loss = tf.reduce_mean(loss)
      predictions = tf.nn.softmax(logits)
  else:
    loss = tf.constant(0.0)
    runtime_batch_size = tf.shape(model_input)[0]
    with variable_scope.variable_scope("LSTMMemNet", dtype=tf.float32):
      with variable_scope.variable_scope("attention_decoder", dtype=tf.float32):
        with variable_scope.variable_scope("AttnOutputProjection"):
          weights, biases = _get_linear_vars([tf.reduce_sum(model_input, axis=1)], vocab_size, True,)
          '''
          input_weights = tf.reduce_sum(input_weights, axis=1)
          model_input = tf.reduce_sum(model_input, axis=1) / input_weights
          output = linear([model_input], vocab_size, True)
          predictions = tf.nn.softmax(output)
          '''
          safe_sentinel = tf.ones((runtime_batch_size, input_size))
          preds = []
          # for num_splits in [1, 3, 6, 12, 25, 30, 60, 100]:
          for num_splits in [300]:
            splits = tf.split(model_input, num_or_size_splits=num_splits, axis=1)
            splits_weights = tf.split(input_weights, num_or_size_splits=num_splits, axis=1)
            for idx, split in enumerate(splits):
              weight = splits_weights[idx]
              nf = tf.reduce_sum(weight, axis=1)
              nsum = tf.reduce_sum(nf, axis=1)

              safe_nf = tf.where(tf.equal(nsum, 0), x=nf, y=safe_sentinel)
              split = tf.reduce_sum(split, axis=1) / safe_nf
              output = _linear([split], weights, biases, True)
              pred = tf.nn.softmax(output)
              # nf = tf.Print(nf, [nf[:, 0:1]])
              pred = pred * tf.tile(nf[:, 0:1], [1, vocab_size])
              preds.append(pred)

          preds = tf.stack(preds, axis=2)
          predictions = tf.reduce_max(preds, axis=2)
          # predictions = tf.add_n(preds)
