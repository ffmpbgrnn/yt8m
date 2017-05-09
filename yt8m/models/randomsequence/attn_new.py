import tensorflow as tf


slim = tf.contrib.slim

def attn(hidden_states, fea_size, num_heads=5, seq_len=300):
  hidden_size = 50
  # hidden: batch_size x seq_len x fea_size
  batch_size = tf.shape(hidden_states)[0]

  with tf.variable_scope("Attention"):
    hidden_features = slim.conv2d(tf.reshape(hidden_states, [-1, seq_len, 1, fea_size]),
                                  hidden_size, [1, 1], activation_fn=None,
                                  scope="to_hidden")
    # hidden_features: batch_size x seq_len x hidden_size
    hidden_features = tf.reshape(hidden_features, [-1, seq_len, hidden_size])

    W = tf.get_variable("W", [1, num_heads, hidden_size], dtype=tf.float32)
    # W_tiled: batch_size, num_heads, hidden_size
    W_tiled = tf.tile(W, [batch_size, 1, 1])

    # s -> batch_size x num_heads x seq_len
    s =  tf.matmul(W_tiled, tf.tanh(hidden_features), transpose_b=True)
    a = tf.nn.softmax(s)

    # blended: batch_size x num_heads x fea_size
    blended = tf.matmul(a, hidden_states)
    return tf.reshape(blended, [-1, num_heads * fea_size])
