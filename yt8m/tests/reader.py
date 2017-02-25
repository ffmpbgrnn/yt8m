import numpy as np

import tensorflow as tf
from tensorflow import app

from tensorflow import gfile

# for example in tf.python_io.tf_record_iterator("/data/uts700/linchao/yt8m/data/train/trainzv.tfrecord"):
  # print tf.train.Example.FromString(example)

def main(_):
  data_pattern = "/data/uts700/linchao/yt8m/data/train/trainzv.tfrecord"
  feature_names = ["rgb"]
  files = gfile.Glob(data_pattern)
  if not files:
    raise IOError("Unable to find files. data_pattern='" +
                  data_pattern + "'")
  print(files)
  filename_queue = tf.train.string_input_producer(
      files, shuffle=False, num_epochs=1)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  contexts, features = tf.parse_single_sequence_example(
      serialized_example,
      context_features={"video_id": tf.FixedLenFeature(
          [], tf.string),
                        "labels": tf.VarLenFeature(tf.int64)},
      sequence_features={
          feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
          for feature_name in feature_names
      })

  # read ground truth labels
  sparse_labels = contexts["labels"].values
  # sparse_labels = tf.gather(sparse_labels, tf.nn.top_k(sparse_labels, k=3,).indices)

  num_max_labels = 10
  def sort_and_pad_old(x):
    x = np.sort(x)
    w = np.ones((num_max_labels), dtype=np.int64)
    if x.shape[0] > num_max_labels:
      s = np.random.randint(x.shape[0] - num_max_labels + 1)
      x = x[s:]
    else:
      pad = np.zeros((num_max_labels), dtype=np.int64)
      pad[: x.shape[0]] = x
      w[x.shape[0]:] = 0
      x = pad
    return (x, w)
  sos_id, eos_id, pad_id = 10000, 10001, 10002
  def sort_and_pad(x):
    x = np.sort(x)
    x = x.tolist()
    w = np.ones((num_max_labels), dtype=np.int64)
    w[-1] = 0
    caps_len = num_max_labels - 2
    if len(x) > caps_len:
      s = np.random.randint(len(x) - caps_len + 1)
      x = [sos_id] + x[s: s+caps_len] + [eos_id]
    else:
      x = [sos_id] + x + [eos_id]
      num_pad = num_max_labels - len(x)
      x = x + [pad_id] * num_pad
      w[-1 * num_pad - 1:] = 0

      # pad = np.zeros((num_max_labels,), dtype=np.int64)
      # pad[: len(x)] = x
      # w[x.shape[0]:] = 0
      # x = pad
    return (x, w)

  sparse_labels, label_weights = tf.py_func(sort_and_pad, [sparse_labels], [tf.int64, tf.int64])

  # labels = (tf.cast(
      # tf.sparse_to_dense(sparse_labels, (4718,), 1,
          # validate_indices=False),
      # tf.bool))
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run([sparse_labels, label_weights]))
    # print(sess.run(labels))
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
  app.run()
