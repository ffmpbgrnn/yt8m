import tensorflow as tf
from tensorflow import gfile

# for example in tf.python_io.tf_record_iterator("/data/uts700/linchao/yt8m/data/train/trainzv.tfrecord"):
  # print tf.train.Example.FromString(example)


def main(data_pattern, feature_names="rgb"):
  files = gfile.Glob(data_pattern)
  if not files:
    raise IOError("Unable to find files. data_pattern='" +
                  data_pattern + "'")
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
  with tf.Session() as sess:
    sess.run(sparse_labels)

main("/data/uts700/linchao/yt8m/data/train/trainzv.tfrecord")
