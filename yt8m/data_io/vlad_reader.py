import tensorflow as tf
from reader import BaseReader


class YT8MVLADFeatureReader(BaseReader):
  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["feas"]):
    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue)

    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64)}
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          (), tf.string)

    features = tf.parse_single_example(serialized_examples, features=feature_map)

    concatenated_features = tf.reshape(tf.cast(tf.decode_raw(features["feas"], tf.float16), tf.float32), [1, -1])

    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    sparse_labels, weights = labels, labels
    return features["video_id"], concatenated_features, labels, sparse_labels, tf.ones([tf.shape(serialized_examples)[0]]), weights
