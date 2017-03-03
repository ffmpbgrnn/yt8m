import tensorflow as tf
from .readers import BaseReader


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
    _, serialized_examples = reader.read(filename_queue)

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

    if False:
      concatenated_features = tf.reshape(tf.cast(tf.decode_raw(features["feas"], tf.float16), tf.float32), [65536])
    else:
      concatenated_features = tf.reshape(tf.cast(tf.decode_raw(features["feas"], tf.float16), tf.float32), [256, 256, 1])
      concatenated_features = tf.tile(concatenated_features, [1, 1, 3])
      height, width = 299, 299
      concatenated_features = tf.image.resize_images(concatenated_features, [height, width], method=0)

    sparse_labels = features["labels"].values
    dense_labels = (tf.cast(
        tf.sparse_to_dense(sparse_labels, (self.num_classes,), 1,
            validate_indices=False),
        tf.float32))
    sparse_labels = dense_labels
    weights = dense_labels
    num_frames = tf.ones([1])

    batch_video_ids = tf.expand_dims(features["video_id"], 0)
    batch_video_matrix = tf.expand_dims(concatenated_features, 0)
    batch_dense_labels = tf.expand_dims(dense_labels, 0)
    batch_sparse_labels = tf.expand_dims(sparse_labels, 0)
    batch_label_weights = tf.expand_dims(weights, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    return batch_video_ids, batch_video_matrix, batch_dense_labels, batch_sparse_labels, batch_frames, batch_label_weights
