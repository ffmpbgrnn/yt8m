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

    if True:
      input_features = tf.reshape(tf.cast(tf.decode_raw(features["feas"], tf.float16), tf.float32), [256, 256])
      input_features = tf.sign(input_features) * tf.sqrt(tf.abs(input_features))
      input_features = tf.nn.l2_normalize(input_features, 1)
      input_features = tf.reshape(input_features, [256*256])
      # input_features = tf.nn.l2_normalize(input_features, axis=0)
    else:
      input_features = tf.reshape(tf.cast(tf.decode_raw(features["feas"], tf.float16), tf.float32), [256, 256, 1])
      # input_features = tf.tile(input_features, [1, 1, 3])
      input_features = tf.pad(input_features, [[0, 0], [0, 0], [0, 2]], "CONSTANT")
      height, width = 224, 224
      input_features = tf.image.resize_images(input_features, [height, width], method=0)

    sparse_labels = features["labels"].values
    dense_labels = (tf.cast(
        tf.sparse_to_dense(sparse_labels, (self.num_classes,), 1,
            validate_indices=False),
        tf.float32))
    sparse_labels = dense_labels
    weights = dense_labels
    num_frames = tf.ones([1])

    batch_video_ids = tf.expand_dims(features["video_id"], 0)
    batch_video_matrix = tf.expand_dims(input_features, 0)
    batch_dense_labels = tf.expand_dims(dense_labels, 0)
    batch_sparse_labels = tf.expand_dims(sparse_labels, 0)
    batch_label_weights = tf.expand_dims(weights, 0)
    batch_frames = tf.expand_dims(num_frames, 0)
    batch_input_weights = batch_label_weights

    # batch_dense_labels = batch_dense_labels[:, 0: 1]
    return batch_video_ids, batch_video_matrix, batch_dense_labels, batch_sparse_labels, batch_frames, batch_label_weights, batch_input_weights
