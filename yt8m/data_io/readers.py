# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import numpy as np
import tensorflow as tf
from yt8m import utils

from tensorflow import logging

def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

def sort_and_pad(x):
  # TODO
  x = np.sort(x)[::-1]
  x = x.tolist()
  w = np.ones((self.num_max_labels), dtype=np.int64)
  w[-1] = 0
  caps_len = self.num_max_labels - 2
  if len(x) > caps_len:
    s = np.random.randint(len(x) - caps_len + 1)
    x = [self.sos_id] + x[s: s+caps_len] + [self.eos_id]
  else:
    x = [self.sos_id] + x + [self.eos_id]
    num_pad = self.num_max_labels - len(x)
    x = x + [self.pad_id] * num_pad
    w[-1 * num_pad - 1:] = 0

    # pad = np.zeros((self.num_max_labels,), dtype=np.int64)
    # pad[: len(x)] = x
    # w[x.shape[0]:] = 0
    # x = pad
  return (x, w)

def random_pick_one(x):
  x = x.tolist()
  # TODO
  if np.random.randint(5) == 0:
    exclude_x = list(self.classes - set(x))
    x = [exclude_x[np.random.randint(len(exclude_x))],]
    w = [self.num_classes + x[0]]
    # w = [1,]
  else:
    idx = np.random.randint(len(x))
    x = [x[idx],]
    w = x
    # w = [1,]
  return (x, w)

def gen_sparse_label(x):
  return_x = np.zeros((4716), dtype=np.int64) + 4716
  return_w = np.zeros((4716), dtype=np.float32) + 1/4716.
  x = x.tolist()
  for i in x:
    return_x[i] = i
    return_w[i] = 1.
  return (return_x, return_w)

def gen_sparse_label_batch(x):
  return_x = np.zeros((x.shape[0], 4716), dtype=np.int64) + 4716
  return_w = np.zeros((x.shape[0], 4716), dtype=np.float32) + 3/4716.
  indicator = np.where(x == 1)
  r, c = indicator[0], indicator[1]
  for i in xrange(r.shape[0]):
    return_x[r[i], c[i]] = c[i]
    return_w[r[i], c[i]] = 1.
  return (return_x, return_w)

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"],
               num_max_labels=-1):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.num_max_labels = num_max_labels

  def prepare_reader(self, filename_queue, batch_size=1024):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

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
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    # TODO
    # labels = tf.zeros((self.num_classes), dtype=tf.float32)
    labels.set_shape([None, self.num_classes])

    concatenated_features = tf.concat([
        features[feature_name] for feature_name in self.feature_names], 1)
    sparse_labels, label_weights, input_weights = labels, labels, labels

    # sparse_labels = features["labels"].values
    if self.num_max_labels == 4716:
      sparse_labels, label_weights = tf.py_func(gen_sparse_label_batch, [labels], [tf.int64, tf.float32])
      sparse_labels = tf.reshape(sparse_labels, [-1, self.num_max_labels])
      label_weights = tf.reshape(label_weights, [-1, self.num_max_labels])

    return features["video_id"], concatenated_features, labels, sparse_labels, tf.ones([tf.shape(serialized_examples)[0]]), label_weights, input_weights

class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["inc3"],
               max_frames=300,
               num_max_labels=-1):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames
    self.num_max_labels = num_max_labels
    self.pad_id = self.num_classes
    self.sos_id = self.num_classes + 1
    self.eos_id = self.num_classes + 2
    self.classes = set(range(self.num_classes))

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })

    # read ground truth labels
    sparse_labels = contexts["labels"].values
    dense_labels = (tf.cast(
        tf.sparse_to_dense(sparse_labels, (self.num_classes,), 1,
            validate_indices=False),
        tf.bool))
    if self.num_max_labels == 1:
      sparse_labels, label_weights = tf.py_func(random_pick_one, [sparse_labels], [tf.int64, tf.int64])
      sparse_labels = tf.reshape(sparse_labels, [1,])
      label_weights = tf.reshape(label_weights, [1,])
    elif self.num_max_labels > 0:
      if self.num_max_labels == 4716:
        sparse_labels, label_weights = tf.py_func(gen_sparse_label, [sparse_labels], [tf.int64, tf.float32])
        sparse_labels = tf.reshape(sparse_labels, [self.num_max_labels])
        label_weights = tf.reshape(label_weights, [self.num_max_labels])
      else:
        sparse_labels, label_weights = tf.py_func(sort_and_pad, [sparse_labels], [tf.int64, tf.int64])
        sparse_labels = tf.reshape(sparse_labels, [self.num_max_labels])
        label_weights = tf.reshape(label_weights, [self.num_max_labels])
    else:
      sparse_labels = dense_labels
      label_weights = tf.constant(1, dtype=tf.int64)

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index],
          self.max_frames,
          max_quantized_value,
          min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature
      else:
        tf.assert_equal(num_frames, num_frames_in_this_feature)

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # convert to batch format.
    # TODO: Do proper batch reads to remove the IO bottleneck.
    batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_dense_labels = tf.expand_dims(dense_labels, 0)
    batch_sparse_labels = tf.expand_dims(sparse_labels, 0)
    batch_label_weights = tf.expand_dims(label_weights, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    input_weights = tf.ones([num_frames,], dtype=tf.float32)
    input_weights = tf.pad(
        input_weights,
        [[0, self.max_frames - num_frames]],
        "CONSTANT")
    input_weights.set_shape([self.max_frames])
    batch_input_weights = tf.expand_dims(input_weights, 0)

    return batch_video_ids, batch_video_matrix, batch_dense_labels, batch_sparse_labels, \
           batch_frames, batch_label_weights, batch_input_weights
