import numpy as np
import sys
import cPickle as pkl

import h5py
import tensorflow as tf
from tensorflow import gfile

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def matrix_to_tfexample(data, labels, video_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'feas': bytes_feature(data),
      'labels': int64_feature(labels),
      'video_id': bytes_feature(video_id),
  }))


def write():
  output_filename = "/tmp/a.tfrecord"
  d = np.array(np.random.rand(256, 256), dtype=np.float16)
  print(d)
  with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    example = matrix_to_tfexample(
        d.tostring(), [1, 2], video_id="asf")
    tfrecord_writer.write(example.SerializeToString())

def write_from_hdfs(split_id):
  train_vid_to_labels = pkl.load(open("/data/D2DCRC/linchao/YT/train_vid_to_labels.pkl"))
  fin = h5py.File("/data/D2DCRC/linchao/YT/vlad_feas_256_256_nonorm/train/feas_{0}.h5".format(split_id))
  vids = fin.keys()
  writer_counter = 0
  tfrecord_writer = None
  for i, vid in enumerate(vids):
    print(i, vid)
    data = fin[vid].value
    data = np.array(data, dtype=np.float16)
    labels = [int(_) for _ in train_vid_to_labels[vid]]
    example = matrix_to_tfexample(
        data.tostring(), labels, video_id=str(vid))
    if writer_counter % 1200 == 0:
      if tfrecord_writer is not None:
        tfrecord_writer.close()
      shard_id = int(writer_counter / 1200)
      # /data/D2DCRC/linchao/YT/vlad_feas_256_256_nonorm/train_tfrecord
      tfrecord_writer = tf.python_io.TFRecordWriter("/data/uts711/linchao/vlad_feas_256_256_nonorm/train_tfrecord/train_{0}_{1}.tfrecord".format(split_id, shard_id))
    tfrecord_writer.write(example.SerializeToString())
    writer_counter += 1

write_from_hdfs(sys.argv[1])

def read(output_filename):
  files = gfile.Glob(output_filename)
  filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  keys_to_features={
      'video_id':
          tf.FixedLenFeature(
              (), tf.string, default_value=''),
      'labels':
          tf.VarLenFeature(tf.int64),
      'feas':
          tf.FixedLenFeature(
              (), tf.string, default_value=''),
  }
  contexts = tf.parse_single_example(
      serialized_example, keys_to_features)
  feas = contexts["feas"]
  feas = tf.reshape(tf.decode_raw(feas, tf.float32), [-1])

  dense_labels = (tf.cast(
          tf.sparse_to_dense(contexts["labels"].values, (10,), 1,
              validate_indices=False),
          tf.bool))

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # feas_val, labels, video_id = sess.run([feas, contexts["labels"], contexts["video_id"]])
    feas_val, labels, video_id = sess.run([feas, dense_labels, contexts["video_id"]])
    print(feas_val.shape)
    print(np.reshape(feas_val, [256, 256]))
    print(labels)
    print(video_id)
    # print(sess.run(labels))
    coord.request_stop()
    coord.join(threads)

# for example in tf.python_io.tf_record_iterator(output_filename):
  # d = tf.train.Example.FromString(example)
