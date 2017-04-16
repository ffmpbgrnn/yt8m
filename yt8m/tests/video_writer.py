import numpy as np
import tensorflow as tf
from tensorflow import gfile

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def matrix_to_tfexample(rgb, audio, labels=[], video_id=None):
  return tf.train.Example(
      features=tf.train.Features(feature={
          "labels": _int64_feature(labels),
          "video_id": _bytes_feature([video_id]),
          "mean_rgb": _float_feature(rgb),
          "mean_audio": _float_feature(audio),
      }),
  )

def write():
  output_filename = "/tmp/a.tfrecord"
  with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    rgb = np.array(np.random.rand(1024), dtype=np.float32)
    audio = np.array(np.random.rand(128), dtype=np.float32)
    print(rgb)
    example = matrix_to_tfexample(
        rgb, audio, labels=[1, 2], video_id="asf")
    tfrecord_writer.write(example.SerializeToString())

    # d = np.array(np.random.rand(1024), dtype=np.float32)
    # print(d)
    # example = matrix_to_tfexample(
        # d.tostring(), labels=[1, 2], video_id="asf")
    # tfrecord_writer.write(example.SerializeToString())

def read():
  stage = "validate"
  output_prefix = "/data/uts700/linchao/yt8m/data/video_level_25/{}".format(stage)
  files = gfile.Glob("/data/uts700/linchao/yt8m/data/video_level/video_level/{}/*.tfrecord".format(stage))
  filename_queue = tf.train.string_input_producer(files,
                                                  shuffle=False, num_epochs=1)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          "video_id": tf.FixedLenFeature([], tf.string),
          "labels": tf.VarLenFeature(tf.int64),
          "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
          "mean_audio": tf.FixedLenFeature([128], tf.float32)
      })

  sparse_labels = features["labels"].values
  video_id = features["video_id"]
  rgb = features["mean_rgb"]
  audio = features["mean_audio"]

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    cnt = 0
    output_filename = "{}/{}.tfrecord".format(output_prefix, cnt / 1200)
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
    target_labels = set(range(25))
    try:
      while True:
        video_id_v, sparse_labels_v, rgb_v, audio_v = sess.run([video_id, sparse_labels, rgb, audio])
        sparse_labels_v = sparse_labels_v.tolist()

        sparse_labels_v = target_labels.intersection(set(sparse_labels_v))
        if len(sparse_labels_v) == 0:
          continue
        rgb_v = rgb_v
        audio_v = audio_v
        example = matrix_to_tfexample(
            rgb_v, audio_v, labels=sparse_labels_v, video_id=video_id_v)
        tfrecord_writer.write(example.SerializeToString())
        if cnt % 1200 == 0:
          tfrecord_writer.close()
          output_filename = "{}/{}.tfrecord".format(output_prefix, cnt / 1200)
          tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
        cnt += 1
        # print(video_id_v, sparse_labels_v)
        # print(rgb_v)
        # print(rgb_v.shape)
        # print(audio_v)
        # print(rgb_v.shape)
    except tf.errors.OutOfRangeError:
      print("Done")
    tfrecord_writer.close()
    coord.request_stop()
    coord.join(threads)

def just_read():
  filename_queue = tf.train.string_input_producer(["/tmp/a.tfrecord"],
  # filename_queue = tf.train.string_input_producer(["/Users/ffmpbgrnn/Works/yt/src/trainZy.tfrecord"],
                                                  shuffle=False, num_epochs=1)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          "video_id": tf.FixedLenFeature([], tf.string),
          "labels": tf.VarLenFeature(tf.int64),
          "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
          "mean_audio": tf.FixedLenFeature([128], tf.float32),
      })

  sparse_labels = features["labels"].values
  video_id = features["video_id"]
  rgb = features["mean_rgb"]
  audio = features["mean_audio"]

  rgb = tf.reshape(rgb, [1024])
  audio = tf.reshape(audio, [128])

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    video_id_v, sparse_labels_v, rgb_v, audio_v = sess.run([video_id, sparse_labels, rgb, audio])
    print(video_id_v)
    print(sparse_labels_v)
    print(rgb_v)
    print(audio_v)
    coord.request_stop()
    coord.join(threads)

# write()
print("------------------------")
read()
# just_read()
