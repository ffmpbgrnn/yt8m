import numpy as np
import tensorflow as tf

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def matrix_to_tfexample(rgb, audio, labels=[], video_id=None):
  return tf.train.SequenceExample(
      context=tf.train.Features(feature={
          "labels": _int64_feature(labels),
          "video_id": _bytes_feature([video_id]),
      }),
      feature_lists=tf.train.FeatureLists(feature_list={
          "rgb": tf.train.FeatureList(feature=[
              tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb]))
              # tf.train.Feature(float_list=tf.train.FloatList(value=feat))
                # for input_ in inputs
          ]),
          "audio": tf.train.FeatureList(feature=[
              tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio]))
          ])
      })
  )

def write():
  output_filename = "/tmp/a.tfrecord"
  with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    d = np.array(np.random.rand(256, 1024), dtype=np.float32)
    print(d)
    example = matrix_to_tfexample(
        d.tostring(), labels=[1, 2], video_id="asf")
    tfrecord_writer.write(example.SerializeToString())

    d = np.array(np.random.rand(300, 1024), dtype=np.float32)
    print(d)
    example = matrix_to_tfexample(
        d.tostring(), labels=[1, 2], video_id="asf")
    tfrecord_writer.write(example.SerializeToString())

def read():
  feature_names = ["rgb", "audio"]
  filename_queue = tf.train.string_input_producer(["../../trainzy.tfrecord"],
                                                  shuffle=False, num_epochs=1)
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

  sparse_labels = contexts["labels"].values
  video_id = contexts["video_id"]
  rgb = features["rgb"]
  audio = features["audio"]
  rgb = tf.decode_raw(rgb, tf.uint8)
  audio = tf.decode_raw(audio, tf.uint8)

  '''
  rgb = tf.reshape(
        tf.cast(tf.decode_raw(rgb, tf.uint8), tf.float32),
        tf.cast(tf.decode_raw(rgb, tf.float32), tf.float32),
        [-1, 1024])
  '''

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # feas_val, labels, video_id = sess.run([feas, contexts["labels"], contexts["video_id"]])
    output_filename = "a.tfrecord"
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
    try:
      while True:
        video_id_v, sparse_labels_v, rgb_v, audio_v = sess.run([video_id, sparse_labels, rgb, audio])
        sparse_labels_v = sparse_labels_v.tolist()
        rgb_v = rgb_v.tostring()
        audio_v = audio_v.tostring()
        example = matrix_to_tfexample(
            rgb_v, audio_v, labels=sparse_labels_v, video_id=video_id_v)
        tfrecord_writer.write(example.SerializeToString())
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
  feature_names = ["rgb", "audio"]
  filename_queue = tf.train.string_input_producer(["a.tfrecord"],
  # filename_queue = tf.train.string_input_producer(["../../trainzy.tfrecord"],
                                                  shuffle=False, num_epochs=1)
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

  sparse_labels = contexts["labels"].values
  video_id = contexts["video_id"]
  rgb = features["rgb"]
  audio = features["audio"]

  rgb = tf.reshape(
        tf.cast(tf.decode_raw(rgb, tf.uint8), tf.float32),
        [-1, 1024])
  audio = tf.reshape(
        tf.cast(tf.decode_raw(audio, tf.uint8), tf.float32),
        [-1, 128])

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
# read()
just_read()
