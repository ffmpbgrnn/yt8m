import time

import h5py
import os
import sys

import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging

import utils
import readers


class HDFSConverter():
  def __init__(self):
    self.feature_names = "mean_rgb"
    self.feature_sizes = "1024"
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        self.feature_names, self.feature_sizes)

    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names,
        feature_sizes=feature_sizes)

    data_pattern = ""
    num_readers = 4
    batch_size = 32

    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in xrange(num_readers)
    ]

    video_id_batch, model_input_raw, labels_batch, num_frames = tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True)

    with tf.Session() as sess:
      sess.run([tf.local_variables_initializer()])
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(
              sess, coord=coord, daemon=True,
              start=True))

        examples_processed = 0
        cnt = 0
        while not coord.should_stop():
          batch_start_time = time.time()
          v_video_id, v_model_input_raw, v_num_frames = sess.run([video_id_batch, model_input_raw, num_frames])
          seconds_per_batch = time.time() - batch_start_time
          example_per_second = v_model_input_raw.shape[0] / seconds_per_batch
          examples_processed += v_model_input_raw.shape[0]

          cnt += 1

          if cnt % 5 == 0:
            print("examples processed: {}".format(example_per_second))

      except tf.errors.OutOfRangeError as e:
        logging.info(
            "Done with batched inference. Now calculating global performance "
            "metrics.")

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
