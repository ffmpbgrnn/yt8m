import time
import numpy as np

import h5py
import os
import sys
import cPickle as pkl

import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging

import utils
import readers


def sample():


def main(stage, split_id=""):
  feature_names = "rgb"
  feature_sizes = "1024"
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      feature_names, feature_sizes)

  reader = readers.YT8MFrameFeatureReader(
      feature_names=feature_names,
      feature_sizes=feature_sizes,)

  data_pattern = "/data/uts700/linchao/yt8m/data/{0}/{0}*.tfrecord".format(stage)
  # data_pattern = "/data/uts700/linchao/yt8m/data/splits/{0}/{1}/{0}*.tfrecord".format(stage, split_id)
  # data_pattern = "/data/uts700/linchao/yt8m/data/splits/{0}/{1}/train*.tfrecord".format(stage, split_id)
  num_readers = 8
  batch_size = 128

  files = gfile.Glob(data_pattern)
  if not files:
    raise IOError("Unable to find the evaluation files.")
  filename_queue = tf.train.string_input_producer(
      files, shuffle=False, num_epochs=1)
  eval_data = [
      reader.prepare_reader(filename_queue) for _ in xrange(num_readers)
  ]

  # eval_data = reader.prepare_reader(filename_queue)
  video_id_batch, model_input_raw, labels_batch, num_frames = tf.train.batch_join(
      eval_data,
      batch_size=batch_size,
      capacity=3 * batch_size,
      allow_smaller_final_batch=True)

  # fout = h5py.File("/data/uts711/linchao/yt8m_hdfs/{}/split_{}.h5".format(stage, split_id), "w")
  # label_dict = {}
  all_videos_frames = 0
  all_videos_cnt = 0
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
        v_video_id, v_num_frames = sess.run([video_id_batch, num_frames])
        all_videos_frames += sum(v_num_frames)
        all_videos_cnt += len(v_video_id)
        if False:
          v_video_id, v_model_input_raw, v_labels_batch, v_num_frames = sess.run([video_id_batch, model_input_raw, labels_batch, num_frames])
          v_batch_size = len(v_video_id)
          for i in xrange(v_batch_size):
            fout.create_dataset(v_video_id[i], data=v_model_input_raw[i][0: v_num_frames[i], :], dtype=np.float16)
            label_dict[v_video_id[i]] = v_labels_batch[i]

        # print(v_model_input_raw.shape)
        # print(v_video_id)
        # print(v_labels_batch)
        # print(v_num_frames)
        # exit(0)
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = v_video_id.shape[0] / seconds_per_batch
        examples_processed += v_video_id.shape[0]

        cnt += 1

        if cnt % 5 == 0:
          print("examples processed: {}".format(examples_processed))

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
  print(all_videos_cnt)
  print(all_videos_frames)
  # fout.close()
  # pkl.dump(label_dict, open("/data/uts711/linchao/yt8m_hdfs/{}/label_{}.pkl".format(stage, split_id), "w"))

# main("train", sys.argv[1])
main("train")
