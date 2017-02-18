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

class HDFS():
  def __init__(self, stage, split_id):
    self.stage = stage
    self.split_id = split_id
    self.fout = h5py.File("/data/uts711/linchao/yt8m_hdfs/{}/split_{}.h5".format(
        self.stage, self.split_id), "w")
    self.label_dict = {}

  '''
  inputs: video_id_batch, model_input_raw, labels_batch, num_frames
  '''
  def run(self, sess, inputs):
    res = sess.run(inputs)
    v_batch_size = len(res['video_id'])
    for i in xrange(v_batch_size):
      self.fout.create_dataset(
          res['video_id'][i],
          data=res['input_raw'][i][0: res['num_frames'][i], :],
          dtype=np.float16)
      self.label_dict[res['video_id'][i]] = res['labels'][i]

  def done(self):
    self.fout.close()
    pkl.dump(self.label_dict, open("/data/uts711/linchao/yt8m_hdfs/{}/label_{}.pkl".format(
        self.stage, self.split_id), "w"))

class Stats():
  def __init__(self):
    self.all_videos_frames = 0
    self.all_videos_cnt = 0

  def run(self, sess, inputs):
    v_video_id, v_num_frames = sess.run([
        inputs["video_id"], inputs["num_frames"]
    ])
    self.all_videos_frames += sum(v_num_frames)
    self.all_videos_cnt += len(v_video_id)

  def done(self):
    print(self.all_videos_cnt)
    print(self.all_videos_frames)


class Sample():
  def __init__(self):
    self.num_to_sample = 256000
    self.num_frames = 256000000
    self.sample_ratio = 2. * self.num_to_sample / self.num_frames
    print("sample ratio: {0}".format(self.sample_ratio))
    self.sampled_cnt = 0
    self.feas = []

  def run(self, input_raw, frame_cnt):
    order = np.arange(frame_cnt)
    np.random.shuffle(order)
    sample_frame = int(self.sample_ratio * frame_cnt)
    sample_frame = 1 if sample_frame == 0 else sample_frame
    order = np.sort(order[: sample_frame], axis=0)
    self.feas.append(input_raw[order, ...])
    self.sampled_cnt += sample_frame
    if self.sampled_cnt > self.num_to_sample:
      break

  def get_feas(self):
    self.feas = np.vstack(self.feas)
    fout = h5py.File("sampled.h5", "w")
    fout.create_dataset('feas', data=self.feas, dtype=np.float16)


def main(stage, split_id=""):
  feature_names = "rgb"
  feature_sizes = "1024"
  input_shuffle = True

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
      files, shuffle=input_shuffle, num_epochs=1)
  eval_data = [
      reader.prepare_reader(filename_queue) for _ in xrange(num_readers)
  ]

  # eval_data = reader.prepare_reader(filename_queue)
  if input_shuffle:
    batch_join = tf.train.shuffle_batch_join
  else:
    batch_join = tf.train.batch_join
  eval_data = batch_join(
      eval_data,
      batch_size=batch_size,
      capacity=5 * batch_size,
      allow_smaller_final_batch=True)
  video_id_batch, model_input_raw, labels_batch, num_frames = eval_data
  inputs = {
      'video_id': video_id_batch,
      'input_raw': model_input_raw,
      'labels': labels_batch,
      'num_frames': num_frames
  }

  task = Stats()

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
        # batch_start_time = time.time()
        task.run(sess, inputs)
        # print(v_model_input_raw.shape)
        # print(v_video_id)
        # print(v_labels_batch)
        # print(v_num_frames)
        # exit(0)
        # seconds_per_batch = time.time() - batch_start_time
        # example_per_second = v_video_id.shape[0] / seconds_per_batch
        # examples_processed += v_video_id.shape[0]
        examples_processed += batch_size

        cnt += 1
        if cnt % 5 == 0:
          print("examples processed: {}".format(examples_processed))

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
  task.done()


# main("train", sys.argv[1])
# main("train")
main("validate")
# main("train")
