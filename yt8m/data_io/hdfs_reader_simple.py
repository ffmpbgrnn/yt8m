import h5py
import time
import math
import threading
import cPickle as pkl
import random
import numpy as np
import Queue

import tensorflow as tf
from tensorflow.python.training import queue_runner
from tensorflow.python.framework import errors
from . import feeding_queue_runner as fqr


class Feed_fn_setup(object):
  def load_video_info(self, stage="train", mem_map=True):
    self.K = 2
    vid_dict = {}
    with open("/data/state/linchao/YT/video_hdfs/{}/mean.pkl".format(stage)) as fin:
      vid_list = pkl.load(fin)
      for i, vid in enumerate(vid_list):
        vid_dict[vid] = i
      mean_data = h5py.File("/data/state/linchao/YT/video_hdfs/{}/mean.h5".format(stage), 'r')['feas'] # TODO
    return vid_dict, mean_data

  def __init__(self, num_classes, phase_train, num_threads):
    self.num_threads = num_threads
    self.phase_train = phase_train
    if self.phase_train:
      stage = "train"
      print("loading vid info")
      with open("/data/uts700/linchao/yt8m/YT/data/vid_info/train_vid_to_labels_-1.pkl") as fin:
        self.vid_to_labels = pkl.load(fin)
    else:
      stage = "validate"
      with open("/data/uts700/linchao/yt8m/YT/data/vid_info/validate_vid_to_labels.pkl") as fin:
        self.vid_to_labels = pkl.load(fin)
    self.vid_dict, self.mean_data = self.load_video_info(stage)

    self.vid_dense_labels = []
    for vid, labels in self.vid_to_labels.iteritems():
      dense_l = np.zeros((num_classes), dtype=np.int64)
      for l in labels:
        l = int(l)
        dense_l[l] = 1
      self.vid_dense_labels.append((vid, dense_l))

    self.num_classes = num_classes
    # TODO
    self.batch_size = 32

    self.batch_id_queue = Queue.Queue(1500)
    if self.phase_train:
      bi_threads = threading.Thread(target=self.input_vid_threads_train)
    else:
      bi_threads = threading.Thread(target=self.input_vid_threads_val)
    bi_threads.start()

  def input_vid_threads_val(self):
    for i in xrange(0, len(self.vid_dense_labels), self.batch_size):
      batch_vids, dense_labels = self.vid_dense_labels[i: i + self.batch_size]
      if len(batch_vids) == 0:
        continue
      self.batch_id_queue.put((batch_vids, np.array(dense_labels)))

    for i in xrange(self.num_threads):
      self.batch_id_queue.put(False)

  def input_vid_threads_train(self):
    vinfo = self.vid_dense_labels
    random.shuffle(vinfo)
    ptr = 0
    num_vs = len(vinfo)
    while True:
      batch_vids, dense_labels = zip(*vinfo[ptr: ptr + self.batch_size])
      self.batch_id_queue.put((batch_vids, np.reshape(np.array(dense_labels), [-1, 1])))
      ptr += self.batch_size
      if ptr >= num_vs:
        ptr = 0
        random.shuffle(vinfo)

class Feed_fn(object):
  def __init__(self, info, placeholders):
    self._i = info
    self.vid_dict = info.vid_dict
    self.vid_to_labels = info.vid_to_labels
    self.placeholders = placeholders

  def __call__(self):
    ins = self._i.batch_id_queue.get()
    if ins is False:
      raise errors.OutOfRangeError(None, None,
                                   "Already emitted epochs.")
    vids, dense_labels = ins
    vid_index = []
    for vid_idx, vid in enumerate(vids):
      idx = self.vid_dict[vid]
      vid_index.append(idx)

    vid_index = np.array(vid_index)
    vid_index_sortidx = np.argsort(vid_index)
    try:
      batch_data = self._i.mean_data[vid_index[vid_index_sortidx], :]
    except:
      print(vid_index[vid_index_sortidx])
      print(vids)
      print("\n")

    batch_data = batch_data[np.argsort(vid_index_sortidx), :]

    vals = [np.array(vids), dense_labels, batch_data]
    feed_dict = {}
    for pl, val in zip(self.placeholders, vals):
      feed_dict[pl.name] = val
    return feed_dict

'''
video_id_batch, dense_labels_batch, model_input_raw = hdfs_reader_bias_simple.enqueue_data(
          self.phase_train, self.batch_size,
          self.model.num_classes,)
'''

def enqueue_data(phase_train, batch_size, num_classes, name="enqueue_input",):
  num_threads = 8
  fn_setup = Feed_fn_setup(num_classes, phase_train, num_threads)
  queue_types = [tf.string, tf.int64, tf.float32]
  feature_size = 4716
  queue_shapes = [(), (num_classes,), (feature_size, fn_setup.K)]
  capacity = 1500
  with tf.name_scope(name):
    queue = tf.FIFOQueue(capacity,
                         dtypes=queue_types,
                         shapes=queue_shapes)

    enqueue_ops = []
    feed_fns = []

    def return_identity(placeholders):
      return placeholders

    for i in range(num_threads):
      # Note the placeholders have no shapes, so they will accept any
      # enqueue_size.  enqueue_many below will break them up.
      placeholders = [tf.placeholder(t) for t in queue_types]
      out_ops = return_identity(placeholders)

      enqueue_ops.append(queue.enqueue_many(out_ops))
      feed_fns.append(Feed_fn(fn_setup, placeholders))

    runner = fqr.FeedingQueueRunner(queue=queue,
                                    enqueue_ops=enqueue_ops,
                                    feed_fns=feed_fns)
    queue_runner.add_queue_runner(runner)

    if phase_train:
      features = queue.dequeue_many(batch_size)
    else:
      features = queue.dequeue_up_to(batch_size)
  return features
