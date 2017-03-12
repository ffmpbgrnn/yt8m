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
    vid_dict = {}
    with open("/data/state/linchao/YT/video_hdfs/{}/mean.pkl".format(stage)) as fin:
      vid_list = pkl.load(fin)
      for i, vid in enumerate(vid_list):
        vid_dict[vid] = i
    if mem_map:
      mean_data = h5py.File("/data/state/linchao/YT/video_hdfs/{}/mean.h5".format(stage), 'r', driver='core')['feas']
    else:
      mean_data = h5py.File("/data/state/linchao/YT/video_hdfs/{}/mean.h5".format(stage), 'r')['feas']
    return vid_dict, mean_data

  def load_vlad_info(self, stage="train"):
    vid_dict = {}
    with open("/data/state/linchao/YT/vlad_hdfs/{}/mean.pkl".format(stage)) as fin:
      vid_list = pkl.load(fin)
      for i, vid in enumerate(vid_list):
        vid_dict[vid] = i
    mean_data = h5py.File("/data/state/linchao/YT/vlad_hdfs/{}/mean.h5".format(stage), 'r')['feas']
    return vid_dict, mean_data

  def __init__(self, feat_type, num_classes, phase_train, num_threads):
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
    if feat_type == "video":
      self.vid_dict, self.mean_data = self.load_video_info(stage)
    elif feat_type == "vlad":
      self.vid_dict, self.mean_data = self.load_vlad_info(stage)

    # TODO
    target_label = 0
    # target_label = 4712
    self.pos_vids, self.neg_vids = [], []
    for vid, labels in self.vid_to_labels.iteritems():
      labels = [int(l) for l in labels]
      if target_label in labels:
        self.pos_vids.append(vid)
      else:
        self.neg_vids.append(vid)
    random.shuffle(self.pos_vids)
    random.shuffle(self.neg_vids)
    print("num pos: {}; num neg: {}".format(len(self.pos_vids), len(self.neg_vids)))

    self.num_classes = num_classes
    # TODO
    self.batch_size = 32

    self.batch_id_queue = Queue.Queue(1500)
    if self.phase_train:
      # TODO
      # bi_threads = threading.Thread(target=self.input_vid_threads_train_keep_ratio)
      bi_threads = threading.Thread(target=self.input_vid_threads_train)
    else:
      bi_threads = threading.Thread(target=self.input_vid_threads_val)
    bi_threads.start()

  def input_vid_threads_val(self):
    for i in xrange(0, len(self.pos_vids), self.batch_size):
      batch_vids = self.pos_vids[i: i + self.batch_size]
      if len(batch_vids) == 0:
        continue
      dense_labels = np.ones((len(batch_vids), 1), dtype=np.int64)
      self.batch_id_queue.put((batch_vids, np.array(dense_labels)))
    for i in xrange(0, len(self.neg_vids), self.batch_size):
      batch_vids = self.neg_vids[i: i + self.batch_size]
      if len(batch_vids) == 0:
        continue
      dense_labels = np.zeros((len(batch_vids), 1), dtype=np.int64)
      self.batch_id_queue.put((batch_vids, np.array(dense_labels)))
    for i in xrange(self.num_threads):
      self.batch_id_queue.put(False)

  def input_vid_threads_train(self):
    vinfo = zip(self.pos_vids + self.neg_vids, [1 for _ in xrange(len(self.pos_vids))] + [0 for _ in xrange(len(self.neg_vids))])
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

  def input_vid_threads_train_keep_ratio(self):
    pos_vid_ptr, neg_vid_ptr = 0, 0
    batch_vids = []
    dense_labels = np.zeros((self.batch_size, 1), dtype=np.int64)
    num_pos = len(self.pos_vids)
    num_neg = len(self.neg_vids)
    ratio = 1. * num_pos / num_neg
    num_pos_batch = int(math.ceil((self.batch_size * ratio)))
    sentinal = []
    # TODO
    num_pos_batch = 90
    for i in xrange(self.batch_size):
      if i < num_pos_batch:
        sentinal.append(1)
      else:
        sentinal.append(0)

    random.shuffle(sentinal)
    shuffle_pos, shuffle_neg = False, False
    while True:
      if sentinal[len(batch_vids)] == 1:
        dense_labels[len(batch_vids), 0] = 1
        batch_vids.append(self.pos_vids[pos_vid_ptr])
        if len(self.pos_vids) == pos_vid_ptr + 1:
          pos_vid_ptr = 0
          shuffle_pos = True
        else:
          pos_vid_ptr += 1
      else:
        dense_labels[len(batch_vids), 0] = 0
        batch_vids.append(self.neg_vids[neg_vid_ptr])
        if len(self.neg_vids) == neg_vid_ptr + 1:
          neg_vid_ptr = 0
          shuffle_neg = True
        else:
          neg_vid_ptr += 1

      if len(batch_vids) == self.batch_size:
        self.batch_id_queue.put((batch_vids, np.array(dense_labels)))
        batch_vids = []
        dense_labels = np.zeros((self.batch_size, 1), dtype=np.int64)
        random.shuffle(sentinal)
        if shuffle_pos:
          random.shuffle(self.pos_vids)
          shuffle_pos = False
        if shuffle_neg:
          random.shuffle(self.neg_vids)
          shuffle_neg = False


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
      # TODO
      batch_data = self._i.mean_data[vid_index[vid_index_sortidx], :]
      # batch_data = self._i.mean_data[vid_index[vid_index_sortidx], :1024]
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

def enqueue_data(feat_type, phase_train, batch_size, num_classes, feature_size, name="enqueue_input",):
  num_threads = 8
  fn_setup = Feed_fn_setup(feat_type, num_classes, phase_train, num_threads)
  queue_types = [tf.string, tf.int64, tf.float32]
  queue_shapes = [(), (num_classes,), (feature_size,)]
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
