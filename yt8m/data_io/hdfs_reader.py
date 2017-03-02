import h5py
import threading
import cPickle as pkl
import random
import numpy as np
import Queue

import tensorflow as tf
from tensorflow.python.training import queue_runner
from . import feeding_queue_runner as fqr


class Feed_fn_setup(object):
  def __init__(self):
    with open("/data/state/linchao/YT/hdfs/video/train/mean.pkl") as fin:
      self.vid_list = pkl.load(fin)
    self.mean_data = h5py.File("/data/state/linchao/YT/hdfs/video/train/mean.h5", 'r')

    with open("/data/uts700/linchao/yt8m/YT/data/vid_info/train_vid_to_labels.pkl") as fin:
      self.vid_to_labels = pkl.load(fin)

    self.label_to_vid_dict = {}
    for vid, labels in self.vid_to_labels.iteritems():
      for l in labels:
        c = self.label_to_vid_dict.get(l, [])
        c.append(vid)
        self.label_to_vid_dict[l] = c

    self.num_classes = 4716
    self.batch_size = 128

    self.batch_id_queue = Queue.Queue(500)
    bi_threads = [threading.Thread(target=self.input_vid_threads)]
    bi_threads.start()


  def input_vid_threads(self):
    labels = np.arange(self.num_classes)

    label_vid_ptr = {}
    for i in xrange(self.num_classes):
      label_vid_ptr[i] = 0

    # step_size = num_classes / batch_size
    # batch_labels = classes[step_size * i: step_size * (i + 1)]
    batch_vids = []
    while True:
      np.random.shuffle(labels)
      for label in labels:
        vids = self.label_to_vid_dict[label]
        vid_ptr = label_vid_ptr[label]
        batch_vids.append(vids[vid_ptr])
        if len(batch_vids) == self.batch_size:
          self.batch_id_queue.put(batch_vids)
          batch_vids = []

        if len(vids) == vid_ptr + 1:
          label_vid_ptr[label] = 0
          random.shuffle(self.label_to_vid_dict[label])
        else:
          label_vid_ptr[label] = vid_ptr + 1


class Feed_fn(object):
  def __init__(self, info):
    self._i = info

  def __call__(self):
    vids = self._i.batch_id_queue.get()
    vid_index = []
    dense_labels = []
    for vid in vids:
      idx = self._i.vid_list.index(vid)
      vid_index.append(idx)
      dense_label = np.zeros((1, self._i.num_classes), dtype=np.int64)
      labels = self._i.vid_to_labels[vid]
      for l in labels:
        dense_label[0, l] = 1
      dense_labels.append(dense_label)
    dense_labels = np.vstack(dense_labels)

    vid_index = np.array(vid_index)
    vid_index_sortidx = np.argsort(vid_index)
    batch_data = self._i.mean_data[vid_index[vid_index_sortidx], :]
    batch_data = batch_data[np.argsort(vid_index_sortidx), :]
    feed_dict = {
      "video_id": np.array(vids),
      "labels": dense_labels,
      "feas": batch_data,
    }

    return feed_dict

def enqueue_data(name="enqueue_input", shuffle=True):
  fn_setup = Feed_fn_setup()
  queue_types = [tf.string, tf.int64, tf.float32]
  queue_shapes = [(), (4716,), (1024+128,)]
  capacity = 500
  num_threads = 4
  batch_size = 128
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
      feed_fns.append(Feed_fn(fn_setup))

    runner = fqr.FeedingQueueRunner(queue=queue,
                                    enqueue_ops=enqueue_ops,
                                    feed_fns=feed_fns)
    queue_runner.add_queue_runner(runner)

    features = queue.dequeue_many(batch_size)
  return features
