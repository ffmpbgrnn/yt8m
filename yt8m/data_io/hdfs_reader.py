import h5py
import random
import numpy as np

import tensorflow as tf

data = h5py.File("/data/state/linchao/YT/hdfs/video/train/mean.h5", 'r')

num_classes = 4716
classes = np.arange(num_classes)
np.random.shuffle(classes)

step_size = num_classes / batch_size
batch_class = classes[step_size * i: step_size * (i + 1)]


def return_identity(placeholders):
  return placeholders

class Feed_fn():
  def __init__(self, label_vid_dict, seed):
    random.seed(seed)
    self.label_vid_dict = dict(label_vid_dict)
    self.label_vid_ptr = {}
    for i in xrange(num_classes):
      self.label_vid_ptr[i] = 0


  def get_next_vid(self, labels):
    for label in labels:
    vids = self.label_vid_dict[label]
    vid_ptr = self.label_vid_ptr[label]
    vid = vids[vid_ptr]
    if len(vids) == vid_ptr + 1:
      self.label_vid_ptr[label] = 0
      random.shuffle(self.label_vid_dict[label])
    else:
      self.label_vid_ptr[label] = vid_ptr + 1
    return vid

  def __call__(self):
    if self._num_epochs and self._epoch >= self._num_epochs:
      raise errors.OutOfRangeError(None, None,
                                   "Already emitted %s epochs." % self._epoch)

    integer_indexes = [
        j % self._max for j in range(self._trav, self._trav + self._batch_size)]

    shuffle_indexes = False
    epoch_inc = False
    if self._epoch_end in integer_indexes:
      # after this batch we will have processed self._epoch epochs, possibly
      # overshooting a bit to fill out a batch.
      epoch_inc = True
      if self._phase_train:
        shuffle_indexes = True
      batch_end_inclusive = integer_indexes.index(self._epoch_end)
      integer_indexes = integer_indexes[:(batch_end_inclusive+1)]
      '''
      if self._epoch + 1 == self._num_epochs:
        # trim this batch, so as not to overshoot the last epoch.
        batch_end_inclusive = integer_indexes.index(self._epoch_end)
        integer_indexes = integer_indexes[:(batch_end_inclusive+1)]
      '''

    self._trav = (integer_indexes[-1] + 1) % self._max
    feed_dict = self.get_feed_dict(integer_indexes)
    if shuffle_indexes:
      np.random.shuffle(self._ranges)
    if epoch_inc:
      self._epoch += 1
    return feed_dict

def enqueue_data(name="enqueue_input", shuffle=True):
  label_to_vids = {}
  vid_to_labels = {}
  with tf.name_scope(name):
    min_after_dequeue = 0  # just for the summary text
    queue = tf.FIFOQueue(capacity,
                          dtypes=queue_types,
                          shapes=queue_shapes)

    enqueue_ops = []
    feed_fns = []

    for i in range(num_threads):
      # Note the placeholders have no shapes, so they will accept any
      # enqueue_size.  enqueue_many below will break them up.
      placeholders = [tf.placeholder(t) for t in placeholder_types]
      out_ops = return_identity(placeholders)

      enqueue_ops.append(queue.enqueue_many(out_ops))
      feed_fns.append(Feed_fn(placeholders,
                                  input_,
                                  enqueue_size,
                                  random_start=shuffle,
                                  num_epochs=num_epochs,
                                  seed=1234 + i,
                                  extras=extras))

    runner = fqr.FeedingQueueRunner(queue=queue,
                                    enqueue_ops=enqueue_ops,
                                    feed_fns=feed_fns)
    queue_runner.add_queue_runner(runner)

    return queue

queue = enqueue_data(
    input_, queue_feed_fn_name, queue_capacity, shuffle=shuffle, num_threads=num_threads, seed=seed,
    enqueue_size=1, num_epochs=num_epochs, min_after_dequeue=0,
    queue_types=queue_types, placeholder_types=placeholder_types, queue_shapes=queue_shapes,
    _get_ops=_get_ops, extras=extras)

if num_epochs is None:
  features = queue.dequeue_many(batch_size)
else:
  features = queue.dequeue_up_to(batch_size)
