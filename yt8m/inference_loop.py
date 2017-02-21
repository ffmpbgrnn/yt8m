import time

import tensorflow as tf
import numpy as np
from tensorflow import logging
from tensorflow import gfile

def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in xrange(batch_size):
    top_indices = np.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index] + "," + " ".join("%i %f" % pair
                                                  for pair in line) + "\n"

def restore(saver, sess, train_dir):
  latest_checkpoint = tf.train.latest_checkpoint(train_dir)
  if latest_checkpoint is None:
    raise Exception("unable to find a checkpoint at location: %s" % train_dir)
  else:
    meta_graph_location = latest_checkpoint + ".meta"
    logging.info("loading meta-graph: " + meta_graph_location)

  saver = tf.train.import_meta_graph(meta_graph_location)
  logging.info("restoring variables from " + latest_checkpoint)
  saver.restore(sess, latest_checkpoint)

def inference_loop(self, saver, model_ckpt_path):
  output_path = "{}.inference_predicts".format(model_ckpt_path.split('/')[-1])
  with tf.Session() as sess, gfile.Open(output_path, "w+") as out_file:
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, model_ckpt_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    out_file.write("VideoId,LabelConfidencePairs\n")

    try:
      while not coord.should_stop():
        res = sess.run(self.feed_out)
        num_examples_processed += len(res["predictions"].shape[0])
        logging.info("num examples processed: %d; elapsed seconds: %.2f " % (
            num_examples_processed, time.time() - start_time))
        for line in format_lines(res["video_id"], res["predictions"], self.config.top_k):
          out_file.write(line)
        out_file.flush()
    except tf.errors.OutOfRangeError:
      logging.info('Done with inference. The output file was written to ' + output_path)
    finally:
      coord.request_stop()

    coord.join(threads)
