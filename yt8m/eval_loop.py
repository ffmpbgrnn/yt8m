import time
import numpy as np

import tensorflow as tf
from tensorflow import logging

from yt8m.evaluation import eval_util
import utils

def restore(saver, sess, train_dir):
  last_global_step_val = -1
  latest_checkpoint = tf.train.latest_checkpoint(train_dir)
  if latest_checkpoint:
    logging.info("Loading checkpoint for eval: " + latest_checkpoint)
    # Restores from checkpoint
    saver.restore(sess, latest_checkpoint)
    # Assuming model_checkpoint_path looks something like:
    # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
    global_step_val = latest_checkpoint.split("/")[-1].split("-")[-1]
  else:
    logging.info("No checkpoint file found.")
    return global_step_val

  if global_step_val == last_global_step_val:
    logging.info("skip this checkpoint global_step_val=%s "
                  "(same as the previous one).", global_step_val)
    return global_step_val

def transform_preds(predictions):
  eos_id = self.num_classes + 2
  batch_size = predictions.shape[0]
  vec_size = predictions.shape[2]
  max_seq_length = 10
  valid_matrix = np.zeros((batch_size, max_seq_length, vec_size), dtype=np.int32)
  pred_matrix = []
  for seq_idx, logit in enumerate(predictions):
    max_pred = np.argmax(logit, axis=1)
    find_eos = np.where(max_pred != eos_id)[0]
    valid_matrix[find_eos, seq_idx] = 1
    pred_matrix.append(np.expand_dims(predictions, axis=1))
  pred_matrix = np.concatenate(pred_matrix, axis=1)
  preds = pred_matrix * valid_matrix
  preds = np.mean(preds, axis=1)
  return preds

def evaluation_loop(self, saver, model_ckpt_path):
  global_step_val = model_ckpt_path.split("/")[-1].split("-")[-1]
  evl_metrics = eval_util.EvaluationMetrics(self.num_classes, self.config.top_k)

  # summary_writer = tf.summary.FileWriter(
      # self.train_dir, graph=tf.get_default_graph())
  summary_writer = None

  with tf.Session() as sess:
    saver.restore(sess, model_ckpt_path)
    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()
        res = sess.run(self.feed_out)
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = res["predictions"].shape[0] / seconds_per_batch
        examples_processed += res["predictions"].shape[0]

        iteration_info_dict = evl_metrics.accumulate(
            transform_preds(self, res["predictions"]),
            res["dense_labels"], res["loss"])
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      if summary_writer:
        summary_writer.add_summary(res["summary"], global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
