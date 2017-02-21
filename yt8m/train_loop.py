import time

import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from yt8m.evaluation import eval_util
import utils

slim = tf.contrib.slim

def train_loop(self, start_supervisor_services=True):
  saver = tf.train.Saver(max_to_keep=1000000)

  sv = tf.train.Supervisor(logdir=self.config.train_dir,
                           is_chief=self.is_chief,
                           global_step=self.global_step,
                           save_model_secs=600,
                           save_summaries_secs=60,
                           saver=saver)
  sess = sv.prepare_or_wait_for_session(
      self.master,
      start_standard_services=start_supervisor_services,
      config=tf.ConfigProto(log_device_placement=False))

  logging.info("prepared session")
  sv.start_queue_runners(sess)
  logging.info("started queue runners")

  try:
    logging.info("entering training loop")
    while not sv.should_stop():
      batch_start_time = time.time()
      res = sess.run(self.feed_out)

      global_step = res["global_step"]
      predictions = res["predictions"]
      labels = res["labels"]

      seconds_per_batch = time.time() - batch_start_time
      examples_per_second = labels.shape[0] / seconds_per_batch

      hit_at_one = eval_util.calculate_hit_at_one(predictions, labels)
      perr = eval_util.calculate_precision_at_equal_recall_rate(predictions,
                                                                labels)
      gap = eval_util.calculate_gap(predictions, labels)

      log_info_str, log_info = "", {
          "Training step": global_step,
          "Hit@1": hit_at_one,
          "PERR": perr,
          "GAP": gap,
          "Loss": res["loss"],
          "Global norm": res["global_norm"],
      }
      for k, v in log_info.iteritems():
        log_info_str += "%s: %.2f;\t" % (k, v)
      logging.info(log_info_str)
      if self.is_chief and global_step % 10 == 0 and self.config.train_dir:
        sv.summary_writer.add_summary(
            utils.MakeSummary("model/Training_Hit@1",
                              hit_at_one), global_step)
        sv.summary_writer.add_summary(
            utils.MakeSummary("model/Training_Perr", perr),
            global_step)
        sv.summary_writer.add_summary(
            utils.MakeSummary("model/Training_GAP", gap),
            global_step)
        sv.summary_writer.add_summary(
            utils.MakeSummary("global_step/Examples/Second",
                              examples_per_second),
            global_step)
        sv.summary_writer.flush()
  except tf.errors.OutOfRangeError:
    logging.info("Done training -- epoch limit reached")
  logging.info("exited training loop")
  sv.Stop()
  return hit_at_one, perr

def get_train_op(self, opt, result, label_loss):
  for variable in slim.get_model_variables():
    tf.summary.histogram(variable.op.name, variable)
  tf.summary.scalar("label_loss", label_loss)

  if "regularization_loss" in result.keys():
    reg_loss = result["regularization_loss"]
  else:
    reg_loss = tf.constant(0.0)
  if self.config.regularization_penalty != 0:
    tf.summary.scalar("reg_loss", reg_loss)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if "update_ops" in result.keys():
    update_ops += result["update_ops"]
  if update_ops:
    with tf.control_dependencies(update_ops):
      barrier = tf.no_op(name="gradient_barrier")
      with tf.control_dependencies([barrier]):
        label_loss = tf.identity(label_loss)

  # Incorporate the L2 weight penalties etc.
  final_loss = self.config.regularization_penalty * reg_loss + label_loss
  # train_op = optimizer.minimize(final_loss, global_step=global_step)
  params = tf.trainable_variables()
  gradients = tf.gradients(final_loss, params)
  global_norm = tf.global_norm(gradients)
  if self.model.clip_global_norm > 0:
    gradients, _ = tf.clip_by_global_norm(gradients, self.model.clip_global_norm)
  gradients = zip(gradients, params)
  train_op = opt.apply_gradients(gradients, self.global_step)

  if self.model.var_moving_average_decay > 0:
    variable_averages = tf.train.ExponentialMovingAverage(
      self.model.var_moving_average_decay, self.global_step)
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)
    train_op = tf.group(train_op, variables_averages_op)

  return train_op, label_loss, global_norm

def recover_session(self):
  # Recover session
  saver = None
  latest_checkpoint = tf.train.latest_checkpoint(self.train_dir)
  if self.config.start_new_model:
    logging.info("'start_new_model' flag is set. Removing existing train dir.")
    try:
      gfile.DeleteRecursively(self.train_dir)
    except:
      logging.error(
          "Failed to delete directory " + self.train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.")
  elif not latest_checkpoint:
    logging.info("No checkpoint file found. Building a new model.")
  else:
    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("No meta graph file found. Building a new model.")
    else:
      logging.info("Restoring from meta graph file %s", meta_filename)
      saver = tf.train.import_meta_graph(meta_filename)
  return saver
