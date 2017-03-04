import time
import os

import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from yt8m.evaluation import eval_util
import utils

slim = tf.contrib.slim

def supervised_tasks(self, sv, res):
  global_step = res["global_step"]
  predictions = res["predictions"]
  batch_start_time = res["batch_start_time"]
  if type(predictions) == list:
    predictions = eval_util.transform_preds(self, predictions)
  dense_labels = res["dense_labels"]

  seconds_per_batch = time.time() - batch_start_time
  examples_per_second = dense_labels.shape[0] / seconds_per_batch

  hit_at_one = eval_util.calculate_hit_at_one(predictions, dense_labels)
  perr = eval_util.calculate_precision_at_equal_recall_rate(predictions,
                                                            dense_labels)
  gap = eval_util.calculate_gap(predictions, dense_labels)

  log_info_str, log_info = "", {
      "Training step": global_step,
      "Hit@1": hit_at_one,
      "PERR": perr,
      "GAP": gap,
      "Loss": res["loss"],
      "Global norm": res["global_norm"],
      "Exps/sec": examples_per_second,
  }
  for k, v in log_info.iteritems():
    log_info_str += "%s: %.2f;  " % (k, v)

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
  return log_info_str

def train_loop(self, model_ckpt_path, init_fn=None, start_supervisor_services=True):
  saver = tf.train.Saver(max_to_keep=1000000)

  if len(model_ckpt_path) > 0:
    variables_to_restore = tf.all_variables()
    init_fn = slim.assign_from_checkpoint_fn(
        model_ckpt_path,
        variables_to_restore,
        ignore_missing_vars=False,)

  sv = tf.train.Supervisor(logdir=self.config.train_dir,
                           is_chief=self.is_chief,
                           global_step=self.global_step,
                           save_model_secs=600,
                           save_summaries_secs=600,
                           saver=saver,
                           init_fn=init_fn)
  sess = sv.prepare_or_wait_for_session(
      self.master,
      start_standard_services=start_supervisor_services,
      config=tf.ConfigProto(log_device_placement=False))

  logging.info("prepared session")
  sv.start_queue_runners(sess)
  logging.info("started queue runners")

  log_fout = open(os.path.join(self.config.train_dir, "train.log"), "w")
  try:
    logging.info("entering training loop")
    while not sv.should_stop():
      batch_start_time = time.time()
      res = sess.run(self.feed_out)
      res["batch_start_time"] = batch_start_time
      if res["predictions"] is None:
        log_info_str = "Step loss:{}".format()
      else:
        log_info_str = supervised_tasks(self, sv, res)
      logging.info(log_info_str)
      log_fout.write(log_info_str+'\n')
      if res["global_step"] % 100 == 0:
        log_fout.flush()

  except tf.errors.OutOfRangeError:
    logging.info("Done training -- epoch limit reached")
  logging.info("exited training loop")
  sv.Stop()

def get_train_op(self, result, label_loss):
  if self.model.optimizer_name == "MomentumOptimizer":
    opt = self.optimizer(self.model.base_learning_rate, 0.9)
  elif self.model.optimizer_name == "RMSPropOptimizer":
    opt = tf.train.RMSPropOptimizer(
        self.model.base_learning_rate,
        decay=0.9,
        momentum=0.9,
        epsilon=1)
  else:
    opt = self.optimizer(self.model.base_learning_rate)
  for variable in slim.get_model_variables():
    tf.summary.histogram(variable.op.name, variable)
  tf.summary.scalar("label_loss", label_loss)

  if "regularization_loss" in result.keys():
    reg_loss = result["regularization_loss"]
  else:
    reg_loss = tf.constant(0.0)
  reg_losses = tf.losses.get_regularization_losses()
  if reg_losses:
    reg_loss += tf.add_n(reg_losses)
  if self.config.regularization_penalty != 0:
    tf.summary.scalar("reg_loss", reg_loss)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if "update_ops" in result.keys():
    update_ops += result["update_ops"]

  final_loss = self.config.regularization_penalty * reg_loss + label_loss
  if update_ops:
    with tf.control_dependencies(update_ops):
      barrier = tf.no_op(name="gradient_barrier")
      with tf.control_dependencies([barrier]):
        final_loss = tf.identity(final_loss)

  # Incorporate the L2 weight penalties etc.
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
