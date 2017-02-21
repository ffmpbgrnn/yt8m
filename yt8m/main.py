import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import gfile
from tensorflow import logging

from yt8m.models import losses
from yt8m.starter import frame_level_models
from yt8m.starter import video_level_models
from yt8m.models.lstm import lstm
from yt8m.data_io import readers
import utils

class Expr(object):
  def __init__(self):
    self.phase_train = ""
    self.task = 0
    self.ps_tasks = 0
    self.is_chief = (self.task == 0)
    self.master = ""

    self.train_dir = ""
    self.config = config()

    saver = None
    if self.phase_train:
      saver = recover_session()
    else:
      tf.set_random_seed(0)

    if not saver:
      self.reader = get_reader()
      self.num_classes = self.reader.num_classes

      self.model = utils.find_class_by_name(self.config.model_name,
          [frame_level_models, video_level_models, lstm])()
      self.label_loss_fn = utils.find_class_by_name(
          self.config.label_loss, [losses])()
      self.optimizer = utils.find_class_by_name(
          self.config.optimizer, [tf.train])

      build_graph(
                  train_data_pattern=FLAGS.train_data_pattern,
                  regularization_penalty=FLAGS.regularization_penalty,
                  )
      logging.info("built graph")
      if phase_train:
        saver = tf.train.Saver(max_to_keep=1000000)
      else:
        saver = tf.train.Saver(tf.global_variables())

    if phase_train:
      self.train_loop(saver)
    else:
      self.evaluation_loop(saver)

  def get_input_data_tensors(data_pattern,
                             num_epochs=None,
                             num_readers=1):
    logging.info("Using batch size of " + str(self.batch_size) + ".")
    with tf.name_scope("model_input"):
      files = gfile.Glob(data_pattern)
      if not files:
        raise IOError("Unable to find files. data_pattern='" +
                      data_pattern + "'")
      logging.info("number of training files: " + str(len(files)))
      filename_queue = tf.train.string_input_producer(
          files, shuffle=self.phase_train, num_epochs=num_epochs)
      data = [
          self.reader.prepare_reader(filename_queue) for _ in xrange(num_readers)]

      if self.phase_train:
        return tf.train.shuffle_batch_join(
            data,
            batch_size=self.batch_size,
            capacity=self.batch_size * 10,
            min_after_dequeue=self.batch_size,
            allow_smaller_final_batch=True)
      else:
        return tf.train.batch_join(
            data,
            batch_size=self.batch_size,
            capacity=3 * self.batch_size,
            allow_smaller_final_batch=True)

  def get_reader(self):
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        self.config.feature_names, self.config.feature_sizes)

    if self.config.use_frame_features:
      reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes)
    else:
      reader = readers.YT8MAggregatedFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes)
    return reader

  def recover_session():
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

  def build_graph(self,
                train_data_pattern,
                regularization_penalty=1e-3,
                ):
    with tf.device(tf.train.replica_device_setter(
        self.ps_tasks, merge_devices=True)):
      self.global_step = tf.Variable(0, trainable=False, name="global_step")

      opt = self.optimizer(self.config.base_learning_rate)

      video_id_batch, model_input_raw, labels_batch, num_frames = get_input_data_tensors(
          train_data_pattern,
          num_readers=self.config.num_readers,
          num_epochs=self.config.num_epochs)
      feature_dim = len(model_input_raw.get_shape()) - 1

      if self.model.normazlie_input:
        print("L2 Normalizing input")
        model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

      with tf.name_scope("model"):
        result = self.model.create_model(
            model_input,
            num_frames=num_frames,
            vocab_size=self.reader.num_classes,
            labels=labels_batch,
            is_training=self.phase_train)

        predictions = result["predictions"]
        if "loss" in result.keys():
          label_loss = result["loss"]
        else:
          label_loss = self.label_loss_fn.calculate_loss(predictions, labels_batch)

      if phase_train:
        self.get_train_op()

      tf.add_to_collection("global_step", global_step)
      tf.add_to_collection("loss", label_loss)
      tf.add_to_collection("predictions", predictions)
      tf.add_to_collection("input_batch_raw", model_input_raw)
      tf.add_to_collection("input_batch", model_input)
      tf.add_to_collection("num_frames", num_frames)
      tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
      tf.add_to_collection("train_op", train_op)
      tf.add_to_collection("global_norm", global_norm)

      tf.add_to_collection("video_id_batch", video_id_batch)

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  Expr()

if __name__ == "__main__":
  app.run()
