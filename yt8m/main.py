import tensorflow as tf
from tensorflow import app
from tensorflow import gfile
from tensorflow import logging
from tensorflow import flags

from yt8m.models import losses
from yt8m.starter import frame_level_models
from yt8m.starter import video_level_models
from yt8m.models.lstm import lstm
from yt8m.models.lstm import lstm_enc_dec
from yt8m.data_io import readers
import utils
from .config import base as base_config
import train_loop
import eval_loop
import inference_loop

FLAGS = flags.FLAGS

flags.DEFINE_string("stage", "train", "")
flags.DEFINE_string("model_ckpt_path", "", "")
flags.DEFINE_string("config_name", "BaseConfig", "")

class Expr(object):
  def __init__(self):
    self.stage = FLAGS.stage
    self.model_ckpt_path = FLAGS.model_ckpt_path
    self.config = utils.find_class_by_name(FLAGS.config_name,
                                           [base_config,])(self.stage)
    self.phase_train = self.config.phase_train
    self.task = 0
    self.ps_tasks = 0
    self.is_chief = (self.task == 0)
    self.master = ""

    self.batch_size = self.config.batch_size

    if not self.phase_train:
      tf.set_random_seed(0)

    self.model = utils.find_class_by_name(self.config.model_name,
        [frame_level_models, video_level_models, lstm, lstm_enc_dec])()
    self.label_loss_fn = utils.find_class_by_name(
        self.config.label_loss, [losses])()
    self.optimizer = utils.find_class_by_name(
        self.model.optimizer_name, [tf.train])

    self.reader = self.get_reader()
    self.num_classes = self.reader.num_classes

    self.build_graph()
    logging.info("built graph")

    if self.model.var_moving_average_decay > 0:
      print("Using moving average")
      variable_averages = tf.train.ExponentialMovingAverage(
          self.model.var_moving_average_decay)
      variables_to_restore = variable_averages.variables_to_restore()
      eval_saver = tf.train.Saver(variables_to_restore)
    else:
      eval_saver = tf.train.Saver(tf.global_variables())

    if self.stage == "train":
      train_loop.train_loop(self)
    elif self.stage == "eval":
      eval_loop.evaluation_loop(self, eval_saver, self.model_ckpt_path)
    elif self.stage == "inference":
      inference_loop.inference_loop(self, eval_saver, self.model_ckpt_path)

  def get_input_data_tensors(self,
                             data_pattern,
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
            allow_smaller_final_batch=True,
            enqueue_many=True)
      else:
        return tf.train.batch_join(
            data,
            batch_size=self.batch_size,
            capacity=self.batch_size * 3,
            allow_smaller_final_batch=True,
            enqueue_many=True)

  def get_reader(self):
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        self.config.feature_names, self.config.feature_sizes)

    if self.config.use_frame_features:
      reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names,
          num_max_labels=self.model.num_max_labels,
          feature_sizes=feature_sizes)
    else:
      reader = readers.YT8MAggregatedFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes)
    return reader

  def build_graph(self):
    with tf.device(tf.train.replica_device_setter(
        self.ps_tasks, merge_devices=True)):
      self.global_step = tf.Variable(0, trainable=False, name="global_step")

      video_id_batch, model_input_raw, dense_labels_batch, sparse_labels_batch, num_frames, label_weights_batch = self.get_input_data_tensors(
          self.config.data_pattern,
          num_readers=self.config.num_readers,
          num_epochs=self.config.num_epochs)
      feature_dim = len(model_input_raw.get_shape()) - 1

      if self.model.normalize_input:
        print("L2 Normalizing input")
        model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
      else:
        model_input = model_input_raw

      with tf.name_scope("model"):
        result = self.model.create_model(
            model_input,
            num_frames=num_frames,
            vocab_size=self.reader.num_classes,
            dense_labels=dense_labels_batch,
            sparse_labels=sparse_labels_batch,
            label_weights=label_weights_batch,
            is_training=self.phase_train)

        predictions = result["predictions"]
        if "loss" in result.keys():
          label_loss = result["loss"]
        else:
          label_loss = self.label_loss_fn.calculate_loss(predictions, dense_labels_batch)

      dense_labels_batch = tf.cast(dense_labels_batch, tf.float32)

      if self.stage == "train":
        opt = self.optimizer(self.model.base_learning_rate)
        train_op, label_loss, global_norm = train_loop.get_train_op(self, opt, result, label_loss)
        self.feed_out = {
            "train_op": train_op,
            "loss": label_loss,
            "global_step": self.global_step,
            "predictions": predictions,
            "dense_labels": dense_labels_batch,
            "global_norm": global_norm,
        }
      elif self.stage == "eval":
        self.feed_out = {
          "video_id": video_id_batch,
          "predictions": predictions,
          "dense_labels": dense_labels_batch,
          "loss": label_loss
        }
      elif self.stage == "inference":
        self.feed_out = {
          "video_id": video_id_batch,
          "predictions": predictions,
        }


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  Expr()

if __name__ == "__main__":
  app.run()
