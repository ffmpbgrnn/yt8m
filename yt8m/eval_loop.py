import time
import h5py
import cPickle as pkl
import numpy as np

import tensorflow as tf
from tensorflow import logging

from yt8m.evaluation import eval_util
import utils


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def matrix_to_tfexample(feats, labels=[], video_id=None):
  return tf.train.Example(
      features=tf.train.Features(feature={
          "labels": _int64_feature(labels),
          "video_id": _bytes_feature([video_id]),
          "feats": _float_feature(feats),
      }),
  )

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


def evaluation_loop(self, saver, model_ckpt_path):
  global_step_val = model_ckpt_path.split("/")[-1].split("-")[-1]
  evl_metrics = eval_util.EvaluationMetrics(self.model.num_classes, self.config.top_k)

  # summary_writer = tf.summary.FileWriter(
      # self.train_dir, graph=tf.get_default_graph())
  summary_writer = None

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
  video_ids = []
  output_scores = 0 # 1->output score, 2-> output features
  if output_scores == 1:
    model_id = model_ckpt_path.split("/")[-2]
    # num_insts = 4906660
    # stage = "train"
    # num_insts = 1401828
    # stage = "validate"
    num_insts = 700640
    stage = "test"
    video_ids_pkl_path = "/data/D2DCRC/linchao/YT/scores/{}.{}.pkl".format(model_id, stage)
    pred_out = h5py.File("/data/D2DCRC/linchao/YT/scores/{}.{}.h5".format(model_id, stage), "w")
    pred_dataset = pred_out.create_dataset('scores', shape=(num_insts, self.model.num_classes),
                                            dtype=np.float32)
  elif output_scores == 2:
    output_prefix = "/data/uts700/linchao/yt8m/data/555_netvlad/train"
    tfrecord_cntr = 0
    output_filename = "{}/{}.tfrecord".format(output_prefix, tfrecord_cntr / 1200)
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)

  with tf.Session(config=sess_config) as sess:
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
        example_per_second = res["dense_labels"].shape[0] / seconds_per_batch
        examples_processed += res["dense_labels"].shape[0]
        predictions = res["predictions"]
        video_id = res["video_id"].tolist()
        if output_scores == 1:
          pred_dataset[len(video_ids): len(video_ids) + len(video_id), :] = predictions
          video_ids += video_id
        elif output_scores == 2:
          for i in xrange(len(video_id)):
            sparse_label = np.array(res["dense_labels"][i], dtype=np.int32)
            sparse_label = np.where(sparse_label == 1)[0].tolist()
            feat = res["feats"][i]
            print(sparse_label, video_id[i])
            example = matrix_to_tfexample(
                feat, labels=sparse_label, video_id=video_id[i])
            tfrecord_writer.write(example.SerializeToString())
            if tfrecord_cntr % 1200 == 0:
              tfrecord_writer.close()
              output_filename = "{}/{}.tfrecord".format(output_prefix, tfrecord_cntr / 1200)
              tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
            tfrecord_cntr += 1

        if type(predictions) == list:
          predictions = eval_util.transform_preds(self, predictions)

        iteration_info_dict = evl_metrics.accumulate(
            predictions, res["dense_labels"], res["loss"])
        iteration_info_dict["examples_per_second"] = example_per_second

        gap = eval_util.calculate_gap(predictions, res["dense_labels"])
        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        '''
        p = [str(_) for _ in np.where(res["dense_labels"][0, :] > 0)[0].tolist()]
        print_labels = "+".join(p)
        p = np.argsort(res["predictions"][0, :])[-20:]
        p = np.sort(p).tolist()
        p = [str(_) for _ in p]
        pred_labels = "+".join(p)
        logging.info("vid: %s; gap: %s; labels %s; predictions %s" % (
            res['video_id'][0], gap, print_labels, pred_labels))
        '''
        logging.info("examples_processed: %d | %s | gap: %s", examples_processed,
                     iterinfo, gap)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      if output_scores == 1:
        pred_out.close()
        pkl.dump(video_ids, open(video_ids_pkl_path, "w"))
      elif output_scores == 2:
        tfrecord_writer.close()
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
