# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

from yt8m.evaluation import eval_util
from yt8m.models import losses
from yt8m.starter import frame_level_models
from yt8m.starter import video_level_models
from yt8m.models.lstm import lstm
from yt8m.data_io import readers
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer(
      "batch_size", 1024,
      "How many examples to process per batch for training.")
  flags.DEFINE_string(
      "label_loss", "CrossEntropyLoss",
      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1e-3,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("master", "", "TensorFlow master to use.")
  flags.DEFINE_integer("task", 0, "Task id of the replica running the training."
                       " 0 implies chief Supervisor.""")
  flags.DEFINE_integer("ps_tasks", 0, """Number of tasks in the ps job.
                       If 0 no ps job is used.""")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
