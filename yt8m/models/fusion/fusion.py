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

"""Contains model definitions."""
import math

import tensorflow.contrib.slim as slim
import tensorflow as tf

from yt8m.models import models
from yt8m.models import aucpr



class FusionModel(models.BaseModel):
  def __init__(self):
    super(FusionModel, self).__init__()
    self.normalize_input = True
    self.clip_global_norm = 1
    self.var_moving_average_decay = 0
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    self.num_max_labels = -1

    self.num_classes = 4716


  def create_model(self,
                   model_input,
                   vocab_size,
                   dense_labels=None,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    num_models = 5
    w = tf.get_variable("weights", [num_models, 4716], regularizer=slim.l2_regularizer(1e-8),
                        initializer=tf.constant_initializer(1./num_models))
    b = tf.get_variable("biases", [4716], initializer=tf.constant_initializer(0.))
    w = tf.nn.softmax(w, axis=0)
    logits = tf.reshape(model_input, [num_models, 4716]) * w + b
    loss = aucpr.aucpr_loss(logits, dense_labels)
    return {"predictions": logits, 'loss': loss}
