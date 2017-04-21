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


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-5,
                   label_smoothing=False, is_training=True, dense_labels=None, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    '''
    output = slim.fully_connected(
        model_input, 4096, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        output, 4096, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    '''

    logits = slim.fully_connected(
        model_input, vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    labels = tf.cast(dense_labels, tf.float32)
    if label_smoothing:
      labels = labels / tf.reduce_sum(labels, axis=1, keep_dims=True)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      loss = tf.reduce_mean(loss)
    else:
      print("Using sigmoid")
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
    if label_smoothing:
      preds = tf.nn.softmax(logits)
    else:
      preds = tf.nn.sigmoid(logits)
    return {"predictions": preds, "loss": loss}

class MoeConfig(object):
  moe_num_mixtures = 100

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  def __init__(self):
    self.normalize_input = True
    self.clip_global_norm = 1
    self.var_moving_average_decay = 0
    self.optimizer_name = "AdamOptimizer"
    self.base_learning_rate = 1e-2
    self.num_max_labels = -1
    # TODO save_model_secs
    # self.num_classes = 25
    # self.num_classes = 2500 - 1250
    # self.num_classes = 3600 - 2500
    self.num_classes = 4716 - 3600

  def create_model_V1(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    num_mixtures = num_mixtures or MoeConfig.moe_num_mixtures

    primary_num_mixtures = 20
    secondary_num_mixtures = 20
    gate_primary_activations = slim.fully_connected(
        model_input,
        vocab_size * (primary_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates_primary")
    gate_secondary_activations = slim.fully_connected(
        model_input,
        vocab_size * (secondary_num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates_secondary")

    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * primary_num_mixtures * secondary_num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_primary_distribution = tf.nn.softmax(tf.reshape(
        gate_primary_activations,
        [-1, primary_num_mixtures + 1]))
    gating_secondary_distribution = tf.nn.softmax(tf.reshape(
        gate_secondary_activations,
        [-1, secondary_num_mixtures + 1]))
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, primary_num_mixtures * secondary_num_mixtures]))

    expert_distribution = tf.reshape(expert_distribution, [-1, primary_num_mixtures, secondary_num_mixtures])
    gating_primary_distribution = tf.reshape(gating_primary_distribution[:, :primary_num_mixtures], [-1, primary_num_mixtures, 1])
    gating_secondary_distribution = tf.reshape(gating_secondary_distribution[:, :secondary_num_mixtures], [-1, 1, secondary_num_mixtures])

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_primary_distribution * gating_secondary_distribution * expert_distribution, [1, 2])
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

  def moe_layer(self, model_input, hidden_size, num_mixtures,
                act_func=None, l2_penalty=None):
    gate_activations = slim.fully_connected(
        model_input,
        hidden_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        hidden_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    expert_act_func = act_func
    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = expert_act_func(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    outputs = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    outputs = tf.reshape(outputs, [-1, hidden_size])
    return outputs

  def create_model_new(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    num_mixtures = num_mixtures or 100

    self.is_training = is_training
    self.l2_penalty = l2_penalty

    outputs = self.moe_layer(model_input, 1024, num_mixtures=100, act_func=tf.nn.relu,
                             l2_penalty=self.l2_penalty)

    final_probabilities = slim.fully_connected(outputs, vocab_size,
                                               activation_fn=tf.nn.sigmoid,
                                               weights_regularizer=slim.l2_regularizer(self.l2_penalty),
                                               scope="final_output")
    return {"predictions": final_probabilities}

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   is_training=True,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or MoeConfig.moe_num_mixtures

    self.is_training = is_training
    # if self.is_training:
      # model_input = tf.nn.dropout(model_input, 0.8)
    '''
    model_input = slim.fully_connected(
        model_input,
        1280,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="input_proj0")
    model_input = slim.fully_connected(
        model_input,
        1280,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="input_proj1")
    '''

    hidden_size = vocab_size
    gate_activations = slim.fully_connected(
        model_input,
        hidden_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        hidden_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class MoeModel_V2(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = 4

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        (vocab_size + 1) * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, vocab_size, num_mixtures + 1]), dim=2)
    expert_distribution = tf.nn.softmax(tf.reshape(
        expert_activations,
        [-1, vocab_size + 1, num_mixtures]), dim=1)

    final_probabilities = tf.reduce_sum(
        gating_distribution[:, :, :num_mixtures] * expert_distribution[:, :vocab_size, :],
        axis=2)
    return {"predictions": final_probabilities}
