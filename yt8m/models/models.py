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

"""Contains the base class for models."""

class BaseModel(object):
  """Inherit from this class when implementing new models."""
  def __init__(self):
    self.normalize_input = True
    self.clip_global_norm = 0
    self.var_moving_average_decay = 0
    self.optimizer_name = "AdamOptimizer"
    # self.optimizer_name = "MomentumOptimizer"
    self.base_learning_rate = 1e-2
    self.num_max_labels = -1
    self.num_classes = 4716

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()

  def get_train_init_fn(self):
    return None
