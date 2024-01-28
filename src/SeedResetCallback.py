# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SeedResetCallback

# 2023/11/01

# See https://github.com/NVIDIA/framework-reproducibility/blob/master/doc/seeder/seeder_tf2.md

import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import random_seed

SEED =137

class SeedResetCallback(tf.keras.callbacks.Callback):
  def __init__(self, seed=SEED):
    self.seed=seed
  
  def on_epoch_begin(self, epoch, logs=None):
     tf.random.set_seed(self.seed)
     random.seed    = self.seed
     np.random.seed = self.seed
     #random_seed.set_seed(self.seed)
     #cv2.setRNGSeed(self.seed)

