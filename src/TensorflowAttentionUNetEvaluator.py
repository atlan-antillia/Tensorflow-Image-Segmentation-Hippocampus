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

# TensorflowAttentionUNetEvaluator.py
# 2023/05/30 to-arai

import os
import sys

import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from TensorflowAttentionUNet import TensorflowAttentionUNet

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"

if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    # Create a UNetMolde and compile
    model          = TensorflowAttentionUNet(config_file)

    dataset        = ImageMaskDataset(config_file)
    x_test, y_test = dataset.create(dataset=EVAL)
  
    model.evaluate(x_test, y_test)

  except:
    traceback.print_exc()
    
