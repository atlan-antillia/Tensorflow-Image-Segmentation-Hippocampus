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

# TensorflowUNetGeneratorTrainer.py
# 2023/08/20 to-arai


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset

from TensorflowUNet import TensorflowUNet

from ImageMaskDatasetGenerator import ImageMaskDatasetGenerator

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"

if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]

    # Create a UNetMolde and compile
    model   = TensorflowUNet(config_file)
        
    train_gen = ImageMaskDatasetGenerator(config_file, dataset=TRAIN)
    train_generator = train_gen.generate()

    valid_gen = ImageMaskDatasetGenerator(config_file, dataset=EVAL)
    valid_generator = valid_gen.generate()

    model.train(train_generator, valid_generator)

  except:
    traceback.print_exc()
    
