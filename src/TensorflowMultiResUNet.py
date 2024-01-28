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

# 2023/06/24 

# This is based on the code in the following web sites:

# 1. Semantic-Segmentation-Architecture
# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/multiresunet.py

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train_eval_infer.config


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import glob
import traceback
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import (Conv2D, Dropout, Conv2D, MaxPool2D, 
                                     Activation, BatchNormalization, UpSampling2D, Concatenate)

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import elu, relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ConfigParser import ConfigParser

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter
from losses import dice_coef, basnet_hybrid_loss, jacard_loss, sensitivity, specificity
from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

BEST_MODEL_FILE = "best_model.h5"

# Define TensorflowMultiResUNet class as a subclass of TensorflowUNet

class TensorflowMultiResUNet(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)

  # The following methods have been taken from the following code.
  # https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/multiresunet.py
  def conv_block(self, x, num_filters, kernel_size, padding="same", act=True):
    x = Conv2D(num_filters, kernel_size, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x

  def multires_block(self, x, num_filters, alpha=1.67):
    W = num_filters * alpha

    x0 = x
    x1 = self.conv_block(x0, int(W*0.167), 3)
    x2 = self.conv_block(x1, int(W*0.333), 3)
    x3 = self.conv_block(x2, int(W*0.5),   3)
    xc = Concatenate()([x1, x2, x3])
    xc = BatchNormalization()(xc)

    nf = int(W*0.167) + int(W*0.333) + int(W*0.5)
    sc = self.conv_block(x0, nf, 1, act=False)

    x = Activation("relu")(xc + sc)
    x = BatchNormalization()(x)
    return x

  def res_path(self, x, num_filters, length):
    for i in range(length):
        x0 = x
        x1 = self.conv_block(x0, num_filters, 3, act=False)
        sc = self.conv_block(x0, num_filters, 1, act=False)
        x = Activation("relu")(x1 + sc)
        x = BatchNormalization()(x)
    return x

  def encoder_block(self, x, num_filters, length):
    x = self.multires_block(x, num_filters)
    s = self.res_path(x, num_filters, length)
    p = MaxPool2D((2, 2))(x)
    return s, p

  def decoder_block(self, x, skip, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = self.multires_block(x, num_filters)
    return x

  # Customizable by the parameters in a configuration file.
  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("=== TensorflowAttentionUNet.create ")
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_height, image_width, image_channels))
    p = Lambda(lambda x: x / 255)(inputs)

    enc = []
    for i in range(num_layers):
      filters = base_filters * (2**i)
      if i < num_layers-1:
        s, p = self.encoder_block(p, filters, num_layers-i)
        print("--- Encoder filters {}".format(filters))
        enc.append(s)
      else:
        print("=== Bridge filters {}".format(filters))
        d = self.multires_block(p, filters)

    enc_len = len(enc)
    enc.reverse()
    n = 0
    for i in range(num_layers-1):
      f = enc_len - 1 - i
      filters = base_filters* (2**f)
      print("+++ Decoder filters {}".format(filters))
      s = enc[n]
      d = self.decoder_block(d, s, filters)
      n += 1

    """ Output """
    outputs = Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(d)

    """ Model """
    model = Model(inputs=[inputs], outputs=[outputs], name="MultiResUNET")

    return model

    
if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    print("=== config_file {}".format(config_file))

    config   = ConfigParser(config_file)

    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowMultiResUNet(config_file)
    

  except:
    traceback.print_exc()
    
