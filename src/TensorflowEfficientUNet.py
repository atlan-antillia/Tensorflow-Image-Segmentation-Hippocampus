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

# TensorflowEfficientUNet.py
# 2023/07/30 to-arai

# Some methods in TensorflowEfficientNetUNet have been taken from the following code.
# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/efficientnetb0_unet.py


import os
import sys

import traceback


import numpy as np
from glob import glob

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

#from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
#CollectiveAllReduceExtended._enable_check_health = False

from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, MaxPool2D, 
                                     Conv2DTranspose, Concatenate, Input)

from tensorflow.keras.applications import EfficientNetB0

from ConfigParser import ConfigParser
from TensorflowUNet import TensorflowUNet

from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss

MODEL = "model"
EVAL  = "eval"
INFER = "infer"

class TensorflowEfficientUNet(TensorflowUNet) :

  def __init__(self, config_file):
    #tf.keras.backend.clear_session()
    self.efficientnet = "B0"
    config = ConfigParser(config_file)
    self.efficientnet = config.get(MODEL, "efficientnet", dvalue="B0")
    
    super().__init__(config_file)
    

  def get_encoder(self, weights, inputs):
    name = self.efficientnet
    print("=== get_encoder EfficientNet{}".format(name))
    if name == "B0":
      return EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs)

    elif name == "B1":
      from tensorflow.keras.applications import EfficientNetB1
      return EfficientNetB1(include_top=False, weights=weights, input_tensor=inputs)

    elif name == "B2":
      from tensorflow.keras.applications import EfficientNetB2
      return EfficientNetB2(include_top=False, weights=weights, input_tensor=inputs)

    elif name == "B3":
      from tensorflow.keras.applications import EfficientNetB3
      return EfficientNetB3(include_top=False, weights=weights, input_tensor=inputs)

    elif name == "B4":
      from tensorflow.keras.applications import EfficientNetB4
      return EfficientNetB4(include_top=False, weights=weights, input_tensor=inputs)

    elif name == "B5":
      from tensorflow.keras.applications import EfficientNetB5
      return EfficientNetB5(include_top=False, weights=weights, input_tensor=inputs)

    elif name == "B6":
      from tensorflow.keras.applications import EfficientNetB7
      return EfficientNetB6(include_top=False, weights=weights, input_tensor=inputs)
    
    elif name == "B7":
      from tensorflow.keras.applications import EfficientNetB7
      return EfficientNetB7(include_top=False, weights=weights, input_tensor=inputs)

    else:
      raise Exception("Invalid EfficientNet name " + name)
    

  # This method has been taken from the following code.
  # https://github.com/he44/EfficientNet-UNet/blob/master/efficientnet_unet/build_eunet.py
  def CBAR_block(self, input, num_filters):
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    xd = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1)(input)
    x = tf.keras.layers.Add()([x, xd])

    return x
  
  # This method has been taken from the following code.
  # https://github.com/he44/EfficientNet-UNet/blob/master/efficientnet_unet/build_eunet.py
  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
   
    print("=== TensorflowEfficientNetUNet.create ")
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs  = Input((image_height, image_width, image_channels))
    weights = "imagenet"
    encoder = self.get_encoder(weights, inputs)
    
    new_input =encoder.input
    # encoder output, we won't use the top_conv (which has 1280 filters)
    # let's just use 7a bn, which is 7 x 7 x 320
    encoder_output = encoder.get_layer(name='block7a_project_bn').output

    # filter number for the bottleneck
    fn_bottle_neck = encoder_output.shape[-1]
    bottleneck = self.CBAR_block(encoder_output, fn_bottle_neck)

    # Decoder block 1
    c1 =encoder.get_layer(name='block5c_drop').output
    fn_1 = c1.shape[-1]
    upsampling1 = tf.keras.layers.UpSampling2D()(bottleneck)
    concatenation1 = tf.keras.layers.concatenate(
            [upsampling1, c1], axis=3)
    decoder1 = self.CBAR_block(concatenation1, fn_1)

    # Decoder block 2
    c2 = encoder.get_layer(name='block3b_drop').output
    fn_2 = c2.shape[-1]
    upsampling2 = tf.keras.layers.UpSampling2D()(decoder1)
    concatenation2 = tf.keras.layers.concatenate(
            [upsampling2, c2], axis=3)
    decoder2 = self.CBAR_block(concatenation2, fn_2)

    # Decoder block 3
    c3 = encoder.get_layer(name='block2b_drop').output
    fn_3 = c3.shape[-1]
    upsampling3 = tf.keras.layers.UpSampling2D()(decoder2)
    concatenation3 = tf.keras.layers.concatenate(
            [upsampling3, c3], axis=3)
    decoder3 = self.CBAR_block(concatenation3, fn_3)

    # Decoder block 4
    # 1a does not have dropout
    c4 = encoder.get_layer(name='block1a_project_bn').output
    fn_4 = c4.shape[-1]
    upsampling4 = tf.keras.layers.UpSampling2D()(decoder3)
    concatenation4 = tf.keras.layers.concatenate(
            [upsampling4, c4], axis=3)
    decoder4 = self.CBAR_block(concatenation4, fn_4)

    # Decoder block 5
    # the only layer with original shape is input...
    fn_5 = fn_4 # let's resuse this filter number for now
    upsampling5 = tf.keras.layers.UpSampling2D()(decoder4)
    concatenation5 = tf.keras.layers.concatenate(
            [upsampling5, new_input], axis=3)
    decoder5 = self.CBAR_block(concatenation5, fn_5)

    # Now we can add in the output portion
    if num_classes == 1 or num_classes == 2:
        final_filter_num = 1
        final_activation = 'sigmoid'
    else:
        final_filter_num = num_classes
        final_activation = 'softmax'
    new_output = tf.keras.layers.Conv2D(filters=final_filter_num, kernel_size=1, activation=final_activation)(decoder5)

    print("output shape", new_output.shape)
    
    model = tf.keras.Model(inputs=new_input, outputs=new_output)
    return model
    

if __name__ == "__main__":
  try:
    config_file = "./train_eval_inf.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    model = TensorflowEfficientUNet(config_file)
    
  except:
    traceback.print_exc()
