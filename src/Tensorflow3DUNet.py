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

# This is based on the code in the following web sites:

# 1. Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation
# https://github.com/mrkolarik/3D-brain-segmentation


# 2. Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation
# https://www.mdpi.com/2076-3417/9/3/404

# You can customize your TensorflowUnNet model by using a configration file
# Example: train.config

"""
[model]
image_width    = 256
image_height   = 256
image_channels = 3

num_classes    = 1
base_filters   = 16
num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.001
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import glob
import traceback
import numpy as np
import cv2

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv3D, Dropout, Conv3D, MaxPool3D

from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import elu, relu
from tensorflow.keras import Model
#from tensorflow.keras.losses import SparseCategoricalCrossentropy
#from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ConfigParser import ConfigParser
#from NucleiDataset import NucleiDataset
from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter

MODEL  = "model"
TRAIN  = "train"
BEST_MODEL_FILE = "best_model.h5"

"""
https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool3D
tf.keras.layers.MaxPool3D(
    pool_size=(2, 2, 2),
    strides=None,
    padding='valid',
    data_format=None,
    **kwargs
)

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3DTranspose
tf.keras.layers.Conv3DTranspose(
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding='valid',
    output_padding=None,
    data_format=None,
    dilation_rate=(1, 1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
"""

class Tensorflow3DUNet:

  def __init__(self, config_file):
    self.config    = ConfigParser(config_file)
    image_depth    = self.config.get(MODEL, "image_depth")

    image_height   = self.config.get(MODEL, "image_height")

    image_width    = self.config.get(MODEL, "image_width")
    image_channels = self.config.get(MODEL, "image_channels")

    num_classes    = self.config.get(MODEL, "num_classes")
    base_filters   = self.config.get(MODEL, "base_filters")
    num_layers     = self.config.get(MODEL, "num_layers")
    
    self.model     = self.create(num_classes, image_depth, image_height, image_width, image_channels, 
                            base_filters = base_filters, num_layers = num_layers)
    
    learning_rate  = self.config.get(MODEL, "learning_rate")

    self.optimizer = Adam(learning_rate = learning_rate, 
         beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, 
         amsgrad=False) 

    self.metrics = ["accuracy"]
    self.model.compile(optimizer = self.optimizer, loss="binary_crossentropy", metrics = self.metrics)
    self.model.summary()


  def create(self, num_classes, image_depth, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_depth, image_height, image_width, image_channels))
    s= Lambda(lambda x: x / 255)(inputs)

    # Encoder
    dropout_rate = self.config.get(MODEL, "dropout_rate")
    enc         = []
    kernel_size = (3, 3, 3)
    pool_size   = (2, 2, 2)
  
    for i in range(num_layers):
      filters = base_filters * (2**i)
      c = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(s)
      c = Dropout(dropout_rate * i)(c)
      c = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(c)
      print("--- enc shape {}".format(c.shape))
      if i < (num_layers-1):
        p = MaxPool3D(pool_size=pool_size)(c)

        s = p
      enc.append(c)
    
    enc_len = len(enc)
    enc.reverse()

    n = 0
    c = enc[n]
    # --- Decoder
    for i in range(num_layers-1):
      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      u = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c)
      n += 1
      print("shape u {}".format(u.shape))
      print("shape enc[n] {}".format(enc[n].shape))

      u = concatenate([u, enc[n]], axis=4)
      u = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(u)
      u = Dropout(dropout_rate * f)(u)
      u = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal',padding='same')(u)
      c  = u

    # outouts
    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c)

    # create Model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


  def train(self, x_train, y_train): 
    batch_size = self.config.get(TRAIN, "batch_size")
    epochs     = self.config.get(TRAIN, "epochs")
    patience   = self.config.get(TRAIN, "patience")
    eval_dir   = self.config.get(TRAIN, "eval_dir")
    model_dir  = self.config.get(TRAIN, "model_dir")

    if os.path.exists(model_dir):
      shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    weight_filepath   = os.path.join(model_dir, BEST_MODEL_FILE)

    early_stopping = EarlyStopping(patience=patience, verbose=1)
    check_point    = ModelCheckpoint(weight_filepath, verbose=1, save_best_only=True)
    epoch_change   = EpochChangeCallback(eval_dir)

    results = self.model.fit(x_train, y_train, 
                    validation_split=0.2, batch_size=batch_size, epochs=epochs, 
                    callbacks=[early_stopping, check_point, epoch_change],
                    verbose=1)

  # 2023/05/05 Added newly.    
  def infer(self, input_dir, output_dir, expand=True):
    writer       = GrayScaleImageWriter()
    # We are intereseted in png and jpg files.
    # Sorry the bmp or tif files are ignored.
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    depth        = self.config.get(MODEL, "image_depth")
    width        = self.config.get(MODEL, "image_width")
    height       = self.config.get(MODEL, "image_height")

    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      img      = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      h = img.shape[0]
      w = img.shape[1]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height))
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
      # Resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      # Probably, this is a natural way for all humans. 
      writer.save_resized(image, (w, h), output_dir, name)
     

  def predict(self, images, expand=True):
    model_dir  = self.config.get(TRAIN, "model_dir")

    if not os.path.exists(model_dir):
      raise Exception("Not found " + model_dir)
    weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)

    self.model.load_weights(weight_filepath)
    print("=== Loaded weight_file {}".format(weight_filepath))
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    


  def evaluate(self, x_test, y_test): 
    model_dir  = self.config.get(TRAIN, "model_dir")

    if not os.path.exists(model_dir):
      raise Exception("Not found " + model_dir)
    weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)

    self.model.load_weights(weight_filepath)
    print("=== Loaded weight_file {}".format(weight_filepath))
    score = self.model.evaluate(x_test, y_test, verbose=1)
    print("Test loss    :{}".format(score[0]))     
    print("Test accuracy:{}".format(score[1]))
     
    
if __name__ == "__main__":
  try:
    config_file    = "./model.config"
    config   = ConfigParser(config_file)
    depth    = config.get(MODEL, "image_depth")
    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")
    channels = config.get(MODEL, "image_channels")

    #if not (width == height and  height % 128 == 0 and width % 128 == 0):
    #  raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = Tensorflow3DUNet(config_file)
    
    """
    resized_image     = (height, width, channels)
    train_datapath    = "./stage1_train/"
    dataset           = NucleiDataset(resized_image)

    x_train, y_train  = dataset.create(train_datapath, has_mask=True)

    model.train(x_train, y_train)
    """

  except:
    traceback.print_exc()
    
