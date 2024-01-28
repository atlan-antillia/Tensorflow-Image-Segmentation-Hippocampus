"""
Title: Highly accurate boundaries segmentation using BASNet
Author: [Hamid Ali](https://github.com/hamidriasat)
Date created: 2023/05/30
Last modified: 2023/07/13
Description: Boundaries aware segmentation model trained on the DUTS dataset.
Accelerator: GPU
"""

"""
## Introduction

Deep semantic segmentation algorithms have improved a lot recently, but still fails to correctly
predict pixels around object boundaries. In this example we implement
**Boundary-Aware Segmentation Network (BASNet)**, using two stage predict and refine
architecture, and a hybrid loss it can predict highly accurate boundaries and fine structures
for image segmentation.

### References:

- [Boundary-Aware Segmentation Network for Mobile and Web Applications](https://arxiv.org/abs/2101.04704)
- [BASNet Keras Implementation](https://github.com/hamidriasat/BASNet/tree/basnet_keras)
- [Learning to Detect Salient Objects with Image-level Supervision](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_Learning_to_Detect_CVPR_2017_paper.html)
"""

"""
## Download the Data

We will use the [DUTS-TE](http://saliencydetection.net/duts/) dataset for training. It has 5,019
images but we will use 140 for training and validation to save notebook running time. DUTS is
relatively large salient object segmentation dataset. which contain diversified textures and
structures common to real-world images in both foreground and background.
"""

"""shell
wget http://saliencydetection.net/duts/download/DUTS-TE.zip
unzip -q DUTS-TE.zip
"""

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

# TensorflowBasNet.py
# 2023/12/10 to-arai

# Some methods in this TensorflowBasNet have been taken from the following web site
# https://github.com/keras-team/keras-io/blob/master/examples/vision/basnet_segmentation.py

import os
from re import A
import sys

import traceback


import numpy as np
from glob import glob
import keras_cv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import AdamW

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

import sys


from ConfigParser import ConfigParser
from TensorflowUNet import TensorflowUNet

from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss



MODEL = "model"
EVAL  = "eval"
INFER = "infer"

class TensorflowBasNet(TensorflowUNet) :

  def __init__(self, config_file):
    #super().__init__(config_file)
    
    self.model_loaded = False

    self.config_file = config_file
    self.config = ConfigParser(config_file)

    num_classes     = self.config.get(MODEL, "num_classes")
    image_width     = self.config.get(MODEL, "image_width")
    image_height    = self.config.get(MODEL, "image_height")
    image_channels  = self.config.get(MODEL, "image_channels")
    base_filters    = self.config.get(MODEL, "base_filters")
    num_layers      = self.config.get(MODEL, "num_layers")
    self.model      = self.create(num_classes, image_height, image_width, image_channels,
                         base_filters = base_filters, num_layers = num_layers)
  
    learning_rate    = self.config.get(MODEL, "learning_rate")
    clipvalue        = self.config.get(MODEL, "clipvalue", dvalue=0.5)
    
    # 2023/11/10
    optimizer = self.config.get(MODEL, "optimizer", dvalue="AdamW")
    if optimizer == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
         beta_1=0.9, 
         beta_2=0.999, 
         weight_decay=0.0,     
         clipvalue=clipvalue,  
         amsgrad=False)
      print("=== Optimizer Adam learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
    
    elif optimizer == "AdamW":
      # 2023/11/10  Adam -> AdamW (tensorflow 2.14.0~)
      self.optimizer = tf.keras.optimizers.AdamW(learning_rate = learning_rate,
         clipvalue=clipvalue,
         )
      print("=== Optimizer AdamW learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
        
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy
    
    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names, ant eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
    
  
    self.model.compile(optimizer= self.optimizer, loss =   self.loss, metrics= self.metrics)


"""
## Building the BASNet Model

BASNet comprises of a predict-refine architecture and a hybrid loss. The predict-refine
architecture consists of a densely supervised encoder-decoder network and a residual refinement
module, which are respectively used to predict and refine a segmentation probability map.

![](https://i.imgur.com/8jaZ2qs.png)
"""


  def basic_block(self, x_input, filters, stride=1, down_sample=None, activation=None):
    """Creates a residual(identity) block with two 3*3 convolutions."""
    residual = x_input

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(
        x_input
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="same", use_bias=False)(
        x
    )
    x = layers.BatchNormalization()(x)

    if down_sample is not None:
        residual = down_sample

    x = layers.Add()([x, residual])

    if activation is not None:
        x = layers.Activation(activation)(x)

    return x


  def convolution_block(self, x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


  def segmentation_head(self, x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = layers.Conv2D(out_classes, kernel_size=(3, 3), padding="same")(x_input)

    if final_size is not None:
        x = layers.Resizing(final_size[0], final_size[1])(x)

    return x


  def get_resnet_block(self, _resnet, block_num):
    """Extract and return ResNet-34 block."""
    resnet_layers = [3, 4, 6, 3]  # ResNet-34 layer sizes at different block.
    return keras.models.Model(
        inputs=_resnet.get_layer(f"v2_stack_{block_num}_block1_1_conv").input,
        outputs=_resnet.get_layer(
            f"v2_stack_{block_num}_block{resnet_layers[block_num]}_add"
        ).output,
        name=f"resnet34_block{block_num + 1}",
    )


"""
## Prediction Module

Prediction module is a heavy encoder decoder structure like U-Net. The encoder includes an input
convolutional layer and six stages. First four are adopted from ResNet-34 and rest are basic
res-blocks. Since first convolution and pooling layer of ResNet-34 is skipped so we will use
`get_resnet_block()` to extract first four blocks. Both bridge and decoder uses three
convolutional layers with side outputs. The module produces seven segmentation probability
maps during training, with the last one considered the final output.
"""


  def create(self, input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 6

    x_input = layers.Input(input_shape)

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    resnet = keras_cv.models.ResNet34Backbone(
        include_rescaling=False,
    )

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            x = get_resnet_block(resnet, i)(x)
            encoder_blocks.append(x)
            x = layers.Activation("relu")(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            shape = keras.backend.int_shape(x)
            x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)

        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block, out_classes, input_shape[:2])
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)


"""
## Residual Refinement Module

Refinement Modules (RMs), designed as a residual block aim to refines the coarse(blurry and noisy
boundaries) segmentation maps generated by prediction module. Similar to prediction module it's
also an encode decoder structure but with light weight 4 stages, each containing one
`convolutional block()` init. At the end it adds both coarse and residual output to generate
refined output.
"""


def basnet_rrm(base_model, out_classes):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 4
    filters = 64

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        shape = keras.backend.int_shape(x)
        x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x, out_classes, None)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=[base_model.input], outputs=[x])


"""
## Combine Predict and Refinement Module
"""


def basnet(input_shape, out_classes):
    """BASNet, it's a combination of two modules
    Prediction Module and Residual Refinement Module(RRM)."""

    # Prediction model.
    predict_model = basnet_predict(input_shape, out_classes)
    # Refinement model.
    refine_model = basnet_rrm(predict_model, out_classes)

    output = [refine_model.output]  # Combine outputs.
    output.extend(predict_model.output)

    output = [layers.Activation("sigmoid")(_) for _ in output]  # Activations.

    return keras.models.Model(inputs=[predict_model.input], outputs=output)


"""
## Hybrid Loss

Another important feature of BASNet is its hybrid loss function, which is a combination of
binary cross entropy, structural similarity and intersection-over-union losses, which guide
the network to learn three-level (i.e., pixel, patch and map level) hierarchy representations.
"""


class BasnetLoss(keras.losses.Loss):
    """BASNet hybrid loss."""

    def __init__(self, **kwargs):
        super().__init__(name="basnet_loss", **kwargs)
        self.smooth = 1.0e-9

        # Binary Cross Entropy loss.
        self.cross_entropy_loss = keras.losses.BinaryCrossentropy()
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        #  Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = backend.sum(backend.abs(y_true * y_pred), axis=[1, 2, 3])
        union = backend.sum(y_true, [1, 2, 3]) + backend.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return backend.mean(
            (intersection + self.smooth) / (union + self.smooth), axis=0
        )

    def call(self, y_true, y_pred):
        cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred)

        ssim_value = self.ssim_value(y_true, y_pred, max_val=1)
        ssim_loss = backend.mean(1 - ssim_value + self.smooth, axis=0)

        iou_value = self.iou_value(y_true, y_pred)
        iou_loss = 1 - iou_value

        # Add all three losses.
        return cross_entropy_loss + ssim_loss + iou_loss


basnet_model = basnet(
    input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], out_classes=OUT_CLASSES
)  # Create model.
basnet_model.summary()  # Show model summary.

optimizer = keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
# Compile model.
basnet_model.compile(
    loss=BasnetLoss(),
    optimizer=optimizer,
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
)

"""
### Train the Model
"""

basnet_model.fit(train_dataset, validation_data=val_dataset, epochs=1)

"""
### Visualize Predictions

In paper BASNet was trained on DUTS-TR dataset, which has 10553 images. Model was trained for 400k
iterations with a batch size of eight and without a validation dataset. After training model was
evaluated on DUTS-TE dataset and achieved a mean absolute error of `0.042`.

Since BASNet is a deep model and cannot be trained in a short amount of time which is a
requirement for keras example notebook, so we will load pretrained weights from [here](https://github.com/hamidriasat/BASNet/tree/basnet_keras)
to show model prediction. Due to computer power limitation this model was trained for 120k
iterations but it still demonstrates its capabilities. For further details about
trainings parameters please check given link.
"""

"""shell
!gdown 1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg
"""


def normalize_output(prediction):
    max_value = np.max(prediction)
    min_value = np.min(prediction)
    return (prediction - min_value) / (max_value - min_value)


# Load weights.
basnet_model.load_weights("./basnet_weights.h5")

"""
### Make Predictions
"""

for image, mask in val_dataset.take(1):
    pred_mask = basnet_model.predict(image)
    display([image[0], mask[0], normalize_output(pred_mask[0][0])])

if __name__ == "__main__":
  try:
    config_file = "./train_eval_inf.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    model = TensorflowBasNet(config_file)
    
  except:
    traceback.print_exc()
