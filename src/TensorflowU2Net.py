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

# TensorflowSwinUNet.py
# 2023/07/04 to-arai

# Some methods on SwinTransformer in this TensorflowSwinUNet class have been taken from the following web-sites.
#
# keras-vision-transformer
# https://github.com/yingkaisha/keras-vision-transformer
#  MIT license

# keras-unet-collection
# https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection
#  MIT license

#Oxford IIIT image segmentation with SwinUNET
#https://github.com/yingkaisha/keras-vision-transformer/blob/main/examples/Swin_UNET_oxford_iiit.ipynb

import os
import sys

import traceback


import numpy as np
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

import sys

# keras-unet-collection
# https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection

from keras_unet_collection.layer_utils import *
from keras_unet_collection.transformer_layers import patch_extract, patch_embedding, SwinTransformerBlock, patch_merging, patch_expanding


from ConfigParser import ConfigParser
from TensorflowUNet import TensorflowUNet

from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss



MODEL = "model"
EVAL  = "eval"
INFER = "infer"

class TensorflowU2Net(TensorflowUNet) :

  def __init__(self, config_file):
    #super().__init__(config_file)
    
    self.model_loaded = False

    self.config_file = config_file
    self.config = ConfigParser(config_file)
    self.filter_num_begin = self.config.get(MODEL, "filter_num_begin", dvalue=128)
    # number of channels in the first downsampling block; it is also the number of embedded dimensions
    
    self.depth            = self.config.get(MODEL, "depth", dvalue=4)
    # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    
    self.stack_num_down   = self.config.get(MODEL, "stack_num_down", dvalue=2)
    # number of Swin Transformers per downsampling level

    self.stack_num_up     = self.config.get(MODEL, "stack_num_up", dvalue=2)
    # number of Swin Transformers per upsampling level
    
    self.patch_size       = self.config.get(MODEL, "patch_size", dvalue=(4,4))
    # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.

    self.num_heads        = self.config.get(MODEL, "num_heads",dvalue=[4, 8, 8, 8])
    # number of attention heads per down/upsampling level
    
    self.window_size      = self.config.get(MODEL, "window_size")
    #the size of attention window per down/upsampling level
    
    self.num_mlp          = self.config.get(MODEL, "num_mlp", dvalue=512)
    # number of MLP nodes within the Transformer
    
    self.shift_window     = self.config.get(MODEL, "shift_window")
    #Apply window shifting, i.e., Swin-MSA

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
    
    # Optimization
    # <---- !!! gradient clipping is important
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=clipvalue)
    self.optimizer = Adam(learning_rate = learning_rate, 
         beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, 
         clipvalue=clipvalue,  
         amsgrad=False)
    
    print("=== Optimizer Adam learning_rate {} clipvalue {}".format(learning_rate, clipvalue))

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


  # The following some methods have been taken from
  # https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_model_u2net_2d.py

  def RSU(self, X, channel_in, channel_out, depth=5, activation='ReLU', batch_norm=True, pool=True, unpool=True, name='RSU'):
    '''
    The Residual U-blocks (RSU).
    
    RSU(X, channel_in, channel_out, depth=5, activation='ReLU', batch_norm=True, pool=True, unpool=True, name='RSU')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        X: input tensor.
        channel_in: number of intermediate channels.
        channel_out: number of output channels.
        depth: number of down- and upsampling levels.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.   
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    pool_size = 2
    
    X_skip = []
    
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, 
                   dilation_rate=1, activation=activation, batch_norm=batch_norm, 
                   name='{}_in'.format(name))
    X_skip.append(X)

    X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_down_0'.format(name))
    X_skip.append(X)
    
    for i in range(depth):
        
        X = encode_layer(X, channel_in, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_{}'.format(name, i))
        
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i+1))
        X_skip.append(X)

    X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, 
               dilation_rate=2, activation=activation, batch_norm=batch_norm, 
               name='{}_up_0'.format(name))
    
    X_skip = X_skip[::-1]
    
    for i in range(depth):
        
        X = concatenate([X, X_skip[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=1, 
                       activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i+1))
        
        X = decode_layer(X, channel_in, pool_size, unpool, 
                         activation=activation, batch_norm=batch_norm, name='{}_decode_{}'.format(name, i))
        
    X = concatenate([X, X_skip[depth]], axis=-1, name='{}_concat_out'.format(name))

    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_out'.format(name))
    X = add([X, X_skip[-1]], name='{}_out_add'.format(name))
    return X 

  def RSU4F(self, X, channel_in, channel_out, dilation_num=[1, 2, 4, 8], activation='ReLU', batch_norm=True, name='RSU4F'):
    '''
    The Residual U-blocks with dilated convolutional kernels (RSU4F).
    
    RSU4F(X, channel_in, channel_out, dilation_num=[1, 2, 4, 8], activation='ReLU', batch_norm=True, name='RSU4F')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        X: input tensor.
        channel_in: number of intermediate channels.
        channel_out: number of output channels.
        dilation_num: an iterable that defines dilation rates of convolutional layers.
                      Qin et al. (2020) suggested `[1, 2, 4, 8]`.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    
    X_skip = []    
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_in'.format(name))
    X_skip.append(X)
    
    for i, d in enumerate(dilation_num):

        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=d, 
                       activation=activation, batch_norm=batch_norm, name='{}_down_{}'.format(name, i))
        X_skip.append(X)

    X_skip = X_skip[:-1][::-1]
    dilation_num = dilation_num[:-1][::-1]
    
    for i, d in enumerate(dilation_num[:-1]):

        X = concatenate([X, X_skip[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = CONV_stack(X, channel_in, kernel_size=3, stack_num=1, dilation_rate=d, 
                       activation=activation, batch_norm=batch_norm, name='{}_up_{}'.format(name, i))

    X = concatenate([X, X_skip[2]], axis=-1, name='{}_concat_out'.format(name))
    X = CONV_stack(X, channel_out, kernel_size=3, stack_num=1, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name='{}_out'.format(name))
    
    return add([X, X_skip[-1]], name='{}_out_add'.format(name))

  def u2net_2d_base(self, input_tensor, 
                      filter_num_down, filter_num_up, 
                      filter_mid_num_down, filter_mid_num_up, 
                      filter_4f_num, filter_4f_mid_num, activation='ReLU',
                      batch_norm=False, pool=True, unpool=True, name='u2net'):
    
    '''
    The base of U^2-Net
    
    u2net_2d_base(input_tensor, 
                  filter_num_down, filter_num_up, 
                  filter_mid_num_down, filter_mid_num_up, 
                  filter_4f_num, filter_4f_mid_num, activation='ReLU',
                  batch_norm=False, pool=True, unpool=True, name='u2net')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., keras.layers.Inpyt((None, None, 3))
        filter_num_down: a list that defines the number of RSU output filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512]`.
                         the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`                         
        filter_mid_num_down: a list that defines the number of RSU intermediate filters for each 
                             downsampling level. e.g., `[16, 32, 64, 128]`.
                             * RSU intermediate and output filters must paired, i.e., list with the same length.
                             * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_mid_num_up: a list that defines the number of RSU intermediate filters for each 
                           upsampling level. e.g., `[16, 32, 64, 128]`.
                           * RSU intermediate and output filters must paired, i.e., list with the same length.
                           * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_4f_num: a list that defines the number of RSU-4F output filters for each 
                       downsampling and bottom level. e.g., `[512, 512]`.
                       the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`.
        filter_4f_mid_num: a list that defines the number of RSU-4F intermediate filters for each 
                           downsampling and bottom level. e.g., `[256, 256]`.
                           * RSU-4F intermediate and output filters must paired, i.e., list with the same length.
                           * RSU-4F intermediate filters numbers are expected to be smaller than output filters numbers.    
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        deep_supervision: True for a model that supports deep supervision. Details see Qin et al. (2020).
        name: prefix of the created keras layers.
        
    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from 
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different, 
          with first tensor has the smallest size.
        
    * Dilation rates of RSU4F layers are fixed to `[1, 2, 4, 8]`.
    * Downsampling is achieved through maxpooling in Qin et al. (2020), 
      and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation in Qin et al. (2020), 
      and can be replaced by transpose convolutional layers here.
    
    '''
    
    pool_size = 2
    
    X_skip = []; X_out = []; OUT_stack = []
    depth_backup = []
    depth_ = len(filter_num_down)
    
    X = input_tensor
    
    X = self.RSU(X, filter_mid_num_down[0], filter_num_down[0], depth=depth_+1, activation=activation, 
            batch_norm=batch_norm, pool=pool, unpool=unpool, name='{}_in'.format(name))
    X_skip.append(X)
    
    depth_backup.append(depth_+1)
    
    for i, f in enumerate(filter_num_down[1:]):
        
        X = encode_layer(X, f, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_{}'.format(name, i))
        
        X = self.RSU(X, filter_mid_num_down[i+1], f, depth=depth_-i, activation=activation, 
                batch_norm=batch_norm, pool=pool, unpool=unpool, name='{}_down_{}'.format(name, i))
        
        depth_backup.append(depth_-i)
        
        X_skip.append(X)

    for i, f in enumerate(filter_4f_num):
        
        X = encode_layer(X, f, pool_size, pool, activation=activation, 
                         batch_norm=batch_norm, name='{}_encode_4f_{}'.format(name, i))
        
        X = self.RSU4F(X, filter_4f_mid_num[i], f, activation=activation, 
                  batch_norm=batch_norm, name='{}_down_4f_{}'.format(name, i))
        X_skip.append(X)
        
    X_out.append(X)
    
    # ---------- #
    X_skip = X_skip[:-1][::-1]
    depth_backup = depth_backup[::-1]
    
    filter_num_up = filter_num_up[::-1]
    filter_mid_num_up = filter_mid_num_up[::-1]
    
    filter_4f_num = filter_4f_num[:-1][::-1]
    filter_4f_mid_num = filter_4f_mid_num[:-1][::-1]
    
    tensor_count = 0
    for i, f in enumerate(filter_4f_num):
        
        X = decode_layer(X, f, pool_size, unpool, 
                         activation=activation, batch_norm=batch_norm, name='{}_decode_4f_{}'.format(name, i))
        
        X = concatenate([X, X_skip[tensor_count]], axis=-1, name='{}_concat_4f_{}'.format(name, i))
        
        X = self.RSU4F(X, filter_4f_mid_num[i], f, activation=activation, 
                  batch_norm=batch_norm, name='{}_up_4f_{}'.format(name, i))
        X_out.append(X)
        
        tensor_count += 1
    
    for i, f in enumerate(filter_num_up):
        
        X = decode_layer(X, f, pool_size, unpool, 
                         activation=activation, batch_norm=batch_norm, name='{}_decode_{}'.format(name, i))
        
        X = concatenate([X, X_skip[tensor_count]], axis=-1, name='{}_concat_{}'.format(name, i))
        
        X = self.RSU(X, filter_mid_num_up[i], f, depth=depth_backup[i], 
                activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, name='{}_up_{}'.format(name, i))
        X_out.append(X)
        
        tensor_count += 1

    return X_out


  def create(self, 
             n_labels, image_height, image_width, image_channels,
             base_filters = 16, num_layers = 6,
             #input_size, 
             #filter_num_down, 
             filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
             filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
             batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='u2net'):
    
    '''
    U^2-Net
    
    u2net_2d(input_size, n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
             filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
             batch_norm=False, deep_supervision=False, name='u2net')
    
    ----------
    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R. and Jagersand, M., 2020. 
    U2-Net: Going deeper with nested U-structure for salient object detection. 
    Pattern Recognition, 106, p.107404.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_down: a list that defines the number of RSU output filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512]`.
                         the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`                         
        filter_mid_num_down: a list that defines the number of RSU intermediate filters for each 
                             downsampling level. e.g., `[16, 32, 64, 128]`.
                             * RSU intermediate and output filters must paired, i.e., list with the same length.
                             * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_mid_num_up: a list that defines the number of RSU intermediate filters for each 
                           upsampling level. e.g., `[16, 32, 64, 128]`.
                           * RSU intermediate and output filters must paired, i.e., list with the same length.
                           * RSU intermediate filters numbers are expected to be smaller than output filters numbers.
        filter_4f_num: a list that defines the number of RSU-4F output filters for each 
                       downsampling and bottom level. e.g., `[512, 512]`.
                       the network depth is expected as `len(filter_num_down) + len(filter_4f_num)`.
        filter_4f_mid_num: a list that defines the number of RSU-4F intermediate filters for each 
                           downsampling and bottom level. e.g., `[256, 256]`.
                           * RSU-4F intermediate and output filters must paired, i.e., list with the same length.
                           * RSU-4F intermediate filters numbers are expected to be smaller than output filters numbers.         
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.           
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.  
        deep_supervision: True for a model that supports deep supervision. Details see Qin et al. (2020).
        name: prefix of the created keras layers.
        
    Output
    ----------
        model: a keras model.
    
    * Automated hyper-parameter estimation will produce a slightly larger network, different from that of Qin et al. (2020).
    * Dilation rates of RSU4F layers are fixed to `[1, 2, 4, 8]`.
    * The default output activation is sigmoid, the same as in Qin et al. (2020).
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers.
    
    '''
    # filter_num_down =`[64, 128, 256, 512]`
    filter_num_down = []
    num = base_filters
    for n in range(num_layers):
      filter_num_down +=[num]
      num = num *2
    print("=== filter_num_down {}".format(filter_num_down))

    verbose = False
    
    if filter_num_up == 'auto':
        verbose = True
        filter_num_up = filter_num_down
        
    if filter_mid_num_down == 'auto':
        verbose = True
        filter_mid_num_down = [num//4 for num in filter_num_down]
        
    if filter_mid_num_up == 'auto':
        verbose = True
        filter_mid_num_up = filter_mid_num_down
        
    if filter_4f_num == 'auto':
        verbose = True
        filter_4f_num = [filter_num_down[-1], filter_num_down[-1]]
        
    if filter_4f_mid_num == 'auto':
        verbose = True
        filter_4f_mid_num = [num//2 for num in filter_4f_num]
        
    if verbose:
        print('Automated hyper-parameter determination is applied with the following details:\n----------')
        print('\tNumber of RSU output channels within downsampling blocks: filter_num_down = {}'.format(filter_num_down))
        print('\tNumber of RSU intermediate channels within downsampling blocks: filter_mid_num_down = {}'.format(filter_mid_num_down))
        print('\tNumber of RSU output channels within upsampling blocks: filter_num_up = {}'.format(filter_num_up))
        print('\tNumber of RSU intermediate channels within upsampling blocks: filter_mid_num_up = {}'.format(filter_mid_num_up))        
        print('\tNumber of RSU-4F output channels within downsampling and bottom blocks: filter_4f_num = {}'.format(filter_4f_num))
        print('\tNumber of RSU-4F intermediate channels within downsampling and bottom blocks: filter_4f_num = {}'.format(filter_4f_mid_num))
        print('----------\nExplicitly specifying keywords listed above if their "auto" settings do not satisfy your needs')
        
    print("----------\nThe depth of u2net_2d = len(filter_num_down) + len(filter_4f_num) = {}".format(len(filter_num_down)+len(filter_4f_num)))
    
    X_skip = []; X_out = []; OUT_stack = []
    depth_backup = []
    depth_ = len(filter_num_down)
    
    #IN = Input(shape=input_size) 
    inputs = Input(image_height, image_width, image_channels)
    IN = Lambda(lambda x: x / 255)(inputs)

    # base (before conv + activation + upsample)
    X_out = self.u2net_2d_base(IN, 
                          filter_num_down, filter_num_up, 
                          filter_mid_num_down, filter_mid_num_up, 
                          filter_4f_num, filter_4f_mid_num, activation=activation, 
                          batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)
    
    # output layers
    X_out = X_out[::-1]
    L_out = len(X_out)
    
    X = CONV_output(X_out[0], n_labels, kernel_size=3, activation=output_activation, 
                    name='{}_output_sup0'.format(name))
    OUT_stack.append(X)
    
    for i in range(1, L_out):
        
        pool_size = 2**(i)
        
        X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i))(X_out[i])
        
        X = decode_layer(X, n_labels, pool_size, unpool, 
                         activation=None, batch_norm=False, name='{}_sup{}'.format(name, i))
        
        if output_activation:
            if output_activation == 'Sigmoid':
                X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i))(X)
            else:
                activation_func = eval(output_activation)
                X = activation_func(name='{}_output_sup{}_activation'.format(name, i))(X)
                
        OUT_stack.append(X)
        
    D = concatenate(OUT_stack, axis=-1, name='{}_output_concat'.format(name))
    
    D = CONV_output(D, n_labels, kernel_size=1, activation=output_activation, 
                    name='{}_output_final'.format(name))
    
    if deep_supervision:
        
        OUT_stack.append(D)
        print('----------\ndeep_supervision = True\nnames of output tensors are listed as follows ("sup0" is the shallowest supervision layer;\n"final" is the final output layer):\n')
        
        if output_activation == None:
            if unpool is False:
                for i in range(L_out):
                    print('\t{}_output_sup{}_trans_conv'.format(name, i))
            else:
                for i in range(L_out):
                    print('\t{}_output_sup{}_unpool'.format(name, i))
                
            print('\t{}_output_final'.format(name))
        
        else:        
            for i in range(L_out):
                print('\t{}_output_sup{}_activation'.format(name, i))
                
            print('\t{}_output_final_activation'.format(name))
            
        model = Model([IN,], OUT_stack)
        
    else:
        model = Model([IN,], [D,])
        
    return model

    

if __name__ == "__main__":
  try:
    config_file = "./train_eval_inf.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    model = TensorflowU2Net(config_file)
    
  except:
    traceback.print_exc()
