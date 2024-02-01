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

"""
#https://github.com/yingkaisha/keras-vision-transformer/blob/main/examples/Swin_UNET_oxford_iiit.ipynb

filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
stack_num_down = 2         # number of Swin Transformers per downsampling level
stack_num_up = 2           # number of Swin Transformers per upsampling level
    
patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    
num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
num_mlp = 512              # number of MLP nodes within the Transformer
shift_window=True          # Apply window shifting, i.e., Swin-MSA
"""

MODEL = "model"
EVAL  = "eval"
INFER = "infer"

class TensorflowSwinUNet(TensorflowUNet) :

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
    
    # 2023/11/10
    optimizer = self.config.get(MODEL, "optimizer", dvalue="AdamW")
    if optimizer == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
         beta_1=0.9, 
         beta_2=0.999, 
         #epsilon=None,        #2023/11/10 epsion=None is not allowed
         #weight_decay=0.0,     #2023/11/10 decay -> weight_decay
         clipvalue=clipvalue,  #2023/06/26
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

  # The following some methods have been taken from
  # https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_model_swin_unet_2d.py
  def swin_transformer_stack(self, X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    print("=== TensorflowSwinUnet.swin_transformer_stack ===")
    mlp_drop_rate  = 0 # Droupout after each MLP layer
    attn_drop_rate = 0 # Dropout after Swin-Attention
    proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = SwinTransformerBlock(dim=embed_dim, num_patch=num_patch, num_heads=num_heads, 
                    window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate, 
                    name='name{}'.format(i))(X)
    return X


  def swin_unet_2d_base(self, input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
               patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):
    '''
    The base of SwinUNET.
    
    ----------
    Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021. 
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num_begin: number of channels in the first downsampling block; 
                          it is also the number of embedded dimensions.
        depth: the depth of Swin-UNET, e.g., depth=4 means three down/upsampling levels and a bottom level.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of Swin-Transformers) ----------
        
        patch_size: The size of extracted patches, 
                    e.g., patch_size=(2, 2) means 2-by-2 patches
                    *Height and width of the patch must be equal.
                    
        num_heads: number of attention heads per down/upsampling level,
                     e.g., num_heads=[4, 8, 16, 16] means increased attention heads with increasing depth.
                     *The length of num_heads must equal to `depth`.
                     
        window_size: the size of attention window per down/upsampling level,
                     e.g., window_size=[4, 2, 2, 2] means decreased window size with increasing depth.
                     
        num_mlp: number of MLP nodes.
        
        shift_window: The indicator of window shifting;
                      shift_window=True means applying Swin-MSA for every two Swin-Transformer blocks.
                      shift_window=False means MSA with fixed window locations for all blocks.

    Output
    ----------
        output tensor.
        
    Note: This function is experimental.
          The activation functions of all Swin-Transformers are fixed to GELU.
    
    '''
    print("--- TensorflowSwinUnet.swin_unet_2d_base ---")

    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = self.swin_transformer_stack(X, stack_num=stack_num_down, 
                               embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], window_size=window_size[0], num_mlp=num_mlp, 
                               shift_window=shift_window, name='{}_swin_down0'.format(name))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = self.swin_transformer_stack(X, stack_num=stack_num_down, 
                                   embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], window_size=window_size[i+1], num_mlp=num_mlp, 
                                   shift_window=shift_window, name='{}_swin_down{}'.format(name, i+1))
        
        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                            embed_dim=embed_dim, upsample_rate=2, return_vector=True, name='{}_swin_up{}'.format(name, i))(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = self.swin_transformer_stack(X, stack_num=stack_num_up, 
                           embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                           num_heads=num_heads[i], window_size=window_size[i], num_mlp=num_mlp, 
                           shift_window=shift_window, name='{}_swin_up{}'.format(name, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                        embed_dim=embed_dim, upsample_rate=patch_size[0], return_vector=False)(X)
    
    return X

  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
    print("==== TensorflowSwiUNet.create ")
    input_size = (image_width, image_height, image_channels)
    #IN = Input(input_size)
    inputs = Input((image_height, image_width, image_channels))
    IN = Lambda(lambda x: x / 255)(inputs)
    '''
    The base of SwinUNET.
    
    ----------
    Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021. 
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_begin: number of channels in the first downsampling block; 
                          it is also the number of embedded dimensions.
        n_labels: number of output labels.
        depth: the depth of Swin-UNET, e.g., depth=4 means three down/upsampling levels and a bottom level.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of Swin-Transformers) ----------
        
        patch_size: The size of extracted patches, 
                    e.g., patch_size=(2, 2) means 2-by-2 patches
                    *Height and width of the patch must be equal.
                    
        num_heads: number of attention heads per down/upsampling level,
                     e.g., num_heads=[4, 8, 16, 16] means increased attention heads with increasing depth.
                     *The length of num_heads must equal to `depth`.
                     
        window_size: the size of attention window per down/upsampling level,
                     e.g., window_size=[4, 2, 2, 2] means decreased window size with increasing depth.
                     
        num_mlp: number of MLP nodes.
        
        shift_window: The indicator of window shifting;
                      shift_window=True means applying Swin-MSA for every two Swin-Transformer blocks.
                      shift_window=False means MSA with fixed window locations for all blocks.
        
    Output
    ----------
        model: a keras model.
    
    Note: This function is experimental.
          The activation functions of all Swin-Transformers are fixed to GELU.
    '''
    IN = Input(input_size)
       
    # Base architecture
    X = self.swin_unet_2d_base(IN, self.filter_num_begin, self.depth, self.stack_num_down, self.stack_num_up, 
                      self.patch_size, self.num_heads, self.window_size, self.num_mlp, 
                      shift_window=self.shift_window, name='swin_unet')
    # output layer
    #OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    OUT = Conv2D(num_classes, kernel_size=1, use_bias=False, activation='sigmoid')(X)

    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='SwinNet')
    
    return model
    

if __name__ == "__main__":
  try:
    config_file = "./train_eval_inf.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    model = TensorflowSwinUNet(config_file)
    
  except:
    traceback.print_exc()
