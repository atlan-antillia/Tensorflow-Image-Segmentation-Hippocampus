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

# ImageMaskDataset.py
# 2023/05/31 to-arai Modified to use config_file
# 2023/10/02 Updated to call self.read_image_file, and self.read_mask_file in create nethod.

import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import traceback
from ConfigParser import ConfigParser
from BaseImageMaskDataset import BaseImageMaskDataset

TRAIN = "train"
DATASET = "dataset"

class ImageMaskDataset(BaseImageMaskDataset):

  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== ImageMaskDataset.constructor")

    self.resize_interpolation = eval(self.config.get(DATASET, "resize_interpolation", dvalue="cv2.INTER_NEAREST"))
    print("--- self.resize_interpolation {}".format(self.resize_interpolation))

  def read_image_file(self, image_file):
    image = cv2.imread(image_file) 
    
    image = cv2.resize(image, dsize= (self.image_height, self.image_width), 
                       interpolation=self.resize_interpolation)

    return image

  def read_mask_file(self, mask_file):
    mask = cv2.imread(mask_file) 
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = cv2.resize(mask, dsize= (self.image_height, self.image_width), 
                       interpolation=self.resize_interpolation)                       
 
    if self.binarize:
      mask[mask< self.threshold] =   0
      mask[mask>=self.threshold] = 255

    # Blur mask 
    if self.blur_mask:
      mask = cv2.blur(mask, self.blur_size)
  
    mask  = np.expand_dims(mask, axis=-1)
    return mask

if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = ImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

