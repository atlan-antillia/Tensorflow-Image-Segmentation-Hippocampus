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
# pip install scikit-image
from skimage.transform import resize
#from skimage.morphology import label

from skimage.io import imread
import traceback
from ConfigParser import ConfigParser

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"
TEST   = "test"
MASK   = "mask"
IMAGE  = "image"

class BaseImageMaskDataset:

  def __init__(self, config_file):
    print("=== BaseImageMaskDataset.constructor")

    self.config = ConfigParser(config_file)
    self.image_width    = self.config.get(MODEL, "image_width")
    self.image_height   = self.config.get(MODEL, "image_height")
    self.image_channels = self.config.get(MODEL, "image_channels")
    
    self.binarize  = self.config.get(MASK, "binarize")
    self.threshold = self.config.get(MASK, "threshold")
    self.blur_mask = self.config.get(MASK, "blur")
  
    self.blur_size = self.config.get(MASK, "blur_size", dvalue=(3,3))


  # If needed, please override this method in a subclass derived from this class.
  def create(self, dataset = TRAIN,  debug=False):
    if not dataset in [TRAIN, EVAL, TEST]:
      raise Exception("Invalid dataset")
    print("=== BaseImagMaskDataset.create dataset {}".format(dataset))
    image_datapath = self.config.get(dataset, "image_datapath")
    mask_datapath  = self.config.get(dataset, "mask_datapath")
    
    image_files  = glob.glob(image_datapath + "/*.jpg")
    image_files += glob.glob(image_datapath + "/*.png")
    image_files += glob.glob(image_datapath + "/*.bmp")
    image_files += glob.glob(image_datapath + "/*.tif")
    image_files  = sorted(image_files)

    mask_files   = None
    if os.path.exists(mask_datapath):
      mask_files  = glob.glob(mask_datapath + "/*.jpg")
      mask_files += glob.glob(mask_datapath + "/*.png")
      mask_files += glob.glob(mask_datapath + "/*.bmp")
      mask_files += glob.glob(mask_datapath + "/*.tif")
      mask_files  = sorted(mask_files)
      
      if len(image_files) != len(mask_files):
        raise Exception("FATAL: Images and masks unmatched")
      
    num_images  = len(image_files)
    if num_images == 0:
      raise Exception("FATAL: Not found image files")
    
    X = np.zeros((num_images, self.image_height, self.image_width, self.image_channels), dtype=np.uint8)

    #Y = np.zeros((num_images, self.image_height, self.image_width, 1                ), dtype=np.bool)
    Y = np.zeros((num_images, self.image_height, self.image_width, 1                ), dtype=bool)

    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
      X[n]  = self.read_image_file(image_file)
      Y[n]  = self.read_mask_file(mask_files[n])

      if debug:
          cv2.imshow("---", Y[n])
          #plt.show()
          cv2.waitKey(27)
          input("XX")   
  
    return X, Y

  def read_image_file(self, image_file):
    image = imread(image_file)
    image = resize(image, (self.image_height, self.image_width, self.image_channels), 
                     mode='constant', 
                     preserve_range=True)
    return image
  
  def read_mask_file(self, mask_file):
    mask = imread(mask_file)
    mask = resize(mask, (self.image_height, self.image_width, 1),  
                    mode='constant', 
                    preserve_range=False, 
                    anti_aliasing=False) 

    return mask

if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"

    dataset = BaseImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=TRAIN, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    x_test, y_test = dataset.create(dataset=EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

