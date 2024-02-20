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

# ImageMaskAugmentor.py
# 2023/08/20 to-arai
# 2023/08/25 Fixed bugs on some wrong section name settings.
# 2023/08/26 Added shrink method to augment images and masks.
# 2023/08/27 Added shear method to augment images and masks.
# 2023/08/28 Added elastic_transorm method to augment images and masks.
# 2024/02/12 Modified shear method to check self.hflip and self.vflip flags

import os
import sys
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from ConfigParser import ConfigParser

MODEL     = "model"
GENERATOR = "generator"
AUGMENTOR = "augmentor"

class ImageMaskAugmentor:
  
  def __init__(self, config_file):
    self.config  = ConfigParser(config_file)
    self.debug    = self.config.get(GENERATOR, "debug",  dvalue=True)
    self.W        = self.config.get(MODEL,     "image_width")
    self.H        = self.config.get(MODEL,     "image_height")

    self.rotation = self.config.get(AUGMENTOR, "rotation", dvalue=True)
    self.SHRINKS  = self.config.get(AUGMENTOR, "shrinks",  dvalue=[0.8])
    self.ANGLES   = self.config.get(AUGMENTOR, "angles",   dvalue=[60, 120, 180, 240, 300])
    #2023/08/27
    self.SHEARS   = self.config.get(AUGMENTOR, "shears",   dvalue=[])

    self.hflip    = self.config.get(AUGMENTOR, "hflip", dvalue=True)
    self.vflip    = self.config.get(AUGMENTOR, "vflip", dvalue=True)
    #print("---self.hflip {}".format(self.hflip))
    #input("----")
    self.transformer = self.config.get(AUGMENTOR, "transformer", dvalue=False)
    self.alpha    = self.config.get(AUGMENTOR, "alpah", dvalue=1300)
    self.sigmoid  = self.config.get(AUGMENTOR, "sigmoid", dvalue=8)
    self.seed     = 137
  
  # It applies  horizotanl and vertical flipping operations to image and mask repectively.
  def augment(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    """ 
    IMAGES: Python list
    MASKS:  Python list
    image:  OpenCV image
    mask:   OpenCV mask
    """
    if self.hflip:
      hflip_image = self.horizontal_flip(image) 
      hflip_mask  = self.horizontal_flip(mask) 
      #print("--- hflp_mask shape {}".format(hflip_mask.shape))
      IMAGES.append(hflip_image )
  
      MASKS.append( hflip_mask  )
      if self.debug:
        filepath = os.path.join(generated_images_dir, "hfliped_" + image_basename)
        cv2.imwrite(filepath, hflip_image)
        filepath = os.path.join(generated_masks_dir,  "hfliped_" + mask_basename)
        cv2.imwrite(filepath, hflip_mask)

    if self.vflip:
      vflip_image = self.vertical_flip(image)
      vflip_mask  = self.vertical_flip(mask)
      #print("== vflip shape {}".format(vflip_mask.shape))
      
      IMAGES.append(vflip_image )    
      MASKS.append( vflip_mask  )
      if self.debug:
        filepath = os.path.join(generated_images_dir, "vfliped_" + image_basename)
        cv2.imwrite(filepath, vflip_image)
        filepath = os.path.join(generated_masks_dir,  "vfliped_" + mask_basename)
        cv2.imwrite(filepath, vflip_mask)

    if self.rotation:
       self.rotate(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
       
    if type(self.SHRINKS) is list and len(self.SHRINKS)>0:
       self.shrink(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

    if type(self.SHEARS) is list and len(self.SHEARS)>0:
       self.shear(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )
    # 2023/08/28
    if self.transformer:
      self.elastic_transform(IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename )

  def horizontal_flip(self, image): 
    image = image[:, ::-1, :]
    return image

  def vertical_flip(self, image):
    image = image[::-1, :, :]
    return image
  
  
  def rotate(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    for angle in self.ANGLES:      

      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H))
      rotated_mask  = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=(self.W, self.H))
      IMAGES.append(rotated_image)
      rotated_mask  = np.expand_dims(rotated_mask, axis=-1) 
      #print("rotated_mask shape {}".format(rotated_mask.shape))
      MASKS.append(rotated_mask)

      if self.debug:
        filepath = os.path.join(generated_images_dir, "rotated_" + str(angle) + "_" + image_basename)
        cv2.imwrite(filepath, rotated_image)
        filepath = os.path.join(generated_masks_dir,  "rotated_" + str(angle) + "_" + mask_basename)
        cv2.imwrite(filepath, rotated_mask)
  

  def shrink(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    # 2023/08/26
    # Added the following shrinking augmentation.
    h, w = image.shape[:2]
  
    for shrink in self.SHRINKS:
      rw = int (w * shrink)
      rh = int (h * shrink)
      resized_image = cv2.resize(image, dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      resized_mask  = cv2.resize(mask,  dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      
      squared_image = self.paste(resized_image, mask=False)
      squared_mask  = self.paste(resized_mask,  mask=True)
      IMAGES.append(squared_image)
      MASKS.append(squared_mask)
      if self.debug:
        ratio   = str(shrink).replace(".", "_")
        image_filename = "shrinked_" + ratio + "_" + image_basename
        image_filepath  = os.path.join(generated_images_dir, image_filename)
        cv2.imwrite(image_filepath, squared_image)
        #print("=== Saved {}".format(image_filepath))
    
        mask_filename = "shrinked_" + ratio + "_" + mask_basename
        mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
        cv2.imwrite(mask_filepath, squared_mask)
        #print("=== Saved {}".format(mask_filepath))


  def paste(self, image, mask=False):
    l = len(image.shape)
   
    h, w,  = image.shape[:2]

    if l==3:
      background = np.zeros((self.H, self.W, 3), dtype=np.uint8)
      (b, g, r) = image[h-10][w-10] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      background += [b, g, r][::-1]
    else:
      v =  image[h-10][w-10] 
      #print("x {}".format(v))
      image  = np.expand_dims(image, axis=-1) 
      background = np.zeros((self.H, self.W, 1), dtype=np.uint8)
      background[background !=v] = v
    x = (self.W - w)//2
    y = (self.H - h)//2
    background[y:y+h, x:x+w] = image

    return background
  

  # Shear image and mask
  # 2023/08/27 Added shear method to augment images and masks.
  # This method has been taken from the following code in stackoverflow.
  # https://stackoverflow.com/questions/57881430/how-could-i-implement-a-centered-shear-an-image-with-opencv
  
  # 2024/02/12 Modified to check self.hflip and self.vflip flags

  def shear(self, IMAGES, MASKS, image, mask,
                 generated_images_dir, image_basename,
                 generated_masks_dir,  mask_basename ):

    if self.SHEARS == None or len(self.SHEARS) == 0:
      return
   
    H, W = image.shape[:2]
    for shear in self.SHEARS:
      ratio = str(shear).replace(".", "_")
      M2 = np.float32([[1, 0, 0], [shear, 1,0]])
      M2[0,2] = -M2[0,1] * H/2 
      M2[1,2] = -M2[1,0] * W/2 

      sheared_image = cv2.warpAffine(image, M2, (W, H))
      sheared_mask  = cv2.warpAffine(mask,  M2, (W, H))

      IMAGES.append(sheared_image)
      #print(" shape {}".format(sheared_mask.shape))
      sheared_mask  = np.expand_dims(sheared_mask, axis=-1) 

      MASKS.append(sheared_mask)

      # 2024/02/12
      if self.hflip:
        hflipped_image  = self.horizontal_flip(sheared_image)
        hflipped_mask   = self.horizontal_flip(sheared_mask)

        IMAGES.append(hflipped_image)
        MASKS.append(hflipped_mask)
      if self.vflip:
        vflipped_image  = self.vertical_flip(sheared_image)
        vflipped_mask   = self.vertical_flip(sheared_mask)

        IMAGES.append(vflipped_image)
        MASKS.append(vflipped_mask)

        hvflipped_image = self.vertical_flip(hflipped_image)
        hvflipped_mask  = self.vertical_flip(hflipped_mask)

        IMAGES.append(hvflipped_image)
        MASKS.append(hvflipped_mask)

      if self.debug:
        filepath = os.path.join(generated_images_dir, "sheared_" + ratio + "_" + image_basename)
        cv2.imwrite(filepath, sheared_image)
        #print("Saved {}".format(filepath))
        filepath = os.path.join(generated_masks_dir,  "sheared_" + ratio + "_" + mask_basename)
        cv2.imwrite(filepath, sheared_mask)
        #print("Saved {}".format(filepath))
        # 2024/02/12
        if self.hflip:
          filepath = os.path.join(generated_images_dir, "hflipped_sheared_" + ratio + "_" + image_basename)
          cv2.imwrite(filepath, hflipped_image)
          #print("Saved {}".format(filepath))
          filepath = os.path.join(generated_masks_dir,  "hflipped_sheared_" + ratio + "_" + mask_basename)
          cv2.imwrite(filepath, hflipped_mask)
          #print("Saved {}".format(filepath))
        if self.vflip:
          filepath = os.path.join(generated_images_dir, "vflipped_sheared_" + ratio + "_" + image_basename)
          cv2.imwrite(filepath, vflipped_image)
          #print("Saved {}".format(filepath))
          filepath = os.path.join(generated_masks_dir,  "vflipped_sheared_" + ratio + "_" + mask_basename)
          cv2.imwrite(filepath, vflipped_mask)
          #print("Saved {}".format(filepath))

          filepath = os.path.join(generated_images_dir, "hvflipped_sheared_" + ratio + "_" + image_basename)
          cv2.imwrite(filepath, hvflipped_image)
          #print("Saved {}".format(filepath))
          filepath = os.path.join(generated_masks_dir,  "hvflipped_sheared_" + ratio + "_" + mask_basename)
          cv2.imwrite(filepath, hvflipped_mask)
          #print("Saved {}".format(filepath))

    
  # This method has been taken from the following code.
  # https://github.com/MareArts/Elastic_Effect/blob/master/Elastic.py
  #
  # https://cognitivemedium.com/assets/rmnist/Simard.pdf
  #
  # See also
  # https://www.kaggle.com/code/jiqiujia/elastic-transform-for-data-augmentation/notebook

  def elastic_transform(self, IMAGES, MASKS, image, mask,
                generated_images_dir, image_basename,
                generated_masks_dir,  mask_basename ):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
    deformed_image = deformed_image.reshape(image.shape)

    shape = mask.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    deformed_mask = map_coordinates(mask, indices, order=1, mode='nearest')  
    deformed_mask = deformed_mask.reshape(mask.shape)

    IMAGES.append(deformed_image)
    MASKS.append(deformed_mask)

    if self.debug:
      image_filename = "elastic" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(self.sigmoid) + "_" + image_basename
      image_filepath  = os.path.join(generated_images_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)
      #print("=== Saved {}".format(image_filepath))
    
      mask_filename = "elastic" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(self.sigmoid) + "_" + mask_basename
      mask_filepath  = os.path.join(generated_masks_dir, mask_filename)
      cv2.imwrite(mask_filepath, deformed_mask)


