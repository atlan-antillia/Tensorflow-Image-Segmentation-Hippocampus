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

# GrayImageWriter.py

# 2023/05/05 to-arai
# 2023/05/24 to-arai
# 2024/02/22 Added self.colorize

import os
import cv2
import numpy as np
import traceback

from PIL import Image, ImageOps

class GrayScaleImageWriter:

  def __init__(self, image_format=".jpg", colorize=False, black="black", white="green"):
    self.image_format = image_format
    self.colorize     = colorize
    self.black   = black
    self.white   = white

  def save(self, data, output_dir, name, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    #(h, w, c) = data.shape
    image = Image.new("L", (w, h))
    print(" image w: {} h: {}".format(w, h))
    for i in range(w):
      for j in range(h):
        z = data[j][i]
        if type(z) == list:
          z = z[0]
        v = int(z * factor)
        image.putpixel((i,j), v)

    image_filepath = os.path.join(output_dir, name + self.image_format)

    image.save(image_filepath)
    print("=== Saved {}". format(image_filepath))

    
  
  def save_resized(self, data, resized, output_dir, name, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    #(h, w, c) = data.shape
    image = Image.new("L", (w, h))
    print(" image w: {} h: {}".format(w, h))
    for i in range(w):
      for j in range(h):
        z = data[j][i]
        if type(z) == list:
          z = z[0]
        v = int(z * factor)
        image.putpixel((i,j), v)
    #print("{} {} {}".format(output_dir, name, self.image_format))
    image_filepath = os.path.join(output_dir, name + self.image_format)
 
    print("== resized to {}".format(resized))
    image = image.resize(resized)
    if self.colorize:
       image = ImageOps.colorize(image, black=self.black, white=self.white)
    # image.putalpha(alpha=0)
    image.save(image_filepath)
    image = image.convert("RGB")
    print("=== Saved {}". format(image_filepath))
    # 2023/0524
    #mask = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return np.array(image)
  
    
if __name__ == "__main__":
  try:
    writer = GrayScaleImageWriter()
    file   = "./sarah-antillia.png"
    img    = cv2.imread(file)
    img    = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    writer.save(img, "./", "sarah-antillia-gray", factor=1.0)
    writer.save_resized(img, (255, 255), "./", "sarah-antillia-gray-resized", factor=1.0)

  except:
    traceback.print_exc()

