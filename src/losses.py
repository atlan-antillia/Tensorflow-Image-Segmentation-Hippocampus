# Antillia.com Toshiyuki Arai
# 2023/05/12
# 2023/05/30 Modified return value of basnet_hybrid_loss.
#
# losses.py
#
# These functions have been taken from the following web site.
#
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
#   MIT license
"""
Citation
@inproceedings{jadon2020survey,
  title={A survey of loss functions for semantic segmentation},
  author={Jadon, Shruti},
  booktitle={2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
@article{JADON2021100078,
title = {SemSegLoss: A python package of loss functions for semantic segmentation},
journal = {Software Impacts},
volume = {9},
pages = {100078},
year = {2021},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2021.100078},
url = {https://www.sciencedirect.com/science/article/pii/S2665963821000269},
author = {Shruti Jadon},
keywords = {Deep Learning, Image segmentation, Medical imaging, Loss functions},
abstract = {Image Segmentation has been an active field of research as it has a wide range of applications, 
ranging from automated disease detection to self-driving cars. In recent years, various research papers 
proposed different loss functions used in case of biased data, sparse segmentation, and unbalanced dataset. 
In this paper, we introduce SemSegLoss, a python package consisting of some of the well-known loss functions 
widely used for image segmentation. It is developed with the intent to help researchers in the development 
of novel loss functions and perform an extensive set of experiments on model architectures for various 
applications. The ease-of-use and flexibility of the presented package have allowed reducing the development 
time and increased evaluation strategies of machine learning models for semantic segmentation. Furthermore, 
different applications that use image segmentation can use SemSegLoss because of the generality of its 
functions. This wide range of applications will lead to the development and growth of AI across all industries.
}
}

"""

"""
Orignal usage example taken from the website above is the follwing:
model.compile(optimizer=Adam(lr=1e-3),
              loss=semantic_loss.unet3p_hybrid_loss,
              metrics=[semantic_loss.dice_coef, semantic_loss.sensitivity, semantic_loss.specificity])

              
We use the following simplified version.
-->
model.compile(optimizer = Adam(lr=1e-3),
              loss      = basnet_hybrid_loss,
              metrics  = [dice_coef],
              or
              #metrics   = [dice_coef, sensitivity, specificity]
              )

"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
   
 
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss / 2.0

def jacard_similarity(y_true, y_pred):
    """
    Intersection-Over-Union (IoU), also known as the Jaccard Index
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # 2023/05/10 Added casting to 'float32'
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
    return intersection / union


def jacard_loss(y_true, y_pred):
    """
    Intersection-Over-Union (IoU), also known as the Jaccard loss
    """
    return 1 - jacard_similarity(y_true, y_pred)

def iou_coef(y_true, y_pred):
    return jacard_similarity(y_true, y_pred)

def iou_loss(y_true, y_pred):
    return 1 - jacard_similarity(y_true, y_pred)

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss
    """
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    return 1 - tf.image.ssim(y_true_f, y_pred_f, max_val=1)

def basnet_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
    The hybrid loss is a combination of the binary cross entropy, structural similarity
    and intersection-over-union losses, which guide the network to learn
    three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
    """
    bce_loss = BinaryCrossentropy(from_logits=False)
    bce_loss = bce_loss(y_true, y_pred)

    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jac_loss     = jacard_loss(y_true, y_pred)

    loss = bce_loss + ms_ssim_loss + jac_loss
    # 2023/05/30
    return loss/3.0

# 2023/06/07 
def bce_iou_loss(y_true, y_pred):
    bce_loss = BinaryCrossentropy(from_logits=False)
    loss1 = bce_loss(y_true, y_pred)
    loss2 = iou_loss(y_true, y_pred)

    loss = loss1 + loss2
    return loss/2.0
