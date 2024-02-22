<h2>Tensorflow-Image-Segmentation-Hippocampus (Updated: 2024/02/22)</h2>

This is an experimental Image Segmentation project for Hippocampus based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
<br>
<li>2024/02/20: Modified to use the latest dataset 
<a href="https://drive.google.com/file/d/1FAgeAlwvzCscZVvAovqpsTQdum90_7y-/view?usp=sharing">
Hippocampus-ImageMask-Dataset.zip</a> (Updated:2024/02/21).
</li>

<li>2024/02/21: Modified to use the latest dataset (rotated 90 degrees counterclockwise)
<a href="https://drive.google.com/file/d/1FAgeAlwvzCscZVvAovqpsTQdum90_7y-/view?usp=sharing">
Hippocampus-ImageMask-Dataset.zip</a> (Updated:2024/02/21).
</li>

<li>2024/02/22: Added [segmentation] section to 
<a href="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/train_eval_infer.config">train_eval_infer.config</a> 
to be able to colorize the inferred mask regions.
</li>
<li>2024/02/22: Updated <a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> to import bce_dice_loss.
</li>
<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/sample_images.png" width="720" height="auto">
<br>
<br>
As a first trial, we use the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Hippocampus Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<br>


<h3>1. Dataset Citation</h3>

The original image dataset used here has been taken from the following kaggle web site.<br>
<a href="https://www.kaggle.com/datasets/andrewmvd/hippocampus-segmentation-in-mri-images">
Hippocampus Segmentation in MRI Images</a><br>
<br>
<b>About Dataset</b>
<pre>
Introduction
The hippocampus is a structure within the brain that plays important roles in the 
consolidation of information from short-term memory to long-term memory, and in spatial 
memory that enables navigation. 
Magnetic resonance imaging is often the optimal modality for brain medical imaging studies, 
being T1 ideal for representing structure.
The hippocampus has become the focus of research in several neurodegenerative disorders. 
Automatic segmentation of this structure from magnetic resonance (MR) imaging scans of the 
brain facilitates this work, especially in resource poor environments.
</pre>
<b>About This Dataset</b>
<pre>
This dataset contains T1-weighted MR images of 50 subjects, 40 of whom are patients with 
temporal lobe epilepsy and 10 are nonepileptic subjects. Hippocampus labels are provided 
for 25 subjects for training. For more information about the dataset, refer to the 
original article.

How To Cite this Dataset
Original Article
K. Jafari-Khouzani, K. Elisevich, S. Patel, and H. Soltanian-Zadeh, 
“Dataset of magnetic resonance images of nonepileptic subjects and temporal lobe epilepsy 
patients for validation of hippocampal segmentation techniques,” 
Neuroinformatics, 2011.

License
The dataset is free to use for research and education. 
Please refer to the original article if you use it in your publications.

Dataset BibTeX
@article{,
title= {MRI Dataset for Hippocampus Segmentation (HFH) (hippseg_2011)},
keywords= {},
author= {K. Jafari-Khouzani and K. Elisevich, S. Patel and H. Soltanian-Zadeh},
abstract= {This dataset contains T1-weighted MR images of 50 subjects, 40 of whom are patients
with temporal lobe epilepsy and 10 are nonepileptic subjects. Hippocampus labels are provided 
for 25 subjects for training. The users may submit their segmentation outcomes for the 
remaining 25 testing images to get a table of segmentation metrics.},
terms= {The dataset is free to use for research and education. Please refer to the following 
article if you use it in your publications:
K. Jafari-Khouzani, K. Elisevich, S. Patel, and H. Soltanian-Zadeh, 
“Dataset of magnetic resonance images of nonepileptic subjects and temporal lobe epilepsy 
patients for validation of hippocampal segmentation techniques,” Neuroinformatics, 2011.},
license= {free to use for research and education},
superseded= {},
url= {https://www.nitrc.org/projects/hippseg_2011/}
}
</pre>

<h3>
<a id="2">
2 Hippocampus ImageMask Dataset
</a>
</h3>
 If you would like to train this Hippocampus Segmentation model by yourself,
 please download the latest dataset from the google drive 
<a href="https://drive.google.com/file/d/1FAgeAlwvzCscZVvAovqpsTQdum90_7y-/view?usp=sharing">
Hippocampus-ImageMask-Dataset.zip.</a> (Updated:2024/02/20)
<br>
Please see also the <a href="https://github.com/atlan-antillia/Hippocampus-Image-Dataset">Hippocampus-Image-Dataset</a>.<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be

<pre>
./dataset
└─Hippocampus
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
 
 
<b>Hippocampus Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/Hippocampus_Statistics.png" width="512" height="auto"><br>

<br>

<h3>
<a id="3">
3 TensorflowSlightlyFlexibleUNet
</a>
</h3>
This <a href="./src/TensorflowUNet.py">TensorflowUNet</a> model is slightly flexibly customizable by a configuration file.<br>
For example, <b>TensorflowSlightlyFlexibleUNet/Hippocampus</b> model can be customizable
by using <a href="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/train_eval_infer.config">
train_eval_infer.config.</a>
<pre>
; train_eval_infer.config
; 2024/02/22 (C) antillia.com

[model]
model          = "TensorflowUNet"

image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 1
base_filters   = 16
num_layers     = 7
dropout_rate   = 0.08
learning_rate  = 0.0001
loss           = "bce_iou_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/Hippocampus/train/images/"
mask_datapath  = "../../../dataset/Hippocampus/train/masks/"

[eval]
image_datapath = "../../../dataset/Hippocampus/valid/images/"
mask_datapath  = "../../../dataset/Hippocampus/valid/masks/"

[infer] 
images_dir    = "../../../dataset/Hippocampus/test/images/"
output_dir    = "./test_output"
merged_dir    = "./test_output_merged"

[segmentation]
colorize      = True
black         = "black"
white         = "green"
blursize      = (5,5)

[mask]
blur      = False
binarize  = True
threshold = 128
</pre>
As shown above, we have been using <b>bce_iou_loss</b> function.<br>
<pre>
[model]
loss           = "bce_iou_loss"
</pre>
Of course, you may use <b>bce_dice_loss</b> function.
<pre>
[model]
loss           = "bce_dice_loss"
</pre>
<br>

<h3>
3.1 Training
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Hippocampus</b> folder,<br>
and run the following bat file to train TensorflowUNet model for Hippocampus.<br>
<pre>
./1.train.bat
</pre>
<pre>
python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config

</pre>
Train console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/train_console_output_at_epoch_41.png" width="720" height="auto"><br>
Train metrics:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/train_metrics_at_epoch_41.png" width="720" height="auto"><br>
Train losses:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/train_losses_at_epoch_41.png" width="720" height="auto"><br>
 
<h3>
3.2 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Hippocampus</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Hippocampus.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/evaluate_console_output_at_epoch_41.png" width="720" height="auto"><br>

<h2>
3.3 Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Hippocampus</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Hippocampus.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer.config
</pre>
Sample test images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/sample_test_images.png" width="1024" height="auto"><br>
Sample test mask<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/sample_test_masks.png" width="1024" height="auto"><br>

<br>
Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/inferred_test_mask_green.png" width="1024" height="auto"><br>
<br>
Merged test images and inferred masks<br> 
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/asset/merged_test_output_green.png" width="1024" height="auto"><br> 

Enlarged samples<br>
<table>
<tr>
<td>
test/images/10040_HFH_017.jpg<br>
<img src="./dataset/Hippocampus/test/images/10040_HFH_017.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/10040_HFH_017.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/test_output_merged/10040_HFH_017.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
test/images/10048_HFH_004.jpg<br>
<img src="./dataset/Hippocampus/test/images/10048_HFH_004.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/10048_HFH_004.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/test_output_merged/10048_HFH_004.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/10049_HFH_012.jpg<br>
<img src="./dataset/Hippocampus/test/images/10049_HFH_012.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/10049_HFH_012.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/test_output_merged/10049_HFH_012.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/10053_HFH_018.jpg<br>
<img src="./dataset/Hippocampus/test/images/10053_HFH_018.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/10053_HFH_018.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/test_output_merged/10053_HFH_018.jpg" width="512" height="auto">
</td> 
</tr>

<!-- 5-->
<tr>
<td>
test/images/10064_HFH_010.jpg<br>
<img src="./dataset/Hippocampus/test/images/10064_HFH_010.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/10064_HFH_010.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hippocampus/test_output_merged/10064_HFH_010.jpg" width="512" height="auto">
</td> 
</tr>

</table>

<h3>
References
</h3>
<b>1. Hippocampus Segmentation Method Applying Coordinate Attention Mechanism and Dynamic Convolution Network</b><br>
Juan Jiang, Hong Liu 1ORCID,Xin Yu,Jin Zhan, ORCID,Bing Xiong andLidan Kuang<br>
Appl. Sci. 2023, 13(13), 7921; https://doi.org/10.3390/app13137921<br>
<pre>
https://www.mdpi.com/2076-3417/13/13/7921
</pre>

<b>2. Hippocampus Segmentation Using U-Net Convolutional Network from Brain Magnetic Resonance Imaging (MRI)</b><br>
Ruhul Amin Hazarika, Arnab Kumar Maji, Raplang Syiem, Samarendra Nath Sur, Debdatta Kandar<br>
PMID: 35304675 PMCID: PMC9485390 DOI: 10.1007/s10278-022-00613-y<br>
<pre>
https://pubmed.ncbi.nlm.nih.gov/35304675/
</pre>

<b>3. Hippocampus-Image-Dataset </b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/atlan-antillia/Hippocampus-Image-Dataset
</pre>
