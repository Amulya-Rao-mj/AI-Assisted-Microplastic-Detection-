
micro plastic detection - v20 2026-07-25 1:56pm
==============================

This dataset was exported via roboflow.com on July 25, 2026 at 7:05 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1600 images.
Micro-plastic-detection are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 320x320 (Fit (black edges))

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 25 percent of the image
* Random rotation of between -45 and +45 degrees
* Random shear of between -0° to +0° horizontally and -0° to +0° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -9 and +9 percent
* Random Gaussian blur of between 0 and 3.2 pixels


