For code usage, you have to be located in the same subfolder as this readme.md file.

For creation of the dataset:

python dataset/augment.py

For dataset partition into evaluation, test and train datasets:

python dataset/partition.py

For cutting board into cutouts:

First, you have to put the board image into the images subfolder. Then run the following code:

python code/cutout.py <image_name>

To classify the board:

python -m code.classify.measurePerformance <model> <image_name>

For determining metric performances:

python -m code.classify.measurePerformance <model> measure

Possible models:
kmeans-base
simple
ResNet50
ResNet50-ImageNet-1k
DINOv2-small
DINOv2-base
