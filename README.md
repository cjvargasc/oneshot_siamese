# One-Shot Logo Recognition Based on Siamese Neural Networks

This code presents a Siamese Neural networks assessment for different embedded models using the QMUL-OpenLogo dataset, following the paper: One Shot Logo Recognition Based on Siamese Neural Networks.

## Requirements
* Python 3.x
* Numpy
* OpenCV
* Pytorch
* Matplotlib
* PIL
* QMUL-OpenLogo dataset (https://qmul-openlogo.github.io/)

## Usage
Use the misc/data_prep.py script to preprocess the QMUL-OpenLogo dataset (crop and data split) by defining the ```python openlogo_path```, ```python train_dir``` and ```python train_dir``` variables.
Set the ```python params/config.py``` file to define the architecture to train and training parameters. Run the ```python main.py``` file to train/test the defined configuration.

## Results

Embedded CNN | TPR | FPR | acc | Pr | F1 | AUC
-------------|-----|-----|-----|----|----|-----
AlexNet | 0.74 | 0.20 | 0.77 | 0.78 | 0.76 | 0.84
vgg | 0.74 | 0.22 | 0.75 | 0.76 | 0.75 | 0.82
Koch | 0.70 | 0.33 | 0.68 | 0.67 | 0.69 | 0.74
Resnet | 0.59 | 0.26 | 0.66 | 0.69 | 0.63 | 0.72
denseNet | 0.67 | 0.27 | 0.70 | 0.71 | 0.69 | 0.76

## Reference
To be published as part of the ACM International Conference on Multimedia Retrieval (ICMR) 2020

## Examples
(Showing the resulting dissimilarity metric for each pair of images)

<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/BMWimg000284.png" width="25%">-<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/BMWimg000284.png" width="25%">
d = 0.014

<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/BMWimg000284.png" width="25%">-<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/toyota14.png" width="25%">
d = 0.344

<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/BMWimg000284.png" width="25%">-<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/adidasimg000000.png" width="25%">
d = 0.837

<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/3m1.png" width="25%">-<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/3m8.png" width="25%">
d = 0.397

<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/3m1.png" width="25%">-<img src="https://github.com/cjvargasc/oneshot_siamese/blob/master/imgs/abusimg000000.png" width="25%">
d = 0.999
