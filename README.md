# temporal_efficient_training
 Code for temporal efficient training

## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

## Preprocess of DVS-CIFAR
 * Download CIFAR10-DVS dataset
 * transform .aedat to .mat by test_dvs.m with matlab.
 * prepare the train and test data set by dvscifar_dataloader.py [1](https://github.com/aa-samad/conv_snn)
 * you can obtain processed data in this [link](https://drive.google.com/file/d/1s2csG5eagX3ZMfFpZCd5d7g8zqJxht4U/view?usp=drive_link).

## Description
 * use a triangle-like surrogate gradient `ZIF` in `models/layer.py` for step function forward and backward.
 * It's very easy to build snn convolution layer by `Layer` in `models/layer.py`. \
   `self.conv = nn.Sequential(Layer(2,64,3,1,1),Layer(64,128,3,1,1),)`
 * The 0-th and 1-th dimension of snn layer's input and output are batch-dimension and time-dimension. 
 

## Citation
Reference [paper](https://openreview.net/forum?id=_XNtisL32jv).
```
@inproceedings{
deng2022temporal,
title={Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting},
author={Shikuang Deng and Yuhang Li and Shanghang Zhang and Shi Gu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=_XNtisL32jv}
}
```
