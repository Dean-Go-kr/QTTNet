## QTTNet: Quantized Tensor Train Neural Networks for 3D Object and Video Recognition

This is code for implementation of QTTNet

## Requirements

- ModelNet: CUDA 9.0, Python 3.6, Tensorflow 1.8.0, Numpy, CV2
- UCF: CUDA 9.0, Python 3.7, Tensorflow 1.14.0, Numpy, CV2, FFMPEG

## Data & Preparation

Download the ModelNet and UCF on official link

- ModelNet: https://modelnet.cs.princeton.edu/
- UCF: https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php

To prepare dataset for training QTTNet, you should run py files by order.

- ModelNet

	* 'QTTNet/ModelNet40/data/__convert_modelnet40.py__': Convert npy.tar from official voxelized data(.mat)
	* 'QTTNet/ModelNet40/data/data_loader.py': Prepare data(.h5) for training QTTNet & Augmentation
	* 'QTTNet/ModelNet40/data/h5.py': Convert to h5 files to tfrecord

- UCF

	* /UCF/PrepareData.py: Read video data and prepare material for training QTTNet

## Experiment


Train small i-RevNet on Cifar-10, takes about 5 hours and yields an accuracy of ~94.5%
```
$ python CIFAR_main.py --nBlocks 18 18 18 --nStrides 1 2 2 --nChannels 16 64 256
```
