### QTTNet: Quantized Tensor Train Neural Networks for 3D Object and Video Recognition

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
	* 'QTTNet/ModelNet40/data/__data_loader.py__': Prepare data(.h5) for training QTTNet & Augmentation
	* 'QTTNet/ModelNet40/data/__h5.py__': Convert to h5 files to tfrecord

- UCF

	* 'QTTNet/UCF/__PrepareData.py__': Read video data and prepare material for training QTTNet
	* 'QTTNet/UCF/__InputData.py__': Input data to QTTNet & Augmentation

## Experiment

ModelNet40
```
$ python Modelnet_main.py --data_dir '/data_dir/' --model_dir '/model_dir/'
```

UCF
```
$ python Action3DCNN.py flag_log_dir '/log_dir/'
```

## Extra

- '__Quantize.py__': Quantization code of Weight(W), Activation(A), Batch Normalization(BN)
- '__BatchNorm.py__': Code for quantized BN
- '__utils.py__': Functions for operations of Tensor Train cores
