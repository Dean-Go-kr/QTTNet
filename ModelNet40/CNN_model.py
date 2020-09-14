# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import Layers_quan as Layers
from Quantize import fw,fa,fBits
from BatchNorm import BatchNorm, BatchNorm3d

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_HEIGHT = 32
_WIDTH = 32
_FRAME = 32
_NUM_CHANNELS = 1
_LABEL_CLASSES = 40

def batch_norm_relu3d(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  
  inputs = BatchNorm3d(inputs,center=True, scale=True, is_training=is_training, decay=0.997, Random=None, data_format='NDHWC')
  inputs = tf.nn.relu(inputs)
  inputs = fa(inputs)
  print('activ after:', inputs)

  return inputs


def conv3d(inputs, filters, kernel_size, strides, data_format,name):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  c_in = inputs.get_shape().as_list()[-1]

  def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
      fan_in = fan_out = 1
    elif len(shape) == 1:
      fan_in = fan_out = shape[0]
    elif len(shape) == 2:
      fan_in = shape[0]
      fan_out = shape[1]
    else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
      receptive_field_size = 1.
      for dim in shape[:-2]:
        receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out

  shape=[kernel_size, kernel_size, kernel_size, c_in, filters]
  fan_in,fan_out=_compute_fans(shape)
  
  with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      W=tf.Variable(fBits(tf.truncated_normal(shape=shape,stddev=1.0/tf.sqrt(float(fan_in))),16),name='conv3d/kernel')
      
  w_q = fw(W)
  if strides == 1:
     padding='SAME'
  else:
     padding='VALID'
  inputs = tf.nn.conv3d(inputs, w_q, strides=[1, strides,strides, strides, 1], padding=padding, data_format='NDHWC', name=name)
  
  inputs = fe2(inputs)
  return inputs


def conv3d_first(inputs, filters, kernel_size, strides, data_format):
  """Strided 3-D convolution"""
  
  inputs=tf.layers.conv3d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,kernel_constraint=None,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

  return inputs 


# Original CNN
def cnn_model(num_classes, data_format=None):

  if data_format is None:
    data_format = 'channels_last'

  def model(inputs, is_training):
    """Constructs the CNN model given the inputs."""
    print('input shape?:, ', inputs.shape)
    inputs = tf.reshape(inputs, (-1,_FRAME, _HEIGHT, _WIDTH,_NUM_CHANNELS))

    print('Input Shape:, ', inputs.shape)
    inputs = conv3d_first(
        inputs=inputs, filters=64, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv1_1')

    inputs = conv3d(
        inputs=inputs, filters=64, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_1_2')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv1_2')
   
  
    inputs = tf.layers.average_pooling3d(
        inputs=inputs, pool_size=2, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'avg_pool_1')
    print('First block output:', inputs.shape)

    #-------------------------------------------------------------------------#

    inputs = conv3d(
        inputs=inputs, filters=128, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_2_1')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv2_1')

    inputs = conv3d(
        inputs=inputs, filters=128, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_2_2')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.identity(inputs, 'conv2_2')

    inputs = conv3d(
        inputs=inputs, filters=128, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_2_3')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.identity(inputs, 'conv2_3')

    inputs = tf.layers.average_pooling3d(
        inputs=inputs, pool_size=2, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'avg_pool_2')
    print('Second block output:', inputs.shape)

#-------------------------------------------------------------------------#

    inputs = conv3d(
        inputs=inputs, filters=256, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_3_1')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.identity(inputs, 'conv3_1')
      
    inputs = conv3d(
        inputs=inputs, filters=256, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_3_2')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.identity(inputs, 'conv3_2')
    
    inputs = conv3d(
        inputs=inputs, filters=256, kernel_size=3, strides=1,
        data_format=data_format, name = 'conv3d_3_3')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.identity(inputs, 'conv3_3')
    

    inputs = tf.layers.average_pooling3d(
        inputs=inputs, pool_size=2, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'avg_pool_3')

    #-------------------------------------------------------------------------#

    sz = np.prod(inputs.get_shape().as_list()[1:])
    inputs = tf.reshape(inputs, [-1,sz])

    inputs = dense_relu(inputs=inputs, channels=4096, name= 'dense-relu_1')
    inputs = tf.nn.dropout(inputs, 0.5)
    inputs = tf.identity(inputs, 'dense_1')

    inputs = dense_relu(inputs=inputs, channels=2048, name= 'dense-relu_2')
    inputs = tf.nn.dropout(inputs, 0.5)
    inputs = tf.identity(inputs, 'dense_2')

    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model

# QTTNet
def t3f_quan_6layer_cnn_model(num_classes, data_format=None):

  if data_format is None:
    data_format = 'channels_last'

  def model(inputs, is_training):
    """Constructs the CNN model given the inputs."""

    inputs = tf.reshape(inputs, (-1,_FRAME, _HEIGHT, _WIDTH,_NUM_CHANNELS))

    print('Input Shape:, ', inputs.shape)
    inputs = conv3d_first(
        inputs=inputs, filters=64, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv1_1')
       
    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 64,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[4, 4, 4],  # 64 
                            output_ch_modes=[4, 4, 4],  # 64 
                            tt_ranks = [16, 16, 16],
                            name_scope = 'tt_conv3d_1_2')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv1_2')

     
    inputs = tf.nn.avg_pool3d(
                inputs, [1,3,3,3,1], [1,3,3,3,1], 'SAME')
    inputs = tf.identity(inputs, 'avg_pool_1')
    print('First block output:', inputs.shape)

    #-------------------------------------------------------------------------#
 
    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 128,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[2, 4, 2, 4],  # 64 
                            output_ch_modes=[4, 2, 4, 4],  # 128 
                            tt_ranks = [16, 16, 16, 16],
                            name_scope = 'tt_conv3d_2_1')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv2_1')
    
    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 128,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[4, 2, 4, 4],  # 128
                            output_ch_modes=[4, 4, 2, 4],  # 128 
                            tt_ranks = [16, 16, 16, 16],
                            name_scope = 'tt_conv3d_2_2')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv2_2')
    
    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 128,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[4, 2, 4, 4],  # 128
                            output_ch_modes=[4, 4, 2, 4],  # 128 
                            tt_ranks = [16, 16, 16, 16],
                            name_scope = 'tt_conv3d_2_3')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv2_3')
    
    
    inputs = tf.nn.avg_pool3d(
                inputs, [1,3,3,3,1], [1,3,3,3,1], 'SAME')
    inputs = tf.identity(inputs, 'avg_pool_3')
    print('Second block output:', inputs.shape)

#-------------------------------------------------------------------------#

    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 256,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[4, 2, 4, 4],  # 128
                            output_ch_modes=[4, 4, 4, 4],  # 256
                            tt_ranks = [16, 16, 16, 16],
                            name_scope = 'tt_conv3d_3_1')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv3_1')
    
    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 256,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[4, 4, 4, 4],  # 256
                            output_ch_modes=[4, 4, 4, 4],  # 256
                            tt_ranks = [16, 16, 16, 16],
                            name_scope = 'tt_conv3d_3_2')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv3_2')
    
    inputs = Layers.conv_3d_tt(inputs,
                            output_chs = 256,
                            filter_shape=[3,3,3], 
                            input_ch_modes=[4, 4, 4, 4],  # 256
                            output_ch_modes=[4, 4, 4, 4],  # 256
                            tt_ranks = [16, 16, 16, 16],
                            name_scope = 'tt_conv3d_3_3')
    inputs = batch_norm_relu3d(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'conv3_3')
  
    inputs = tf.nn.avg_pool3d(
        inputs, [1,3,3,3,1], [1,4,4,4,1], 'SAME')
    inputs = tf.identity(inputs, 'avg_pool_3')

    #-------------------------------------------------------------------------#

    sz = np.prod(inputs.get_shape().as_list()[1:])

    inputs = tf.reshape(inputs, [-1,sz])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model
