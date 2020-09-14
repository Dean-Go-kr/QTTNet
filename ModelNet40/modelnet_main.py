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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import logging

import tensorflow as tf

import CNN_model
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/data/',
    help='The directory where the UCF11 input data is stored.')

parser.add_argument(
        '--model_dir', type=str, default='../model/',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--train_epochs', type=int, default=32,
    help='The number of epochs to use for training.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

parser.add_argument(
    '--model_structure', type=int, default=1,
    choices = [0, 1, 2],
    help = '0: load original CNN structure'
           '1: load QTTNet')

parser.add_argument(
    '--multi_gpu', action = 'store_true', help = 'If set, run across all available GPUs.')

_FRAME = 32  # Depth
_HEIGHT = 32
_WIDTH =32
NUM_CHANNELS = 1
_LABEL_CLASSES = 40

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 118116 ,
    'validation': 29616,
}

_FILE_SHUFFLE_BUFFER = 10000
_SHUFFLE_BUFFER = 10000

###################
# Data processing #
###################

def validate_batch_size_for_multi_gpu(batch_size):
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type =='GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs were found')

    remainder = FLAGS.batch_size % num_gpus
    if remainder:
        err = ('batch size, gpu error')
        raise ValueError(err)

def filenames(is_training, data_dir):
    if is_training:
        return [
                os. path.join(data_dir,'modelnet40_train_data_00%d.tfrecords' %(i+1))
                for i in range(9)]

    else:
        return [
                os.path.join(data_dir, 'modelnet40_test_data_00%d.tfrecords' %(i+1))
                for i in range(2)]


# Tfrecord file decoding
def record_parser(raw_record,is_training):
  
  # Dense features in Example proto.
  feature_map = {
      
      'height': tf.FixedLenFeature([], tf.int64, default_value=-1),
      'width': tf.FixedLenFeature([], tf.int64, default_value=-1),
      'depth': tf.FixedLenFeature([], tf.int64, default_value=-1),
      'label': tf.FixedLenFeature([], tf.int64, default_value=-1),
      'channel': tf.FixedLenFeature([], tf.int64, default_value=-1),
      'image_raw': tf.FixedLenFeature((), tf.string, default_value=''),
      
      
  }
 
  parsed = tf.parse_single_example(serialized=raw_record,
                                        features=feature_map)
  
  label = tf.cast(tf.reshape(parsed['label'], shape=[]), dtype = tf.int32)
  height = tf.cast(tf.reshape(parsed['height'], shape=[]), dtype = tf.int32)
  depth = tf.cast(tf.reshape(parsed['depth'], shape=[]), dtype = tf.int32)
  width = tf.cast(tf.reshape(parsed['width'], shape=[]), dtype = tf.int32)
  channels = tf.cast(tf.reshape(parsed['channel'], shape=[]), dtype = tf.int32)
  
  # Decode image
  image = tf.reshape(tf.decode_raw(parsed['image_raw'], out_type=tf.float32,little_endian=True), shape=[_FRAME,_HEIGHT,_WIDTH,NUM_CHANNELS])

  print('LABEL',label.shape)



  #------------------Data Preprocessing----------------------#
  # Training data random shuffle, not test data
  # If traning data set, after data augmentation of Height & Width,
  # Frame(Depth) random shuffle 

  
  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input function which provides batches for train or eval."""
  dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir))

  if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value, is_training),
                        num_parallel_calls=5)

  dataset = dataset.prefetch(batch_size)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels



def cnn_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  #tf.summary.image('images', features, max_outputs=6)
  
  if FLAGS.model_structure == 0:
      network = cnn_model.cnn_model(_LABEL_CLASSES, params['data_format'])
  else:
      network = cnn_model.t3f_quan_6layer_cnn_model(_LABEL_CLASSES, params['data_format'])
  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)
  tf.trainable_variables()
  print(tf.trainable_variables())
  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'BatchNorm' not in v.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    initial_learning_rate = 0.003
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 8, 16, and 24 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [8, 16, 24, 28]]
    values = [
        initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    if params.get('multi_gpu'):
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op=optimizer.minimize(loss,global_step)

  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  accuracy5 = tf.metrics.mean(tf.nn.in_top_k(logits,tf.argmax(labels, axis=1),k=5))
      
  metrics = {'accuracy': accuracy, 'accuracy5':accuracy5}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'

  model_function = cnn_model_fn
  # Set up a RunConfig to only save checkpoints once per training cycle.

  if FLAGS.multi_gpu:
      print('multi_gpu ON')
      validate_batch_size_for_multi_gpu(FLAGS.batch_size)

      model_function = tf.contrib.estimator.replicate_model_fn(
              cnn_model_fn, loss_reduction = tf.losses.Reduction.MEAN)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction=0.99
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False


  run_config = tf.estimator.RunConfig(session_config=config).replace(save_checkpoints_steps=1846)
  resnet_classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })

  eval_record=[]

  for train_epoch in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy',
    }

    print('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])

    print('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print(eval_results)
    eval_record.append(eval_results)
    np.save('../data/eval_results.npy',eval_record)  
  np.save('../data/eval_results_final.npy',eval_record)

if __name__ == '__main__':
  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)
  fh=logging.FileHandler('../log/')
  fh.setLevel(logging.DEBUG)
  log.addHandler(fh)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
