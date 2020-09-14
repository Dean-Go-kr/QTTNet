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
from Quantize import fg,flr,fgBN,fBits
#from Quantize import layer_output

#from tensorpack.tfutils.varreplace import remap_variables

import tensorflow as tf
import numpy as np

import resnet_model
import inception_preprocessing


g_scale=128

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/data/yangyk/dataset/imagenet_dataset/TFRecord256/',
    help='The directory where the ImageNet input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default='../model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size', type=int, default=18, choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument(
    '--train_epochs', type=int, default=100,
    help='The number of epochs to use for training.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=128,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

#_MOMENTUM = 0.9
#_MOMENTUM = 115.0/128  
#_MOMENTUM = 7./8
_MOMENTUM = 1./4
#_MOMENTUM = 1.0
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500


def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(1024)]
        #for i in range(2)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def record_parser(value, is_training):
  """Parse an ImageNet record from `value`."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = inception_preprocessing.preprocess_image(
      image=image,
      height=_DEFAULT_IMAGE_SIZE,
      width=_DEFAULT_IMAGE_SIZE,
      is_training=False)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input function which provides batches for train or eval."""
  dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir))

#  if is_training:
#    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value, is_training),
                        num_parallel_calls=5)
  dataset = dataset.prefetch(batch_size)

#  if is_training:
#    # When choosing shuffle buffer sizes, larger sizes result in better
#    # randomness, while smaller sizes have better performance.
#    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  tf.summary.image('images', features, max_outputs=6)
  
  network = resnet_model.imagenet_resnet_v2(
      params['resnet_size'], _LABEL_CLASSES, params['data_format'])
  
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
  
  #cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
  
  #cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      #logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'BatchNorm' not in v.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 256, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 256
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        #int(batches_per_epoch * epoch) for epoch in [20, 30, 40, 50]]
    values = [
        initial_learning_rate * decay for decay in [1, 0.12, 0.06, 0.03, 0.03]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    
    g_values = [128.,128.,32.,8.,2.]
    g_scale = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, g_values)
    tf.identity(g_scale, name='g_scale')
    
    
    learning_rate=flr(learning_rate)
    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gradTrainBatch = optimizer.compute_gradients(loss)
    
    
    
    grad=[]
    var=[]
    for grad_and_vars in gradTrainBatch:
        grad.append(grad_and_vars[0])
        var.append(grad_and_vars[1])
        
    
    def QuantizeG(gradTrainBatch):
        grads = []
        for grad_and_vars in gradTrainBatch:
            if grad_and_vars[1].name == 'conv2d/kernel:0' or   grad_and_vars[1].name.find('dense')>-1:
                #print('no quantize')
                #print(grad_and_vars[1].name)
                #print('********************************************************************')
                grads.append([grad_and_vars[0]*1.0 , grad_and_vars[1] ])            
            elif   grad_and_vars[1].name.find('BatchNorm')>-1:
                #print('BN Quantize')
                #print(grad_and_vars[1].name)
                grads.append([fgBN(grad_and_vars[0],1.0) , grad_and_vars[1] ])
                
                #SSprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                
            else:
                #if grad_and_vars[1].name == 'block_layer2/block1/conv2/conv2d/kernel:0':
                    #print('111111111')
                grads.append([fg(grad_and_vars[0],1.0,g_scale) , grad_and_vars[1] ])
                #print('Weights Quantize')
                #print(grad_and_vars[0].name)
                #print(grad_and_vars[1].name)
                #print('#####################################################################')
                
                #print(grad_and_vars[0])
                #print(grad_and_vars[1])
                #print('<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>')
           #print(grad_and_vars[0])
           #print(grad_and_vars[1])
           #grads.append([fg(grad_and_vars[0]) , grad_and_vars[1] ])
        return grads
        #return gradTrainBatch
    
    gradTrainBatch=QuantizeG(gradTrainBatch)
    
    

    #for grad_and_vars in gradTrainBatch:
      #print(grad_and_vars[0])
      #print(grad_and_vars[1])
      #print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
      #print('\n')
    
    Mom_Q=[]    
    Mom_W=[]
    
    w_vars=tf.trainable_variables()
    for w_var in w_vars:
        if w_var.name==('conv2d/kernel:0')  or   w_var.name.find('dense')>-1:
            Mom_W.append(tf.assign(w_var,w_var))
            print(w_var.name)
            print('**************************')
        #elif w_var.name.find('BatchNorm'):
            #Mom_W.append(tf.assign(w_var,fBits(w_var,16)))
            #print(w_var.name)
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%')   
            
        else:
            Mom_W.append(tf.assign(w_var,fBits(w_var,24)))
    
    
    
    with tf.control_dependencies(update_ops):
      #train_op = optimizer.minimize(loss, global_step)
      train_op = optimizer.apply_gradients(gradTrainBatch, global_step=global_step)
      opt_slot_name=optimizer.get_slot_names()
      train_vars=tf.trainable_variables()
      for train_var in train_vars:
         mom_var=optimizer.get_slot(train_var,opt_slot_name[0])         
         if train_var.name == ('conv2d/kernel:0')  or   train_var.name.find('dense')>-1:
             print(mom_var.name)
         else:
             Mom_Q.append(tf.assign(mom_var,fBits(mom_var,13)))
             #Mom_Q.append(tf.assign(mom_var,(mom_var)))
      
      train_op=tf.group([train_op,Mom_Q,Mom_W])
    
      
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
  #os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '0'
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'

  # Set up a RunConfig to only save checkpoints once per training cycle.

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction=0.99
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False

  #run_config = tf.estimator.RunConfig(session_config=config).replace(save_checkpoints_secs=1e9)
  #run_config = tf.estimator.RunConfig(session_config=config).replace(save_checkpoints_steps=10010)
  run_config = tf.estimator.RunConfig(session_config=config).replace(save_checkpoints_steps=10010)
  resnet_classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })
  
#  for var in tf.trainable_variables():
#      print(var.name)
#  
  eval_record=[]
  #train_record=[]

  for train_epoch in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy',
        'g_scale':'g_scale',
        #'grad_mom':'block_layer4/block1/conv2/conv2d/kernel/Momentum:0'
        #'grad':'block_layer4/block1/conv2/conv2d/kernel:0'
        #'kernel':'conv2d/kernel_3:0',
        #'grad':'gradients/AddN_71:0'
        #'grad':'gradients/AddN_25:0',
        #'kernel':'block_layer2/block1/conv2/conv2d/kernel:0'
        #'kernel':'block_layer4/block0/conv2/conv2d/kernel:0'
        #'BN_GAMMA':'block_layer2/block0/BatchNorm/gamma',
        #'BN_Beta':'block_layer2/block0/BatchNorm/beta'

    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

#    print('Starting to evaluate.')
#    eval_results = resnet_classifier.evaluate(
#        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
#    print(eval_results)



    class GetTensorHook(tf.train.SessionRunHook):        
      def __init__(self, tensors,datapath,every_n_iter=None, every_n_secs=None,
               at_end=False, formatter=None):
                   
        only_log_at_end = (
           at_end and (every_n_iter is None) and (every_n_secs is None))
         
        if (not only_log_at_end and (every_n_iter is None) == (every_n_secs is None)):
           raise ValueError("either at_end and/or exactly one of every_n_iter and every_n_secs "
          "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
           raise ValueError("invalid every_n_iter=%s." % every_n_iter)
#        if not isinstance(tensors, dict):
#          self._tag_order = tensors
#          tensors = {item: item for item in tensors}
#        else:
#          self._tag_order = sorted(tensors.keys())
        self._tensors = tensors
        self._datapath=datapath
        self.your_tensor=[]
        self._formatter = formatter
        self._timer = (
          tf.train.NeverTriggerTimer() if only_log_at_end else
          tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))
        self._log_at_end = at_end

      def begin(self):
         # You can add ops to the graph here.
      
         self._timer.reset()
         self._iter_count = 0
         print('Starting the session.')
         #self.your_tensor = self._tensors
         #self.your_tensor = tf.get_default_graph().get_tensor_by_name('block_layer1/block0/IdentityN_2:0')
         for tensor in self._tensors:
             self.your_tensor.append(tf.get_default_graph().get_tensor_by_name(tensor))   
         print('Begining!!!')
         
         
      def after_create_session(self, session, coord):
        # When this is called, the graph is finalized and
        # ops can no longer be added to the graph.
         print('Session created.')
      def before_run(self, run_context):
          self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
          if self._should_trigger:
             print('Before calling session.run().')
             return tf.train.SessionRunArgs(self.your_tensor)
          else:
             return None
      def after_run(self, run_context, run_values): 
#          print('Done running one step. The value of my tensor: %s',
#          run_values.results.shape())
          if self._should_trigger:
             elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
             #if elapsed_secs is  None:
             print('saving results>>>>>>>')
               #np.save(self._datapath,run_values.results)
             i=0
             for result in run_values.results:
                print(result.shape)
                i=i+1
                np.save(self._datapath+str(i)+'.npy',result)
          
          
      def end(self, session):
          print('Done with the session.')
          
    Wtensors=('block_layer1/block0/conv0/conv2d/kernel:0',
              'block_layer1/block0/conv1/conv2d/kernel:0',
              'block_layer1/block0/conv2/conv2d/kernel:0',
              'block_layer1/block1/conv1/conv2d/kernel:0',
              'block_layer1/block1/conv2/conv2d/kernel:0',
              'block_layer2/block0/conv0/conv2d/kernel:0',
              'block_layer2/block0/conv1/conv2d/kernel:0',
              'block_layer2/block0/conv2/conv2d/kernel:0',
              'block_layer2/block1/conv1/conv2d/kernel:0',
              'block_layer2/block1/conv2/conv2d/kernel:0',
              'block_layer3/block0/conv0/conv2d/kernel:0',
              'block_layer3/block0/conv1/conv2d/kernel:0',
              'block_layer3/block0/conv2/conv2d/kernel:0',
              'block_layer3/block1/conv1/conv2d/kernel:0',
              'block_layer3/block1/conv2/conv2d/kernel:0',
              'block_layer4/block0/conv0/conv2d/kernel:0',
              'block_layer4/block0/conv1/conv2d/kernel:0',
              'block_layer4/block0/conv2/conv2d/kernel:0',
              'block_layer4/block1/conv1/conv2d/kernel:0',
              'block_layer4/block1/conv2/conv2d/kernel:0')
    
    
    Wqtensors=('block_layer1/block0/IdentityN_4:0',
               'block_layer1/block0/IdentityN_7:0',
               'block_layer1/block0/IdentityN_13:0',
               'block_layer1/block1/IdentityN_4:0',
               'block_layer1/block1/IdentityN_10:0',
               'block_layer2/block0/IdentityN_4:0',
               'block_layer2/block0/IdentityN_7:0',
               'block_layer2/block0/IdentityN_13:0',
               'block_layer2/block1/IdentityN_4:0',
               'block_layer2/block1/IdentityN_10:0',
               'block_layer3/block0/IdentityN_4:0',
               'block_layer3/block0/IdentityN_7:0',
               'block_layer3/block0/IdentityN_13:0',
               'block_layer3/block1/IdentityN_4:0',
               'block_layer3/block1/IdentityN_10:0',
               'block_layer4/block0/IdentityN_4:0',
               'block_layer4/block0/IdentityN_7:0',
               'block_layer4/block0/IdentityN_13:0',
               'block_layer4/block1/IdentityN_4:0',
               'block_layer4/block1/IdentityN_10:0')
    
    Atensors=('block_layer1/block0/Relu:0',
    'block_layer1/block0/Relu_1:0',
    'block_layer1/block1/Relu:0',
    'block_layer1/block1/Relu_1:0',
    'block_layer2/block0/Relu:0',
    'block_layer2/block0/Relu_1:0',
    'block_layer2/block1/Relu:0',
    'block_layer2/block1/Relu_1:0',
    'block_layer3/block0/Relu:0',
    'block_layer3/block0/Relu_1:0',
    'block_layer3/block1/Relu:0',
    'block_layer3/block1/Relu_1:0',
    'block_layer4/block0/Relu:0',
    'block_layer4/block0/Relu_1:0',
    'block_layer4/block1/Relu:0',
    'block_layer4/block1/Relu_1:0',
    'Relu:0' )
          
    Aqtensors=(#'IteratorGetNext:0',
              'block_layer1/block0/IdentityN_2:0',
              'block_layer1/block0/IdentityN_11:0',
              'block_layer1/block1/IdentityN_2:0',
              'block_layer1/block1/IdentityN_8:0',
              'block_layer2/block0/IdentityN_2:0',
              'block_layer2/block0/IdentityN_11:0',
              'block_layer2/block1/IdentityN_2:0',
              'block_layer2/block1/IdentityN_8:0',
              'block_layer3/block0/IdentityN_2:0',
              'block_layer3/block0/IdentityN_11:0',
              'block_layer3/block1/IdentityN_2:0',
              'block_layer3/block1/IdentityN_8:0',
              'block_layer4/block0/IdentityN_2:0',
              'block_layer4/block0/IdentityN_11:0',
              'block_layer4/block1/IdentityN_2:0',
              'block_layer4/block1/IdentityN_8:0',
              'IdentityN_2:0') 
    
    BNtensors=('block_layer1/block0/BatchNorm/div_2:0',
               'block_layer1/block0/BatchNorm_1/div_2:0',
               'block_layer1/block1/BatchNorm/div_2:0',
               'block_layer1/block1/BatchNorm_1/div_2:0',
               'block_layer2/block0/BatchNorm/div_2:0',
               'block_layer2/block0/BatchNorm_1/div_2:0',
               'block_layer2/block1/BatchNorm/div_2:0',
               'block_layer2/block1/BatchNorm_1/div_2:0',
               'block_layer3/block0/BatchNorm/div_2:0',
               'block_layer3/block0/BatchNorm_1/div_2:0',
               'block_layer3/block1/BatchNorm/div_2:0',
               'block_layer3/block1/BatchNorm_1/div_2:0',
               'block_layer4/block0/BatchNorm/div_2:0',
               'block_layer4/block0/BatchNorm_1/div_2:0',
               'block_layer4/block1/BatchNorm/div_2:0',
               'block_layer4/block1/BatchNorm_1/div_2:0',
               'BatchNorm/div_2:0')
               
               
               
    BNqtensors=('block_layer1/block0/BatchNorm/IdentityN_5:0',
                'block_layer1/block0/BatchNorm_1/IdentityN_5:0',
                'block_layer1/block1/BatchNorm/IdentityN_5:0',
                'block_layer1/block1/BatchNorm_1/IdentityN_5:0',
                'block_layer2/block0/BatchNorm/IdentityN_5:0',
                'block_layer2/block0/BatchNorm_1/IdentityN_5:0',
                'block_layer2/block1/BatchNorm/IdentityN_5:0',
                'block_layer2/block1/BatchNorm_1/IdentityN_5:0',
                'block_layer3/block0/BatchNorm/IdentityN_5:0',
                'block_layer3/block0/BatchNorm_1/IdentityN_5:0',
                'block_layer3/block1/BatchNorm/IdentityN_5:0',
                'block_layer3/block1/BatchNorm_1/IdentityN_5:0',
                'block_layer4/block0/BatchNorm/IdentityN_5:0',
                'block_layer4/block0/BatchNorm_1/IdentityN_5:0',
                'block_layer4/block1/BatchNorm/IdentityN_5:0',
                'block_layer4/block1/BatchNorm_1/IdentityN_5:0',
                'BatchNorm/IdentityN_5:0')
    
    Gtensors=('gradients/AddN_42:0',
              'gradients/AddN_45:0',
              'gradients/AddN_41:0',
              'gradients/AddN_38:0',
              'gradients/AddN_36:0',
              'gradients/AddN_31:0',
              'gradients/AddN_34:0',
              'gradients/AddN_30:0',
              'gradients/AddN_27:0',
              'gradients/AddN_25:0',
              'gradients/AddN_20:0',
              'gradients/AddN_23:0',
              'gradients/AddN_19:0',
              'gradients/AddN_16:0',
              'gradients/AddN_14:0',
              'gradients/AddN_9:0',
              'gradients/AddN_12:0',
              'gradients/AddN_8:0',
              'gradients/AddN_5:0',
              'gradients/AddN_3:0') 
              
    Gqtensors=('IdentityN_12:0',
              'IdentityN_15:0',
              'IdentityN_24:0',
              'IdentityN_33:0',
              'IdentityN_42:0',
              'IdentityN_51:0',
              'IdentityN_54:0',
              'IdentityN_63:0',
              'IdentityN_72:0',
              'IdentityN_81:0',
              'IdentityN_90:0',
              'IdentityN_93:0',
              'IdentityN_102:0',
              'IdentityN_111:0',
              'IdentityN_120:0',
              'IdentityN_129:0',
              'IdentityN_132:0',
              'IdentityN_141:0',
              'IdentityN_150:0',
              'IdentityN_159:0' ) 
    
    E1tensors=('gradients/average_pooling2d/AvgPool_grad/AvgPoolGrad:0',
               'gradients/block_layer4/block1/conv2_1_grad/tuple/control_dependency:0',
               'gradients/block_layer4/block1/conv1_1_grad/tuple/control_dependency:0',
               'gradients/block_layer4/block0/conv2_1_grad/tuple/control_dependency:0',
               'gradients/AddN_11:0',
               'gradients/block_layer3/block1/conv2_1_grad/tuple/control_dependency:0',
               'gradients/block_layer3/block1/conv1_1_grad/tuple/control_dependency:0',
               'gradients/block_layer3/block0/conv2_1_grad/tuple/control_dependency:0',
               'gradients/AddN_22:0',
               'gradients/block_layer2/block1/conv2_1_grad/tuple/control_dependency:0',
               'gradients/block_layer2/block1/conv1_1_grad/tuple/control_dependency:0',
               'gradients/block_layer2/block0/conv2_1_grad/tuple/control_dependency:0',
               'gradients/AddN_33:0',
               'gradients/block_layer1/block1/conv2_1_grad/tuple/control_dependency:0',
               'gradients/block_layer1/block1/conv1_1_grad/tuple/control_dependency:0',
               'gradients/block_layer1/block0/conv2_1_grad/tuple/control_dependency:0',
               'gradients/AddN_44:0')
               
               
               
               
    E1qtensors=('gradients/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer4/block1/IdentityN_8_grad/mul_1:0',
                'gradients/block_layer4/block1/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer4/block0/IdentityN_11_grad/mul_1:0',
                'gradients/block_layer4/block0/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer3/block1/IdentityN_8_grad/mul_1:0',
                'gradients/block_layer3/block1/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer3/block0/IdentityN_11_grad/mul_1:0',
                'gradients/block_layer3/block0/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer2/block1/IdentityN_8_grad/mul_1:0',
                'gradients/block_layer2/block1/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer2/block0/IdentityN_11_grad/mul_1:0',
                'gradients/block_layer2/block0/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer1/block1/IdentityN_8_grad/mul_1:0',
                'gradients/block_layer1/block1/IdentityN_2_grad/mul_1:0',
                'gradients/block_layer1/block0/IdentityN_11_grad/mul_1:0',
                'gradients/block_layer1/block0/IdentityN_2_grad/mul_1:0')          

    E2tensors=('gradients/block_layer4/block1/add_2_grad/tuple/control_dependency:0',
               'gradients/block_layer4/block1/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer4/block0/add_3_grad/tuple/control_dependency:0',
               'gradients/block_layer4/block0/add_3_grad/tuple/control_dependency_1:0',
               'gradients/block_layer4/block0/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer3/block1/add_2_grad/tuple/control_dependency:0',
               'gradients/block_layer3/block1/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer3/block0/add_3_grad/tuple/control_dependency:0',
               'gradients/block_layer3/block0/add_3_grad/tuple/control_dependency_1:0',
               'gradients/block_layer3/block0/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer2/block1/add_2_grad/tuple/control_dependency:0',
               'gradients/block_layer2/block1/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer2/block0/add_3_grad/tuple/control_dependency:0',
               'gradients/block_layer2/block0/add_3_grad/tuple/control_dependency_1:0',
               'gradients/block_layer2/block0/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer1/block1/add_2_grad/tuple/control_dependency:0',
               'gradients/block_layer1/block1/BatchNorm_1/transpose_grad/transpose:0',
               'gradients/block_layer1/block0/add_3_grad/tuple/control_dependency:0',
               'gradients/block_layer1/block0/add_3_grad/tuple/control_dependency_1:0',
               'gradients/block_layer1/block0/BatchNorm_1/transpose_grad/transpose:0')
    
    E2qtensors=('gradients/block_layer4/block1/IdentityN_11_grad/mul_3:0',
               'gradients/block_layer4/block1/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer4/block0/IdentityN_14_grad/mul_3:0',
               'gradients/block_layer4/block0/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer4/block0/IdentityN_8_grad/mul_3:0',
               'gradients/block_layer3/block1/IdentityN_11_grad/mul_3:0',
               'gradients/block_layer3/block1/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer3/block0/IdentityN_14_grad/mul_3:0',
               'gradients/block_layer3/block0/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer3/block0/IdentityN_8_grad/mul_3:0',
               'gradients/block_layer2/block1/IdentityN_11_grad/mul_3:0',
               'gradients/block_layer2/block1/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer2/block0/IdentityN_14_grad/mul_3:0',
               'gradients/block_layer2/block0/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer2/block0/IdentityN_8_grad/mul_3:0',
               'gradients/block_layer1/block1/IdentityN_11_grad/mul_3:0',
               'gradients/block_layer1/block1/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer1/block0/IdentityN_14_grad/mul_3:0',
               'gradients/block_layer1/block0/IdentityN_5_grad/mul_3:0',
               'gradients/block_layer1/block0/IdentityN_8_grad/mul_3:0',     
     )
    
    
    WHook=GetTensorHook(tensors=Wtensors,datapath='../data/W/W',every_n_iter=100)    
    BNHook=GetTensorHook(tensors=BNtensors,datapath='../data/BN/BN',every_n_iter=100)
    AHook=GetTensorHook(tensors=Atensors,datapath='../data/A/A',every_n_iter=100)
    EHook=GetTensorHook(tensors=E2tensors,datapath='../data/E2/E2_',every_n_iter=100)
    E1Hook=GetTensorHook(tensors=E1tensors,datapath='../data/E1/E1_',every_n_iter=100)
    GHook=GetTensorHook(tensors=Gtensors,datapath='../data/G/G',every_n_iter=100)
    
    
    WQHook=GetTensorHook(tensors=Wqtensors,datapath='../data/Wq/Wq',every_n_iter=100)    
    BNQHook=GetTensorHook(tensors=BNqtensors,datapath='../data/BNq/BNq',every_n_iter=100)
    AQHook=GetTensorHook(tensors=Aqtensors,datapath='../data/Aq/Aq',every_n_iter=100)
    EQHook=GetTensorHook(tensors=E2qtensors,datapath='../data/E2q/E2q',every_n_iter=100)
    E1QHook=GetTensorHook(tensors=E1qtensors,datapath='../data/E1q/E1q',every_n_iter=100)
    GQHook=GetTensorHook(tensors=Gqtensors,datapath='../data/Gq/Gq',every_n_iter=100)



















   
    
#    print('Starting to evaluate.')
#    eval_results = resnet_classifier.evaluate(
#        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
#    print(eval_results)


#    print('Starting a training cycle.')
#    resnet_classifier.train(
#        input_fn=lambda: input_fn(
#            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
#        hooks=[logging_hook,WHook,WQHook,BNHook,BNQHook,AHook,AQHook,EHook,EQHook,GHook,GQHook])
#            #hooks=[logging_hook,GHook])
    
    print('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook,E1Hook,E1QHook])
        
    
    
    step = resnet_classifier.get_variable_value('global_step')
    
#    if step < 30*10010:
#        g_scale = 128.
#    elif 30*10010 <= step < 60*10010:  
#        g_scale = 32.
#    elif 60 <= step < 80*10010:
#        g_scale = 4.
#    else:
#        g_scale = 1.
        
    
    
    print('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print(eval_results)
    eval_record.append(eval_results)
    np.save('../data/eval_results.npy',eval_record)  
  np.save('../data/eval_results_final.npy',eval_record)



if __name__ == '__main__':
  #tf.logging.set_verbosity(tf.logging.INFO)
  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)
  fh=logging.FileHandler('../log/tensorflow.log')
  fh.setLevel(logging.DEBUG)
  log.addHandler(fh)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
