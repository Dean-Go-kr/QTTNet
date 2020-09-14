import tensorflow as tf
import numpy as np
import t3f

import Layers
import Layers_quan

NUM_CLASSES = 11


def neural_network_tiny(cubes, labels, tfv_train_phase = None, name = None):
	if name is not None:
		name = '_' + name

	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope('neural_network_tiny' + name):
		l_layers = []

		l_layers.append(cubes)

		l_layers.append(Layers.conv_3d(l_layers[-1], 8, [3,3,3], name_scope = 'conv_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_conv_1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_1'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [2,2,2], name_scope = 'pool_1'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 16, [3,3,3], name_scope = 'conv_2'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_conv_2'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_2'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [2,2,2], name_scope = 'pool_2'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 64, [3,3,3], name_scope = 'conv_3'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_conv_3'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_3'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,2,2], [2,2,2], name_scope = 'pool_3'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 128, [3,3,3], name_scope = 'conv_4'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_conv_4'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_4'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,2,2], [2,2,2], name_scope = 'pool_4'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 256, [3,3,3], name_scope = 'conv_5'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_conv_5'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_5'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,2,2], [2,2,2], name_scope = 'pool_5'))

		l_layers.append(tf.nn.avg_pool3d(l_layers[-1], [1,2,2,3,1], [1,2,2,3,1], 'SAME', name = 'avg_pool'))

		l_layers.append(Layers.linear(tf.reshape(l_layers[-1], [-1, np.prod(l_layers[-1].get_shape().as_list()[1:])]), 256, name_scope = 'linear_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_linear_1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_linear_1'))
		l_layers.append(tf.nn.dropout(l_layers[-1], dropout_rate(0.5), name = 'dropout_1'))

		l_layers.append(Layers.linear(l_layers[-1], NUM_CLASSES, name_scope = 'linear_3'))

	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = l_layers[-1], name = 'softmax_xentropy' + name)
	losses = tf.reduce_mean(xentropy, name = 'losses' + name)
	total_loss = tf.add_n([losses], name = 'total_loss' + name)
	loss_averages = tf.train.ExponentialMovingAverage(0.99, name = 'avg_loss' + name)
	tfop_loss_averages = loss_averages.apply([losses] + [total_loss])
	with tf.control_dependencies([tfop_loss_averages]):
		total_loss = tf.identity(total_loss)
	correct_flags = tf.nn.in_top_k(l_layers[-1], labels, 1, name = 'eval' + name)
	evaluation = tf.cast(correct_flags, tf.int32)

	return total_loss, evaluation, l_layers[-1]



def neural_network_struct(cubes, labels, tfv_train_phase = None, name = None):
	if name is not None:
		name = '_' + name
	
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope('neural_network_original' + name):
		l_layers = []


		l_layers.append(cubes)
		print(l_layers[-1])
		#--------------- Conv block 1 ---------------#
		l_layers.append(Layers.conv_3d(l_layers[-1], 64, [3,3,3], name_scope = 'conv_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_1'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 64, [3,3,3], name_scope = 'conv_2'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_2'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_2'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [4,4,4], name_scope = 'pool_1'))
		print(l_layers[-1])
		#-------------- Conv block 2 ---------------#
		l_layers.append(Layers.conv_3d(l_layers[-1], 128, [3,3,3], name_scope = 'conv_3'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_3'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_3'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 128, [3,3,3], name_scope = 'conv_4'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_4'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_4'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [4,3,4], name_scope = 'pool_2'))
		print(l_layers[-1])
		'''
		#-------------- Conv block 3 ---------------#
		l_layers.append(Layers.conv_3d(l_layers[-1], 192, [3,3,3], name_scope = 'conv_5'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_5'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_5'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 192, [3,3,3], name_scope = 'conv_6'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_6'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_6'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [3,3,3], name_scope = 'pool_3'))
		print(l_layers[-1])
		'''
		#-------------- Conv block 4 ---------------#
		l_layers.append(Layers.conv_3d(l_layers[-1], 256, [3,3,3], name_scope = 'conv_7'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_7'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_5'))

		l_layers.append(Layers.conv_3d(l_layers[-1], 256, [3,3,3], name_scope = 'conv_8'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_8'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_8'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [4,3,3], name_scope = 'pool_4'))
		print(l_layers[-1])
		# FC layer
		
		print([-1, np.prod(l_layers[-1].get_shape().as_list()[1:])])
		'''
		l_layers.append(Layers.linear(tf.reshape(l_layers[-1], [-1, np.prod(l_layers[-1].get_shape().as_list()[1:])]), 256, name_scope = 'linear_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_linear_1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_linear_1'))
		l_layers.append(tf.nn.dropout(l_layers[-1], dropout_rate(0.5), name = 'dropout_1'))
		'''
		l_layers.append(Layers.linear(tf.reshape(l_layers[-1], [-1, np.prod(l_layers[-1].get_shape().as_list()[1:])]), NUM_CLASSES, name_scope = 'linear_last'))
		

	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = l_layers[-1], name = 'softmax_xentropy' + name)
	losses = tf.reduce_mean(xentropy, name = 'losses' + name)
	total_loss = tf.add_n([losses], name = 'total_loss' + name)
	loss_averages = tf.train.ExponentialMovingAverage(0.99, name = 'avg_loss' + name)
	tfop_loss_averages = loss_averages.apply([losses] + [total_loss])
	with tf.control_dependencies([tfop_loss_averages]):
		total_loss = tf.identity(total_loss)
	correct_flags = tf.nn.in_top_k(l_layers[-1], labels, 1, name = 'eval' + name)
	evaluation = tf.cast(correct_flags, tf.int32)

	return total_loss, evaluation, l_layers[-1]



def neural_network_struct_tt_conv_fc(cubes, labels, tfv_train_phase = None, name = None):
	if name is not None:
		name = '_' + name
	
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope('neural_network_tt_conv_fc' + name):
		l_layers = []
		l_layers.append(cubes)

		#--------------- Conv block 1 ---------------#
		l_layers.append(Layers.conv_3d(l_layers[-1],64, [3,3,3], name_scope = 'conv_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_1'))

		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 64, [3,3,3], [4,4,4], [4,4,4], [16,16,16], name_scope = 'conv_tt_2'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_2'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_2'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [4,4,4], [2,2,2], name_scope = 'pool_1'))
		print(l_layers[-1])
		#--------------- Conv block 2 ---------------#
		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 128, [3,3,3], [4,2,4,2], [2,4,4,4], [16,16,16,16], name_scope = 'conv_tt_3'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_3'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_3'))

		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 128, [3,3,3], [4,2,4,4], [4,4,2,4], [16,16,16,16], name_scope = 'conv_tt_4'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_4'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_4'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [4,4,4], [2,2,2], name_scope = 'pool_2'))
		print(l_layers[-1])
		#--------------- Conv block 3 ---------------#
		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 192, [3,3,3], [4,2,4,4], [4,4,3,4], [16,16,16,16], name_scope = 'conv_tt_5'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_5'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_5'))
		
		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 192, [3,3,3], [4,3,4,4], [4,4,3,4], [16,16,16,16], name_scope = 'conv_tt_6'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_6'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_6'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [4,4,4], [2,2,2], name_scope = 'pool_3'))
		print(l_layers[-1])
		#--------------- Conv block 4 ---------------#
		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 256, [3,3,3], [4,3,4,4], [4,4,4,4], [16,16,16,16], name_scope = 'conv_tt_7'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_7'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_7'))
		
		l_layers.append(Layers.conv_3d_tt(l_layers[-1], 256, [3,3,3], [4,4,4,4], [4,4,4,4], [16,16,16,16], name_scope = 'conv_tt_8'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_8'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_8'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [4,4,4], [2,2,2], name_scope = 'pool_44'))
		print(l_layers[-1]) 
		# FC layer
		print([-1, np.prod(l_layers[-1].get_shape().as_list()[1:])])
		l_layers.append(Layers.linear(tf.reshape(l_layers[-1], [-1, np.prod(l_layers[-1].get_shape().as_list()[1:])]), NUM_CLASSES, name_scope = 'linear_1'))

	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = l_layers[-1], name = 'softmax_xentropy' + name)
	losses = tf.reduce_mean(xentropy, name = 'losses' + name)
	total_loss = tf.add_n([losses], name = 'total_loss' + name)
	loss_averages = tf.train.ExponentialMovingAverage(0.99, name = 'avg_loss' + name)
	tfop_loss_averages = loss_averages.apply([losses] + [total_loss])
	with tf.control_dependencies([tfop_loss_averages]):
		total_loss = tf.identity(total_loss)
	correct_flags = tf.nn.in_top_k(l_layers[-1], labels, 1, name = 'eval' + name)
	evaluation = tf.cast(correct_flags, tf.int32)

	return total_loss, evaluation, l_layers[-1]




def neural_network_struct_tt_quan_conv_fc(cubes, labels, tfv_train_phase = None, name = None):
	if name is not None:
		name = '_' + name
	
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope('neural_network_tt_conv_fc' + name):
		l_layers = []
		l_layers.append(cubes)

		#--------------- Conv block 1 ---------------#
		l_layers.append(Layers.conv_3d(l_layers[-1],64, [3,3,3], name_scope = 'conv_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_1'))

		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 64, [3,3,3], [4,4,4], [4,4,4], [16,16,16], name_scope = 'conv_tt_2'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_2'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_2'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [4,4,4], name_scope = 'pool_1'))

		print(l_layers[-1])
		#--------------- Conv block 2 ---------------#
		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 128, [3,3,3], [4,2,4,2], [2,4,4,4], [16,16,16,16], name_scope = 'conv_tt_3'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_3'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_3'))

		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 128, [3,3,3], [4,2,4,4], [4,4,2,4], [16,16,16,16], name_scope = 'conv_tt_4'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_4'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_4'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [4,3,4], name_scope = 'pool_2'))
		print(l_layers[-1])
		'''
		#--------------- Conv block 3 ---------------#
		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 192, [3,3,3], [4,2,4,4], [4,4,3,4], [16,16,16,16], name_scope = 'conv_tt_5'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_5'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_5'))
		
		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 192, [3,3,3], [4,3,4,4], [4,4,3,4], [16,16,16,16], name_scope = 'conv_tt_6'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_6'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_6'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [3,3,3], name_scope = 'pool_3'))
		print(l_layers[-1])
		'''
		#--------------- Conv block 4 ---------------#
		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 256, [3,3,3], [4,2,4,4], [4,4,4,4], [16,16,16,16], name_scope = 'conv_tt_7'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_7'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_7'))
		
		l_layers.append(Layers_quan.conv_3d_tt(l_layers[-1], 256, [3,3,3], [4,4,4,4], [4,4,4,4], [16,16,16,16], name_scope = 'conv_tt_8'))
		l_layers.append(Layers_quan.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_8'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_conv_8'))
		l_layers.append(Layers.maxpool_3d(l_layers[-1], [3,3,3], [4,3,3], name_scope = 'pool_44'))
		print(l_layers[-1]) 
		# FC layer
		print([-1, np.prod(l_layers[-1].get_shape().as_list()[1:])])
		'''
		l_layers.append(Layers.linear_tt(tf.reshape(l_layers[-1],[-1, np.prod(l_layers[-1].get_shape().as_list()[1:])]), 256, [8,4,8,8], [4,4,4,4], [16,16,16], name_scope='linear_tt_1'))
		l_layers.append(Layers.batch_normalization(l_layers[-1], tfv_train_phase, name_scope = 'bn_l1'))
		l_layers.append(tf.nn.relu(l_layers[-1], name = 'relu_linear_1'))
		l_layers.append(tf.nn.dropout(l_layers[-1], dropout_rate(0.8), name = 'dropout_1'))
		'''
		l_layers.append(Layers.linear(tf.reshape(l_layers[-1], [-1, np.prod(l_layers[-1].get_shape().as_list()[1:])]), NUM_CLASSES, name_scope = 'linear_last'))

	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = l_layers[-1], name = 'softmax_xentropy' + name)
	losses = tf.reduce_mean(xentropy, name = 'losses' + name)
	total_loss = tf.add_n([losses], name = 'total_loss' + name)
	loss_averages = tf.train.ExponentialMovingAverage(0.99, name = 'avg_loss' + name)
	tfop_loss_averages = loss_averages.apply([losses] + [total_loss])
	with tf.control_dependencies([tfop_loss_averages]):
		total_loss = tf.identity(total_loss)
	correct_flags = tf.nn.in_top_k(l_layers[-1], labels, 1, name = 'eval' + name)
	evaluation = tf.cast(correct_flags, tf.int32)

	return total_loss, evaluation, l_layers[-1]
