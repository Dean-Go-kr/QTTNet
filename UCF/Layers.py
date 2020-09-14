import math
import numpy as np
import tensorflow as tf
import t3f


def linear(input,
		   output_size,
		   weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
		   weights_regularizer = None,
		   biases_initializer = tf.zeros_initializer,
		   biases_regularizer = None,
		   name_scope = None):
	
	with tf.variable_scope(name_scope):
		input_size = input.get_shape()[-1].value
		tfv_weights = tf.get_variable('var_weights', [input_size, output_size], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)

		output = tf.matmul(input, tfv_weights, name = 'output_nb')
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_size], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output')

	return output


def conv_3d(input,
			output_chs,
			filter_shape,
			strides = [1, 1, 1],
			filter_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
			filter_regularizer = None,
			biases_initializer = tf.zeros_initializer,
			biases_regularizer = None,
			name_scope = None):
	
	with tf.variable_scope(name_scope):
		input_chs = input.get_shape()[-1].value
		tfv_filter = tf.get_variable('var_filter', filter_shape + [input_chs, output_chs], initializer = filter_initializer, regularizer = filter_regularizer, trainable = True)

		output = tf.nn.conv3d(input, tfv_filter, [1] + strides + [1], 'SAME', name = 'output_nb')
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_chs], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output')
	
	return output


def maxpool_3d(input,
			   ksize,
			   stride,
			   name_scope = None):
	
	with tf.variable_scope(name_scope):
		output = tf.nn.max_pool3d(input, [1] + ksize + [1], [1] + stride + [1], 'SAME', name = 'output')

	return output


def linear_tt(input,
			  output_size,
			  input_modes,
			  output_modes,
			  tt_ranks,
			  weights_regularizer = None,
			  biases_initializer = tf.zeros_initializer,
			  biases_regularizer = None,
			  name_scope = None):
	
	assert input.get_shape()[-1].value == np.prod(input_modes), 'Input modes must be the factors of input tensor.'
	assert output_size == np.prod(output_modes), 'Output modes must be the factors of output tensor.'
	assert len(input_modes) == len(output_modes), 'Modes of input and output must be equal.'
	assert len(tt_ranks) == len(input_modes) - 1, 'The number of TT ranks must be matching to the tensor modes.'

	with tf.variable_scope(name_scope):
		tt_initializer = t3f.glorot_initializer([input_modes, output_modes], tt_rank = [1] + tt_ranks + [1])
		tt_weights = t3f.get_variable('tt_weights', initializer = tt_initializer, regularizer = weights_regularizer, trainable = True)

		output = t3f.matmul(input, t3f.renormalize_tt_cores(tt_weights))
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_size], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.nn.bias_add(output, tfv_biases, name = 'output')

	return output


def conv_3d_tt(input,
			   output_chs,
			   filter_shape,
			   input_ch_modes,
			   output_ch_modes,
			   tt_ranks,
			   strides = [1, 1, 1],
			   filter_regularizer = None,
			   biases_initializer = tf.zeros_initializer,
			   biases_regularizer = None,
			   name_scope = None):
	
	assert input.get_shape()[-1].value == np.prod(input_ch_modes), 'Input modes must be the factors of the value of input channels.'
	assert output_chs == np.prod(output_ch_modes), 'Output modes must be the factors of the value of output channels.'
	assert len(input_ch_modes) == len(output_ch_modes), 'Modes of input and output channels must be equal.'
	if np.prod(filter_shape) != 1:
		assert len(tt_ranks) == len(input_ch_modes), 'The number of TT ranks must be equal to the input or output modes.'
	if np.prod(filter_shape) == 1:
		assert len(tt_ranks) == len(input_ch_modes) - 1 , 'The number of TT ranks must be matching to the tensor modes for 1x1x1 conv.'

	with tf.variable_scope(name_scope):

		filters_size = np.prod(filter_shape)
		filters_sqrt = math.sqrt(filters_size)
		lower = int(filters_sqrt)
		upper = math.ceil(filters_sqrt)
		while (True):
			if filters_size % upper == 0:
				lower = filters_size // upper
				break
			elif filters_size % lower == 0:
				upper = filters_size // lower
				break
			else:
				lower -= 1
				upper += 1
		

		if upper == 1 and lower == 1 :
			filters_shape = [input_ch_modes, output_ch_modes]
		else:
			filters_shape = [[upper] + input_ch_modes, [lower] + output_ch_modes]
		tt_initializer = t3f.glorot_initializer(filters_shape, tt_rank = [1] + tt_ranks + [1])
		tt_filters = t3f.get_variable('tt_filters', initializer = tt_initializer, regularizer = filter_regularizer, trainable = True)


		identity_matrix = tf.eye(np.prod(filters_shape[0]))
		filters = t3f.matmul(identity_matrix, t3f.renormalize_tt_cores(tt_filters))

		filters = tf.reshape(filters, [upper] + input_ch_modes + [lower] + output_ch_modes)

		inch_orders = []
		outch_orders = []
		d = len(input_ch_modes)
		for i in range(d):
			inch_orders.append(1 + i)
			outch_orders.append(2 + d + i)
		filters = tf.transpose(filters, [0, d + 1] + inch_orders + outch_orders)

		input_chs = np.prod(input_ch_modes)
		filters = tf.reshape(filters, [upper * lower, input_chs, output_chs])

		filters = tf.reshape(filters, filter_shape + [input_chs] + [output_chs])

		output = tf.nn.conv3d(input, filters, [1] + strides + [1], 'SAME', name = 'output_nb')
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_chs], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output')

	return output


def batch_normalization(input,
						tfv_train_phase,
						ema_decay = 0.99,
                        eps = 1e-3,
                        use_scale = True,
                        use_shift = True,
                        name_scope = None):
	reuse = tf.get_variable_scope().reuse
	with tf.variable_scope(name_scope):
		shape = input.get_shape().as_list()
		n_out = shape[-1]

		if len(shape) == 2:
			batch_mean, batch_variance = tf.nn.moments(input, [0], name = 'moments')
		else:
			batch_mean, batch_variance = tf.nn.moments(input, [0, 1, 2, 3], name = 'moments')
		ema = tf.train.ExponentialMovingAverage(decay = ema_decay, zero_debias = True)
		if not reuse or reuse == tf.AUTO_REUSE:
			def mean_variance_with_update():
				with tf.control_dependencies([ema.apply([batch_mean, batch_variance])]):
					return (tf.identity(batch_mean), tf.identity(batch_variance))
			mean, variance = tf.cond(tfv_train_phase, mean_variance_with_update, lambda: (ema.average(batch_mean), ema.average(batch_variance)))
		else:
			vars = tf.get_variable_scope().global_variables()
			transform = lambda s: '/'.join(s.split('/')[-5:])
			mean_name = transform(ema.average_name(batch_mean))
			variance_name = transform(ema.average_name(batch_variance))
			existed = {}
			for v in vars:
				if (transform(v.op.name) == mean_name):
					existed['mean'] = v
				if (transform(v.op.name) == variance_name):
					existed['variance'] = v
			mean, variance = tf.cond(tfv_train_phase, lambda: (batch_mean, batch_variance), lambda: (existed['mean'], existed['variance']))

		std = tf.sqrt(variance + eps, name = 'std')
		output = (input - mean) / std

		if use_scale:
			weights = tf.get_variable('weights', [n_out], initializer = tf.ones_initializer, trainable = True)
			output = tf.multiply(output, weights)

		if use_shift:
			biases = tf.get_variable('biases', [n_out], initializer = tf.zeros_initializer, trainable = True)
			output = tf.add(output, biases)

	return output


def conv_3d_tt_quan(input,
			   output_chs,
			   filter_shape,
			   input_ch_modes,
			   output_ch_modes,
			   tt_ranks,
			   strides = [1, 1, 1],
			   filter_regularizer = None,
			   biases_initializer = tf.zeros_initializer,
			   biases_regularizer = None,
			   name_scope = None):
	

	assert input.get_shape()[-1].value == np.prod(input_ch_modes), 'Input modes must be the factors of the value of input channels.'
	assert output_chs == np.prod(output_ch_modes), 'Output modes must be the factors of the value of output channels.'
	assert len(input_ch_modes) == len(output_ch_modes), 'Modes of input and output channels must be equal.'
	if np.prod(filter_shape) != 1:
		assert len(tt_ranks) == len(input_ch_modes), 'The number of TT ranks must be equal to the input or output modes.'
	if np.prod(filter_shape) == 1:
		assert len(tt_ranks) == len(input_ch_modes) - 1 , 'The number of TT ranks must be matching to the tensor modes for 1x1x1 conv.'

	with tf.variable_scope(name_scope):

		filters_size = np.prod(filter_shape)
		filters_sqrt = math.sqrt(filters_size)
		lower = int(filters_sqrt)
		upper = math.ceil(filters_sqrt)
		while (True):
			if filters_size % upper == 0:
				lower = filters_size // upper
				break
			elif filters_size % lower == 0:
				upper = filters_size // lower
				break
			else:
				lower -= 1
				upper += 1
		

		if upper == 1 and lower == 1 :
			filters_shape = [input_ch_modes, output_ch_modes]
		else:
			filters_shape = [[upper] + input_ch_modes, [lower] + output_ch_modes]
		tt_initializer = t3f.glorot_initializer(filters_shape, tt_rank = [1] + tt_ranks + [1])
		tt_filters = t3f.get_variable('tt_filters', initializer = tt_initializer, regularizer = filter_regularizer, trainable = True)


		identity_matrix = tf.eye(np.prod(filters_shape[0]))
		filters = t3f.matmul(identity_matrix, t3f.renormalize_tt_cores(tt_filters))

		filters = tf.reshape(filters, [upper] + input_ch_modes + [lower] + output_ch_modes)

		inch_orders = []
		outch_orders = []
		d = len(input_ch_modes)
		for i in range(d):
			inch_orders.append(1 + i)
			outch_orders.append(2 + d + i)
		filters = tf.transpose(filters, [0, d + 1] + inch_orders + outch_orders)

		input_chs = np.prod(input_ch_modes)
		filters = tf.reshape(filters, [upper * lower, input_chs, output_chs])

		filters = tf.reshape(filters, filter_shape + [input_chs] + [output_chs])

		output = tf.nn.conv3d(input, filters, [1] + strides + [1], 'SAME', name = 'output_nb')
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_chs], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output')

	return output
