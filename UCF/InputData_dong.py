import os
import h5py
import numpy as np
import tensorflow as tf
import cv2
import random

#import OpticalColor

base_path = '/data/dong/ucf11/dingheng/'
dataset_path = '/data/dong/ucf11/dingheng/UCF11_updated_mpg/'
stack_path = '/data/dong/ucf11/dingheng/stack/'

NORM_FRAMES = 50


def affine_transform_no_copy(cube, seed = None):
	def aug_mirror(image):
		return tf.image.flip_left_right(image)
	def aug_no_doing(image):
		return image


	n_scale = tf.random_uniform([], minval = -30, maxval = 30)


	n_range_x = 8
	n_trans_x = tf.random_uniform([], minval = -n_range_x, maxval = n_range_x, dtype = tf.int32)

	n_range_y = 4
	n_trans_y = tf.random_uniform([], minval = -n_range_y, maxval = n_range_y, dtype = tf.int32)

	for i in range(0,cube.shape[-1].value):
		slice_cube = tf.slice(cube, [0,0,0,i], [cube.shape[0], cube.shape[1], cube.shape[2], 1])
		slice_cube = tf.reshape(slice_cube, [slice_cube.shape[0], slice_cube.shape[1], slice_cube.shape[2]])


		slice_cube = tf.transpose(slice_cube, perm = [1, 2, 0])

		src_width = slice_cube.shape[1].value
		src_height = slice_cube.shape[0].value
		temp_width = src_width + src_width * n_scale / 100
		temp_height = src_height + src_height * n_scale / 100
		slice_cube = tf.image.resize_image_with_crop_or_pad(slice_cube, tf.cast(temp_height, dtype = tf.int32), tf.cast(temp_width, dtype = tf.int32))
		slice_cube = tf.image.resize_images(slice_cube, [src_height, src_width])

		slice_cube = tf.image.pad_to_bounding_box(slice_cube, n_range_y, n_range_x, src_height + n_range_y * 2, src_width + n_range_x * 2)
		slice_cube = tf.image.crop_to_bounding_box(slice_cube, n_range_y + n_trans_y, n_range_x + n_trans_x, src_height, src_width)

		slice_cube = tf.cond(seed > 55, lambda: aug_no_doing(slice_cube), lambda: aug_mirror(slice_cube))

		slice_cube = tf.transpose(slice_cube, perm = [2, 0, 1])

		if i == 0:
			out_cube = tf.expand_dims(slice_cube, axis = -1)
		else:
			out_cube = tf.concat([out_cube, tf.expand_dims(slice_cube, axis = -1)], axis = -1)

	return out_cube


def random_contrast_no_copy(cube):
	ori_shape = [cube.shape[0].value, cube.shape[1].value, cube.shape[2].value, cube.shape[3].value]
	out_cube = tf.transpose(cube, [1, 2, 0, 3])
	out_cube = tf.reshape(out_cube, [ori_shape[1], ori_shape[2], ori_shape[0] * ori_shape[3]])
	out_cube = tf.image.random_contrast(out_cube, 0.75, 1.25)
	out_cube = tf.reshape(out_cube, [ori_shape[1], ori_shape[2], ori_shape[0], ori_shape[3]])
	out_cube = tf.transpose(out_cube, [2, 0, 1, 3])
	
	return out_cube


def random_saturation_no_copy(cube):
	out_cube = tf.image.random_saturation(cube, 0.75, 1.25)
	return out_cube


def naive_augmentation_online_single_stream(input_cube, mean, std):
	return (input_cube - mean) / std


def naive_augmentation_online_no_doing(input_cube):
	return input_cube


def normal_augmentation_online_train_rgb(input_cube_rgb, mean_rgb, std_rgb, aug_seed):
	input_cube_rgb = affine_transform_no_copy(input_cube_rgb, aug_seed)
	input_cube_rgb = random_contrast_no_copy(input_cube_rgb)
	input_cube_rgb = random_saturation_no_copy(input_cube_rgb)
	
	return naive_augmentation_online_single_stream(input_cube_rgb, mean_rgb, std_rgb)


def normal_augmentation_online_train_optical(input_cube_optical, mean_optical, std_optical, aug_seed):
	return affine_transform_no_copy(input_cube_optical, aug_seed)


def get_one_group_dataset(str_log_dir, str_restore_ckpt = None, flag_optical_flow = None, flag_group = None, flag_batch_size = None):
	lst_train =[]
	lst_validation = []
	lst_train_label = []
	lst_validation_label = []

	if str_restore_ckpt is None:
		dict_labels = {}

		lst_classes = os.listdir(dataset_path)
		i = 0
		for k in range(0, len(lst_classes)):
			if os.path.isdir(dataset_path + lst_classes[k]) == True:
				dict_labels[lst_classes[k]] = i
				i = i + 1

		for word in dict_labels:
			class_path = stack_path + 'rgb/' + word + '/'
			lst_contents = os.listdir(class_path)
			for k in range(0, len(lst_contents)):
				if os.path.isfile(class_path + lst_contents[k]) == True:
					gp = int((lst_contents[k].split('_'))[-2])
					if gp != flag_group:
						lst_train.append(word + '/' + lst_contents[k])
						lst_train_label.append(dict_labels[word])
					else:
						lst_validation.append(word + '/' + lst_contents[k])
						lst_validation_label.append(dict_labels[word])

		n_seed = np.random.randint(0, 100)
		random.seed(n_seed)
		random.shuffle(lst_train)
		random.seed(n_seed)
		random.shuffle(lst_train_label)

		n_seed = np.random.randint(0, 100)
		random.seed(n_seed)
		random.shuffle(lst_validation)
		random.seed(n_seed)
		random.shuffle(lst_validation_label)

		t = flag_batch_size - len(lst_train_label) % flag_batch_size
		lst_train = lst_train + lst_train[0:t]
		lst_train_label = lst_train_label + lst_train_label[0:t]
		v = flag_batch_size - len(lst_validation_label) % flag_batch_size
		lst_validation = lst_validation + lst_validation[0:v]
		lst_validation_label = lst_validation_label + lst_validation_label[0:v]

		with h5py.File(base_path + './labels.h5', 'w') as file:
			arr_train_label = np.array(lst_train_label, dtype = np.int8)
			arr_validation_label = np.array(lst_validation_label, dtype = np.int8)
			file.create_dataset('train_label_list', data = arr_train_label)
			file.create_dataset('test_label_list', data = arr_validation_label)

		file_train_list = open(base_path + 'train_data_list.txt','w')
		i = 0
		for data in lst_train:
			file_train_list.write(data)
			file_train_list.write('\n')
			i = i + 1
		file_train_list.close()

		file_test_list = open(base_path + 'test_data_list.txt','w')
		i = 0
		for data in lst_validation:
			file_test_list.write(data)
			file_test_list.write('\n')
			i = i + 1
		file_test_list.close()
	else:
		f_train_list = open(base_path + 'train_data_list.txt','r')
		for data in f_train_list:
			if data.strip() == '':
				break
			else:
				lst_train.append(data.strip('\n'))
		f_train_list.close()

		f_test_list = open(base_path + 'test_data_list.txt','r')
		for data in f_test_list:
			if data.strip() == '':
				break
			else:
				lst_validation.append(data.strip('\n'))
		f_test_list.close()

		with h5py.File(base_path + 'labels.h5','r') as file:
			arr_train_label = file.get('train_label_list').value
			arr_validation_label = file.get('test_label_list').value

	with h5py.File(base_path + 'mean_std.h5','r') as file:
		arr_mean_rgb = file.get('rgb_mean').value
		if flag_optical_flow:
			arr_mean_opticalflow = file.get('opticalflow_mean').value
		else:
			arr_mean_opticalflow = arr_mean_rgb
		arr_std_rgb = file.get('rgb_std').value
		if flag_optical_flow:
			arr_std_opticalflow = file.get('opticalflow_std').value
		else:
			arr_std_opticalflow = arr_std_rgb

	lst_mean_rgb = []
	lst_mean_opticalflow = []
	lst_std_rgb = []
	lst_std_opticalflow = []
	for i in range(NORM_FRAMES):
		lst_mean_rgb.append(arr_std_rgb)
		lst_mean_opticalflow.append(arr_mean_opticalflow)
		lst_std_rgb.append(arr_std_rgb)
		lst_std_opticalflow.append(arr_std_opticalflow)

	dict_dataset = {}
	dict_dataset['train'] = {
		'train_labels' : arr_train_label,
		'train_data_list' : lst_train
		}
	dict_dataset['validation'] = {
		'validation_labels' : arr_validation_label,
		'validation_data_list' : lst_validation
		}

	dict_mean_std = {}
	dict_mean_std['mean'] = {
		'mean_rgb' : np.array(lst_mean_rgb),
		'mean_opticalflow' : np.array(lst_mean_opticalflow)
		}
	dict_mean_std['std'] = {
		'std_rgb' : np.array(lst_std_rgb),
		'std_opticalflow' : np.array(lst_std_opticalflow)
		}

	return dict_dataset, dict_mean_std


def construct_batch_part_two_stream(dict_mean_std, flag_batch_size, tfv_aug_online = None, flag_optical_flow = True):
	sample_rgb = dict_mean_std['mean']['mean_rgb']
	shape_rgb = [sample_rgb.shape[0], sample_rgb.shape[1], sample_rgb.shape[2], sample_rgb.shape[3]]
	if flag_optical_flow:
		sample_optical = dict_mean_std['mean']['mean_opticalflow']
		shape_optical = [sample_optical.shape[0], sample_optical.shape[1], sample_optical.shape[2], sample_optical.shape[3]]
	else:
		shape_optical = shape_rgb

	tfph_train_cubes_rgb = tf.placeholder(dtype = tf.float32, shape = [flag_batch_size] + shape_rgb, name = 'ph_train_cubes_rgb')
	tfph_train_cubes_optical = tf.placeholder(dtype = tf.float32, shape = [flag_batch_size] + shape_optical, name = 'ph_train_cubes_optical')
	tfph_train_labels = tf.placeholder(dtype = tf.int32, shape = [flag_batch_size], name = 'ph_train_labels')

	tfph_validation_cubes_rgb = tf.placeholder(dtype = tf.float32, shape = [flag_batch_size] + shape_rgb, name = 'ph_validation_cubes_rgb')
	tfph_validation_cubes_optical = tf.placeholder(dtype = tf.float32, shape = [flag_batch_size] + shape_optical, name = 'ph_validation_cubes_optical')
	tfph_validation_labels = tf.placeholder(dtype = tf.int32, shape = [flag_batch_size], name = 'ph_validation_labels')

	tfph_mean_rgb = tf.placeholder(dtype = tf.float32, shape = shape_rgb, name = 'ph_mean_rgb')
	tfph_mean_optical = tf.placeholder(dtype = tf.float32, shape = shape_optical, name = 'ph_mean_optical')
	tfph_std_rgb = tf.placeholder(dtype = tf.float32, shape = shape_rgb, name = 'ph_std_rgb')
	tfph_std_optical = tf.placeholder(dtype = tf.float32, shape = shape_optical, name = 'ph_std_optical')

	for k in range(0, flag_batch_size):
		input_train_cube_rgb = tfph_train_cubes_rgb[k]
		input_train_cube_optical = tfph_train_cubes_optical[k]

		input_validation_cube_rgb = tfph_validation_cubes_rgb[k]
		input_validation_cube_optical = tfph_validation_cubes_optical[k]

		if tfv_aug_online is not None:
			aug_train_cube_rgb = tf.cond(tfv_aug_online[k] > 10, \
				lambda: normal_augmentation_online_train_rgb(input_train_cube_rgb, tfph_mean_rgb, tfph_std_rgb, tfv_aug_online[k]), \
				lambda: naive_augmentation_online_single_stream(input_train_cube_rgb, tfph_mean_rgb, tfph_std_rgb))
			aug_train_cube_optical = tf.cond(tfv_aug_online[k] > 10, \
				lambda: normal_augmentation_online_train_optical(input_train_cube_optical, tfph_mean_optical, tfph_std_optical, tfv_aug_online[k]), \
				lambda: naive_augmentation_online_no_doing(input_train_cube_optical))
		else:
			aug_train_cube_rgb = naive_augmentation_online_single_stream(input_train_cube_rgb, tfph_mean_rgb, tfph_std_rgb)
			aug_train_cube_optical = input_train_cube_optical
		aug_validation_cube_rgb = naive_augmentation_online_single_stream(input_validation_cube_rgb, tfph_mean_rgb, tfph_std_rgb)
		aug_validation_cube_optical = input_validation_cube_optical

		if k == 0:
			batch_train_cubes_rgb = tf.expand_dims(aug_train_cube_rgb, 0)
			batch_train_cubes_optical = tf.expand_dims(aug_train_cube_optical, 0)
			batch_validation_cubes_rgb = tf.expand_dims(aug_validation_cube_rgb, 0)
			batch_validation_cubes_optical = tf.expand_dims(aug_validation_cube_optical, 0)
		else:
			batch_train_cubes_rgb = tf.concat([batch_train_cubes_rgb, tf.expand_dims(aug_train_cube_rgb, 0)], axis = 0)
			batch_train_cubes_optical = tf.concat([batch_train_cubes_optical, tf.expand_dims(aug_train_cube_optical, 0)], axis = 0)
			batch_validation_cubes_rgb = tf.concat([batch_validation_cubes_rgb, tf.expand_dims(aug_validation_cube_rgb, 0)], axis = 0)
			batch_validation_cubes_optical = tf.concat([batch_validation_cubes_optical, tf.expand_dims(aug_validation_cube_optical, 0)], axis = 0)

	result = {}
	result['batches'] = {
		'batch_train_cubes_rgb' : batch_train_cubes_rgb,
		'batch_train_cubes_optical' : batch_train_cubes_optical,
		'batch_train_labels' : tfph_train_labels,
		'batch_validation_cubes_rgb' : batch_validation_cubes_rgb,
		'batch_validation_cubes_optical' : batch_validation_cubes_optical,
		'batch_validation_labels' : tfph_validation_labels
		}
	result['input_placeholders'] = {
		'tfph_train_cubes_rgb' : tfph_train_cubes_rgb,
		'tfph_train_cubes_optical' : tfph_train_cubes_optical,
		'tfph_train_labels' : tfph_train_labels,
		'tfph_validation_cubes_rgb' : tfph_validation_cubes_rgb,
		'tfph_validation_cubes_optical' : tfph_validation_cubes_optical,
		'tfph_validation_labels' : tfph_validation_labels,
		'tfph_mean_rgb' : tfph_mean_rgb,
		'tfph_mean_optical' : tfph_mean_optical,
		'tfph_std_rgb' : tfph_std_rgb,
		'tfph_std_optical' : tfph_std_optical
		}
	return result


def get_batch_part_train(dict_dataset, dict_mean_std, dict_placeholders, n_index_head, flag_batch_size, flag_optical_flow):
	n_size = dict_dataset['train']['train_labels'].shape[0]
	assert n_size % flag_batch_size == 0, 'Batch size must be divided extractly.'

	n_index_end = n_index_head + flag_batch_size
	if n_index_end > n_size:
		n_index_end = n_size - 1
		n_index_head = n_index_end - flag_batch_size


	lst_volume_rgb = []
	lst_volume_optic = []
	for data_str in dict_dataset['train']['train_data_list'][n_index_head:n_index_end]:

		if data_str.split('/')[0] == 'mirror':
			mirror_dir = 'mirror/'
			data_str = data_str.split('/')[1] + '/' + data_str.split('/')[2]
		else:
			mirror_dir = ''

		if os.path.exists(stack_path + mirror_dir + 'rgb/' + data_str) == True:
			with h5py.File(stack_path + mirror_dir + 'rgb/' + data_str,'r') as file:
				arr_ori_frames_rgb = file.get('stack').value
			with h5py.File(stack_path + mirror_dir + 'opticalflow/' + data_str,'r') as file:
				arr_ori_frames_optic = file.get('stack').value
			
			n_frame_num = arr_ori_frames_rgb.shape[0]
			n_anchor = np.random.randint(0, (n_frame_num + 1) - NORM_FRAMES)
			lst_frames_rgb = []
			lst_frames_optic = []
			for f in range(NORM_FRAMES):
				lst_frames_rgb.append(arr_ori_frames_rgb[n_anchor])
				if flag_optical_flow:
					lst_frames_optic.append(arr_ori_frames_optic[n_anchor])
				else:
					lst_frames_optic.append(lst_frames_rgb[-1])
				n_anchor = n_anchor + 1
			arr_frames_rgb = np.array(lst_frames_rgb)
			arr_frames_optic = np.array(lst_frames_optic)
		else:
			return None

		lst_volume_rgb.append(arr_frames_rgb)
		lst_volume_optic.append(arr_frames_optic)
	arr_volume_rgb = np.array(lst_volume_rgb)
	arr_volume_optic = np.array(lst_volume_optic)

	dict_feeder = {
		dict_placeholders['tfph_train_cubes_rgb'] : arr_volume_rgb,
		dict_placeholders['tfph_train_cubes_optical'] : arr_volume_optic,
		dict_placeholders['tfph_train_labels'] : dict_dataset['train']['train_labels'][n_index_head:n_index_end],
		dict_placeholders['tfph_validation_cubes_rgb'] : arr_volume_rgb,
		dict_placeholders['tfph_validation_cubes_optical'] : arr_volume_optic,
		dict_placeholders['tfph_validation_labels'] : dict_dataset['validation']['validation_labels'][0:flag_batch_size],
		dict_placeholders['tfph_mean_rgb'] : dict_mean_std['mean']['mean_rgb'],
		dict_placeholders['tfph_mean_optical'] : dict_mean_std['mean']['mean_opticalflow'],
		dict_placeholders['tfph_std_rgb'] : dict_mean_std['std']['std_rgb'],
		dict_placeholders['tfph_std_optical'] : dict_mean_std['std']['std_opticalflow']
		}

	return dict_feeder


def get_batch_part_validation(dict_dataset, dict_mean_std, dict_placeholders, n_index_head, flag_batch_size, flag_optical_flow):
	n_size = dict_dataset['validation']['validation_labels'].shape[0]
	assert n_size % flag_batch_size == 0, 'Batch size must be divided extractly.'

	n_index_end = n_index_head + flag_batch_size
	if n_index_end > n_size:
		n_index_end = n_size - 1
		n_index_head = n_index_end - flag_batch_size

	lst_volume_rgb = []
	lst_volume_optic = []
	for data_str in dict_dataset['validation']['validation_data_list'][n_index_head:n_index_end]:
		
		if os.path.exists(stack_path + 'rgb/' + data_str) == True:
			with h5py.File(stack_path + 'rgb/' + data_str,'r') as file:
				arr_ori_frames_rgb = file.get('stack').value
			with h5py.File(stack_path + 'opticalflow/' + data_str,'r') as file:
				arr_ori_frames_optic = file.get('stack').value
			
			n_frame_num = arr_ori_frames_rgb.shape[0]
			n_anchor = np.random.randint(0, (n_frame_num + 1) - NORM_FRAMES)
			lst_frames_rgb = []
			lst_frames_optic = []
			for f in range(NORM_FRAMES):
				lst_frames_rgb.append(arr_ori_frames_rgb[n_anchor])
				if flag_optical_flow:
					lst_frames_optic.append(arr_ori_frames_optic[n_anchor])
				else:
					lst_frames_optic.append(lst_frames_rgb[-1])
				n_anchor = n_anchor + 1
			arr_frames_rgb = np.array(lst_frames_rgb)
			arr_frames_optic = np.array(lst_frames_optic)
		else:
			return None

		lst_volume_rgb.append(arr_frames_rgb)
		lst_volume_optic.append(arr_frames_optic)
	arr_volume_rgb = np.array(lst_volume_rgb)
	arr_volume_optic = np.array(lst_volume_optic)

	dict_feeder = {
		dict_placeholders['tfph_train_cubes_rgb'] : arr_volume_rgb,
		dict_placeholders['tfph_train_cubes_optical'] : arr_volume_optic,
		dict_placeholders['tfph_train_labels'] : dict_dataset['train']['train_labels'][0:flag_batch_size],
		dict_placeholders['tfph_validation_cubes_rgb'] : arr_volume_rgb,
		dict_placeholders['tfph_validation_cubes_optical'] : arr_volume_optic,
		dict_placeholders['tfph_validation_labels'] : dict_dataset['validation']['validation_labels'][n_index_head:n_index_end],
		dict_placeholders['tfph_mean_rgb'] : dict_mean_std['mean']['mean_rgb'],
		dict_placeholders['tfph_mean_optical'] : dict_mean_std['mean']['mean_opticalflow'],
		dict_placeholders['tfph_std_rgb'] : dict_mean_std['std']['std_rgb'],
		dict_placeholders['tfph_std_optical'] : dict_mean_std['std']['std_opticalflow']
		}

	return dict_feeder
