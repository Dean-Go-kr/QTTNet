import os
import cv2
import h5py
import numpy as np


base_path = '/data/'
dataset_path = '/data/UCF/UCF11_updated_mpg/'
stack_path = '/data/UCF/stack/'

g_dict_labels = {'basketball':1, 'biking':2, 'diving':3, 'golf_swing':4, 'horse_riding':5,
				 'soccer_juggling':6, 'swing':7, 'tennis_swing':8, 'trampoline_jumping':9, 'volleyball_spiking':10, 'walking':11}

g_batch_size = 100

TARGET_WIDTH = 40
TARGET_HEIGHT = 30
NORM_FRAMES = 40

# Label
def construct_dict():
	if os.path.exists(dataset_path) == False:
		return False

	lst_contents = os.listdir(dataset_path)
	i = 0
	for k in range(0, len(lst_contents)):
		if os.path.isdir(dataset_path + lst_contents[k]) == True:
			g_dict_labels[lst_contents[k]] = i
			i = i + 1

	return True


def Calc_mean_std_and_stack(n_nums, n_count, l_data = None, l_label = None):
	if l_data is None or l_label is None:
		return None
	if len(l_data) != len(l_label):
		return None

	lst_volume_rgb = []
	lst_volume_optic = []
	for i in range(0,len(l_data)):
		lst_frames_rgb = []
		lst_frames_optic = []
		str_class_name = list(g_dict_labels.keys())[list(g_dict_labels.values()).index(l_label[i])]

		rgb_stack_str = stack_path + 'rgb/' + str_class_name + '/' + l_data[i].split('.')[0] + '.h5'
		optic_stack_str = stack_path + 'opticalflow/' + str_class_name + '/' + l_data[i].split('.')[0] + '.h5'
		if os.path.exists(rgb_stack_str) and os.path.exists(optic_stack_str):
			with h5py.File(rgb_stack_str,'r') as file:
				arr_frames_rgb = file.get('stack').value
			with h5py.File(optic_stack_str,'r') as file:
				arr_frames_optic = file.get('stack').value
			lst_volume_rgb.append(arr_frames_rgb[0])
			lst_volume_optic.append(arr_frames_optic[0])
			n_count = n_count + 1
			print('%d/%d. Standard score and stacking process for %s has done.'%(n_count, n_nums, l_data[i]))
			continue

		data_str = dataset_path + str_class_name + '/' + l_data[i]
		cv_cap = cv2.VideoCapture(data_str)
		n_frame_num = int(cv_cap.get(7))
		if n_frame_num < 2:
			n_count = n_count + 1
			print('%d/%d. Standard score and stacking process for %s has done.'%(n_count, n_nums, l_data[i]))
			continue

		lst_frames_rgb = []
		lst_frames_nums = []
		for k in range(n_frame_num):
			cv_cap.set(cv2.CAP_PROP_POS_FRAMES, k)
			b_flag, frame = cv_cap.read()
			if frame is not None:
				rs_frame = cv2.resize(frame, (TARGET_WIDTH,TARGET_HEIGHT), 0, 0, interpolation = cv2.INTER_CUBIC)
				lst_frames_rgb.append(rs_frame)
				lst_frames_nums.append(k)

		if len(lst_frames_nums) == 0:
			n_count = n_count + 1
			print('%d/%d. Standard score and stacking process for %s has done.'%(n_count, n_nums, l_data[i]))
			continue

		while len(lst_frames_nums) < NORM_FRAMES:
			lst_frames_rgb.append(lst_frames_rgb[-1])
			lst_frames_nums.append(lst_frames_nums[-1])
		arr_frames_rgb = np.array(lst_frames_rgb, dtype = np.uint8)

		lst_frames_optic = []
		b_any_problem = False
		for k in range(len(lst_frames_nums)):
			if k == len(lst_frames_nums) - 1:
				break

			cv_cap.set(cv2.CAP_PROP_POS_FRAMES, lst_frames_nums[k])
			_, pre_frame = cv_cap.read()
			cv_cap.set(cv2.CAP_PROP_POS_FRAMES, lst_frames_nums[k + 1])
			_, next_frame = cv_cap.read()
			gray_pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
			gray_next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(gray_pre_frame, gray_next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			if flow is not None:
				flow_store = cv2.resize(flow, (TARGET_WIDTH,TARGET_HEIGHT), 0, 0, interpolation = cv2.INTER_CUBIC)
				lst_frames_optic.append(flow_store)
			else:
				lst_frames_optic.append([])
				b_any_problem = True
		lst_frames_optic.append(lst_frames_optic[-1])

		if b_any_problem:
			for k in range(len(lst_frames_optic)):
				if k == 0 and lst_frames_optic[k] == []:
					p = 1
					while lst_frames_optic[p] == []:
						p = p + 1
					lst_frames_optic[k] = lst_frames_optic[p]
				else:
					if lst_frames_optic[k] == []:
						p = k + 1
						if p == len(lst_frames_optic):
							lst_frames_optic[k] = lst_frames_optic[k - 1]
						else:
							while lst_frames_optic[p] == []:
								p = p + 1
								if p == len(lst_frames_optic):
									break
							if p == len(lst_frames_optic):
								lst_frames_optic[k] = lst_frames_optic[k - 1]
							else:
								lst_frames_optic[k] = np.mean(np.array([lst_frames_optic[k - 1], lst_frames_optic[p]]), axis = 0)
		arr_frames_optic = np.array(lst_frames_optic, dtype = np.float32)

		data_name = l_data[i].split('.')[0]
		str_rgb_path = stack_path + 'rgb/' + str_class_name + '/'
		str_optic_path = stack_path + 'opticalflow/' + str_class_name + '/'
		if os.path.exists(str_rgb_path) == False:
			os.mkdir(str_rgb_path)
		if os.path.exists(str_optic_path) == False:
			os.mkdir(str_optic_path)
		with h5py.File(str_rgb_path + data_name + '.h5', 'w') as file:
			file.create_dataset('stack', data = arr_frames_rgb)
		with h5py.File(str_optic_path + data_name + '.h5', 'w') as file:
			file.create_dataset('stack', data = arr_frames_optic)

		lst_volume_rgb.append(arr_frames_rgb[0])
		lst_volume_optic.append(arr_frames_optic[0])
		n_count = n_count + 1
		print('%d/%d. Standard score and stacking process for %s has done.'%(n_count, n_nums, l_data[i]))

	arr_volume_rgb = np.array(lst_volume_rgb, dtype = np.uint8)
	arr_volume_optic = np.array(lst_volume_optic, dtype = np.float32)

	arr_mean_rgb = np.mean(arr_volume_rgb, dtype = np.float32, axis = 0)
	arr_mean_optic = np.mean(arr_volume_optic, dtype = np.float32, axis = 0)
	arr_std_rgb = np.std(arr_volume_rgb, dtype = np.float32, axis = 0)
	arr_std_optic = np.std(arr_volume_optic, dtype = np.float32, axis = 0)

	result = {}
	result['mean'] = {
		'mean_rgb' : arr_mean_rgb,
		'mean_opticalflow' : arr_mean_optic
		}
	result['std'] = {
		'std_rgb' : arr_std_rgb,
		'std_opticalflow' : arr_std_optic
		}

	return n_count, result


def read_file_list():
	lst_data = []
	lst_label = []

	for word in g_dict_labels:
		class_path = dataset_path + word + '/'
		print(class_path)

		if os.path.exists(class_path) == False:
			return None

		lst_contents = os.listdir(class_path)
		print(lst_contents)
		print(class_path + lst_contents[0])
		for k in range(0, len(lst_contents)):
			if os.path.isfile(class_path + lst_contents[k]) == True:

				lst_data.append(lst_contents[k])
				lst_label.append(g_dict_labels[word])

	return (lst_data, lst_label)


def main():
	if os.path.exists(stack_path) == False:
		os.mkdir(stack_path)
	if os.path.exists(stack_path + 'rgb/') == False:
		os.mkdir(stack_path + 'rgb/')
	if os.path.exists(stack_path + 'opticalflow/') == False:
		os.mkdir(stack_path + 'opticalflow/')

	if construct_dict() == False:
		print('There is no classInd file.')
		return None

	tuple_data_label = read_file_list()
	print(tuple_data_label)
	if tuple_data_label is None:
		print('There is no data or label file.')
		return None

	lst_mean_rgb = []
	lst_mean_opt = []
	lst_std_rgb = []
	lst_std_opt = []
	nlen = len(tuple_data_label[0]) // g_batch_size
	n_nums = len(tuple_data_label[0])
	n_count = 0
	for i in range(nlen):
		begin = i * g_batch_size
		end = begin + g_batch_size
		n_count, rslt = Calc_mean_std_and_stack(n_nums, n_count, (tuple_data_label[0])[begin:end], (tuple_data_label[1])[begin:end])
		if rslt is None:
			print('No train data has been produced.')
			return None
		lst_mean_rgb.append(rslt['mean']['mean_rgb'])
		lst_mean_opt.append(rslt['mean']['mean_opticalflow'])
		lst_std_rgb.append(rslt['std']['std_rgb'])
		lst_std_opt.append(rslt['std']['std_opticalflow'])

	arr_mean_rgb = np.mean(np.array(lst_mean_rgb), axis = 0)
	arr_mean_opt = np.mean(np.array(lst_mean_opt), axis = 0)
	arr_std_rgb = np.mean(np.array(lst_std_rgb), axis = 0)
	arr_std_opt = np.mean(np.array(lst_std_opt), axis = 0)

	with h5py.File(base_path + './mean_std.h5', 'w') as file:
		file.create_dataset('rgb_mean', data = arr_mean_rgb)
		file.create_dataset('opticalflow_mean', data = arr_mean_opt)
		file.create_dataset('rgb_std', data = arr_std_rgb)
		file.create_dataset('opticalflow_std', data = arr_std_opt)
	print("h5 prepare data file is saved.")


if __name__ == '__main__':
    main()
