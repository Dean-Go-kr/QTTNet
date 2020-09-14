import numpy as np
import h5py

base_path = '/data/dong/ucf11/dingheng/'

len_class = 11
test_index = []


basketball = 139 # 27 28 28 28 28
biking = 145 # 29 29 29 29 29
diving = 156 # 31 31 31 31 32
golf_swing = 142 # 28 28 28 29 29
horse_riding = 198 # 39 39 40 40 40
soccer_juggling = 156 # 31 31 31 31 32
swing = 137 # 27 27 27 28 28
tennis_swing = 167 # 33 33 33 34 34
trampoline_jumping = 119 # 23 24 24 24 24 
volleyball_spiking = 116 # 23 23 23 23 24
walking = 123 # 24 24 25 25 25





one_class_index = np.arange(walking)

np.random.shuffle(one_class_index)

a = one_class_index[0 : 24]
b = one_class_index[24 : 24+24]
c = one_class_index[24+24 : 24+24+25]
d = one_class_index[24+24+25 : 24+24+25+25]
e = one_class_index[24+24+25+25 : 24+24+25+25+25]



with h5py.File('/data/dong/ucf11/dingheng/fold_index/walking_fold_index.h5', 'w') as file:
    file.create_dataset('fold_1', data = a)
    file.create_dataset('fold_2', data = b)
    file.create_dataset('fold_3', data = c)
    file.create_dataset('fold_4', data = d)
    file.create_dataset('fold_5', data = e)
'''

with h5py.File('/data/dong/ucf11/dingheng/fold_index.h5', 'r') as file:
    fold_index = file.get('fold_index').value

print(fold_index.shape)

print(fold_index[0][0:29])


'''

