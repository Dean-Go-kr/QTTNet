from voxnet import npytar
import numpy as np
import h5py
#import tensorflow as tf

num_train = 118116
num_test = 29616

num_train_divide = 3281 #3281 * 3 = 9843
num_test_divide = 2468 #2468 * 12 = 29616

yc = []
data_list = []

Train = True #False

# Augmentation
def jitter_chunk(src):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :] = dst

    max_ij = 2
    max_k = 2
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    print(shift_ijk)

    for axis, shift in enumerate(shift_ijk):
        #print(axis, shift)
        
        if shift !=0:
            dst = np.roll(dst, shift, axis+1)

    return dst

def tar_reader(reader, xc, Train):
    if Train:
        num_divide = num_train_divide
    else:
        num_divide = num_test_divide
    print('tar_reader divide by:', num_divide)
    for ix ,(x, name) in enumerate(reader):
        cix = ix % num_divide
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)

        if ix % num_divide == 0:
            xc = jitter_chunk(xc)
            xc = 2.0 * xc - 1.0
            xc = np.array(xc)
            xc = np.reshape(xc, (-1,32,32,32))
            data_list.append(xc)
            xc.fill(0)
    return data_list, yc


if Train:
    # Read all dataset
    reader = npytar.NpyTarReader('shapenet40_train.tar')
    # Set subset for augmentation
    xc = np.zeros((num_train_divide, 32, 32, 32))
    # After augmentation
    # data_list & yc includes whole dataset
    data_list, yc = tar_reader(reader, xc, Train)
    yc = np.array(yc)

    print('Start Saving!')
    num_h5 = 9
    len_train = int(num_train/num_train_divide) # 36
    data_list_divide = int(len_train/num_h5) # 3
    label_divide = int(num_train / num_h5) # 9843

    for i in range(num_h5):
        final_xc = np.stack(data_list[data_list_divide*i:data_list_divide*(i+1)])
        print(final_xc.shape)

        xc = np.reshape(final_xc, (-1, 32, 32, 32, 1))
        divided_yc = np.stack(yc[label_divide*i:label_divide*(i+1)])
        #yc = np.asarray(yc, dtype=np.float32)

        print(xc.shape, divided_yc.shape)
        with h5py.File('./modelnet40_train_h5_' + str(i+1) + '.h5', 'w') as hf:
            hf.create_dataset('data', data=xc, compression = 'gzip', compression_opts = 9),
            hf.create_dataset('label', data=divided_yc)
        print('SAVED')



else:
    # Read all dataset
    reader = npytar.NpyTarReader('shapenet40_test.tar')
    # Set subset for augmentation
    xc = np.zeros((num_test_divide, 32, 32, 32))
    # After augmentation
    # data_list & yc includes whole dataset
    data_list, yc = tar_reader(reader, xc, Train)
    yc = np.array(yc)

    print('label shape: ', yc.shape)
    num_h5 = 2
    len_test = int(num_test / num_test_divide)  # 29616 / 2468 = 12
    data_list_divide = int(len_test / num_h5)  # 12
    label_divide = int(num_test / num_h5)  # 29616

    for i in range(num_h5):
        final_xc = np.stack(data_list[data_list_divide*i:data_list_divide*(i+1)])
        xc = np.reshape(final_xc, (-1, 32, 32, 32, 1))

        divided_yc = np.stack(yc[label_divide*i:label_divide*(i+1)])
        #yc = np.asarray(yc, dtype=np.float32)

        print(xc.shape, divided_yc.shape)
        with h5py.File('./modelnet40_test_h5_' + str(i+1) + '.h5', 'w') as hf:
            hf.create_dataset('data', data=xc),
            hf.create_dataset('label', data=divided_yc)
