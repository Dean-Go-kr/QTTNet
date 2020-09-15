
import logging
import random
import numpy as np
import scipy.io
from path import Path
import argparse

#import voxnet
from voxnet import npytar
from voxnet.data import shapenet40

import h5py

arr_list = []
label=[]

def write(records, fname):
    writer = npytar.NpyTarWriter(fname)
    i = 0
    for (classname, instance, rot, fname) in records:
        i +=1
        class_id = int(shapenet40.class_name_to_id[classname])
        name = '{:03d}.{}.{:03d}'.format(class_id, instance, rot)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        arrpad = np.zeros((32,)*3, dtype=np.uint8)
        arrpad[1:-1,1:-1,1:-1] = arr
        writer.add(arrpad, name)

    writer.close()


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=Path)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')

#base_dir = Path('~/code/3DShapeNets2/3DShapeNets/volumetric_data').expand()
base_dir = (args.data_dir/'volumetric_data').expand()
#print(base_dir)

records = {'train': [], 'test': []}

logging.info('Loading .mat files')

for fname in sorted(base_dir.walkfiles('*.mat')):
    if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'):
        continue
    elts = fname.splitall()
    instance_rot = Path(elts[-1]).stripext()
    instance = instance_rot[:instance_rot.rfind('_')]
    rot = int(instance_rot[instance_rot.rfind('_')+1:])
    split = elts[-2]
    classname = elts[-4].strip()
    if classname not in shapenet40.class_names:
        continue
    #if rot <= 4:
    #    records[split].append((classname, instance, rot, fname))
    records[split].append((classname, instance, rot, fname))

    #print(fname, elts, instance_rot, instance, rot, split, classname)


# just shuffle train set
logging.info('Saving train npy tar file')
train_records = records['train']
random.shuffle(train_records)
write(train_records, 'shapenet40_train.tar')

# order test set by instance and orientation
logging.info('Saving test npy tar file')
test_records = records['test']
test_records = sorted(test_records, key=lambda x: x[2])
test_records = sorted(test_records, key=lambda x: x[1])
write(test_records, 'shapenet40_test.tar')

