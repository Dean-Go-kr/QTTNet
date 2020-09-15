import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

Train = True #False

def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if Train:

    for i in range(9):
        with h5py.File('./modelnet40_train_h5_'+str(i+1)+'.h5') as hf:
            print(hf.keys())
            X_train = hf["data"][:]
            y_train = hf["label"][:]

        print(X_train.shape, y_train.shape)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = np.float32(X_train)
        #y_train = np.int(y_train)

        tfrecords_filename = 'modelnet40_train_data_00'+str(i+1)+'.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        original_images = []


        for j in range(len(y_train)):
            image = X_train[j]
            label = y_train[j]

            height = image.shape[0]
            width = image.shape[1]
            depth = image.shape[2]
            channel = image.shape[3]

            image_raw = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'channel': _int64_feature(channel),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_raw)}))

            writer.write(example.SerializeToString())
        writer.close()


else:

    for i in range(2):
        with h5py.File('./modelnet40_test_h5_'+str(i+1)+'.h5') as hf:
            print(hf.keys())
            X_test = hf["data"][:]
            y_test = hf["label"][:]

        print(X_test.shape, y_test.shape)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_test = np.float32(X_test)
        #y_train = np.int(y_train)

        tfrecords_filename = 'modelnet40_test_data_00'+str(i+1)+'.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        original_images = []


        for j in range(len(y_test)):
            image = X_test[j]
            label = y_test[j]

            height = image.shape[0]
            width = image.shape[1]
            depth = image.shape[2]
            channel = image.shape[3]

            image_raw = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'channel': _int64_feature(channel),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_raw)}))

            writer.write(example.SerializeToString())
        writer.close()


