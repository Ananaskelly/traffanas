import numpy
import collections
import os
import glob
import cv2
from tensorflow.python.framework import dtypes
import tensorflow as tf

DIR_NAME = 'signs'
DIR = 'signs\\*\\*.ppm'


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=False):

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # shuffle
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_data_sets(fake_data=False, one_hot=False,
                        dtype=dtypes.float64, reshape=True,
                        validation_size=5000):
    num_all = sum([len(files) for r, d, files in os.walk(os.path.abspath(DIR_NAME))])
    num_training = round(num_all*0.9)
    # num_validation = 1000
    num_test = round(num_all*0.1)

    all_images = numpy.zeros((num_all, 32, 32, 3))

    for ind, file in enumerate(glob.glob(os.getcwd() + '\\' + DIR)):
        all_images[ind] = numpy.multiply(cv2.resize(cv2.imread(file), (32, 32)), 1.0 / 255.0)

    # all_images = all_images[..., numpy.newaxis]

    num_classes = sum([len(d) for r, d, f in os.walk(DIR_NAME)])
    labels_list = [numpy.full(len(f), ind, 'int32') for ind, (r, d, f) in enumerate(os.walk(os.path.abspath(DIR_NAME)))]

    labels_array = numpy.concatenate(labels_list)
    all_labels = dense_to_one_hot(labels_array, num_classes)

    perm = numpy.arange(num_all)
    numpy.random.shuffle(perm)
    all_images = all_images[perm]
    all_labels = all_labels[perm]
    mask = range(num_training)
    train_images = all_images[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_test)
    test_images = all_images[mask]
    test_labels = all_labels[mask]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)

    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    ds = collections.namedtuple('Datasets', ['train', 'test'])

    return ds(train=train, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    labels_one_hot = numpy.zeros(shape=(len(labels_dense), num_classes))
    for i in range(len(labels_dense)):
        labels_one_hot.itemset((i, labels_dense[i] - 1), 1)
    return labels_one_hot
