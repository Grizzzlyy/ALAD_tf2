import tensorflow as tf
import keras
from keras.initializers import GlorotNormal
from keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, Flatten, Layer
# from tensorflow.keras.layers import SpectralNormalization
# from tensorflow.train import ExponentialMovingAverage
import numpy as np
from sklearn import metrics
import time

import data.kdd99.kdd99 as kdd

def batch_fill(testx, batch_size):
    """ Quick and dirty hack for filling smaller batch

    :param testx:
    :param batch_size:
    :return:
    """
    nr_batches_test = int(testx.shape[0] / batch_size)
    ran_from = nr_batches_test * batch_size
    ran_to = (nr_batches_test + 1) * batch_size
    size = testx[ran_from:ran_to].shape[0]
    new_shape = [batch_size - size] + list(testx.shape[1:])
    fill = np.ones(new_shape).astype(np.float32)
    return np.concatenate([testx[ran_from:ran_to], fill], axis=0), size