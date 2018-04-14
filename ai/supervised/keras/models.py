import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../utils')
sys.path.append(dir_path + '/../../')

import config

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import regularizers

from metrics import fmeasure, recall, precision

def conv_model(l1=0.00000, l2=0.0001):

    models = {}

    models['rgb'] = Sequential([
                Conv2D(input_shape=(9, 128, 128),
                       name='conv1',
                       filters=5,
                       kernel_size=8,
                       strides=3,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm1', axis=1),

                Conv2D(name='conv2',
                       filters=32,
                       kernel_size=4,
                       strides=2,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm2', axis=1),

                Conv2D(name='conv3',
                       filters=64,
                       kernel_size=4,
                       strides=2,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm3', axis=1),

                Flatten(name='flatten'),

                Dense(name='dense1',
                      units=512,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2)),

                Dense(name='dense2',
                      units=512,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2)),

                Dense(name='out',
                      units=10,
                      activation='softmax',
                      kernel_regularizer=regularizers.l2(l2))
                ])

    models['gray-diff'] = Sequential([
                Conv2D(input_shape=(1, 128, 128),
                       name='conv1',
                       filters=16,
                       kernel_size=8,
                       strides=4,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm1', axis=1),

                Conv2D(name='conv2',
                       filters=24,
                       kernel_size=4,
                       strides=2,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm2', axis=1),

                Conv2D(name='conv3',
                       filters=32,
                       kernel_size=4,
                       strides=1,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm3', axis=1),

                Conv2D(name='conv4',
                       filters=48,
                       kernel_size=4,
                       strides=1,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(name='batchnorm4', axis=1),

                Flatten(name='flatten'),

                Dense(name='dense1',
                      units=512,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2)),

                Dense(name='out',
                      units=10,
                      activation='softmax',
                      kernel_regularizer=regularizers.l2(l2))
                ])

    model = models[config.mode]
    model.compile(loss='categorical_crossentropy',  optimizer='adam',
                  metrics=['accuracy', fmeasure, recall, precision])
    return model

if __name__ == "__main__":
    model = conv_model()
    model.summary()
