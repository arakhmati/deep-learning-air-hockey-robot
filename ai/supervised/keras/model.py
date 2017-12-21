from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import regularizers

from metrics import fmeasure, recall, precision

def conv_model(l1=0.00000, l2=0.0001):
    model = Sequential([
                Conv2D(20, 5, activation='relu', name='conv1', 
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2),
                       input_shape=(9, 128, 128)),
                BatchNormalization(axis=1, name='batchnorm1'),
                MaxPooling2D(2, name='pool1', data_format='channels_first'),
                Conv2D(30, 5, activation='relu', name='conv2', 
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(axis=1, name='batchnorm2'),
                MaxPooling2D(2, name='pool2', data_format='channels_first'),
                Conv2D(40, 5, activation='relu', name='conv3',  
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(axis=1, name='batchnorm3'),
                MaxPooling2D(2, name='pool3', data_format='channels_first'),
                Conv2D(50, 5, activation='relu', name='conv4',  
                       data_format='channels_first', kernel_regularizer=regularizers.l2(l2)),
                BatchNormalization(axis=1, name='batchnorm4'),
                Flatten(name='flatten'),
                Dense(25, activation='relu', name='dense1', kernel_regularizer=regularizers.l2(l2)),
                Dense(25, activation='relu', name='dense2', kernel_regularizer=regularizers.l2(l2)),
                Dense(10,   activation='softmax', name='out', kernel_regularizer=regularizers.l2(l2))
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', 
                  metrics=['accuracy', fmeasure, recall, precision])
    return model



def convlstm_model(l1=0.00000, l2=0.00000):
    model = Sequential([
                ConvLSTM2D(8, 12, activation='relu', name='conv1', input_shape=(None, 128, 128, 3)),
                MaxPooling2D(2, name='pool1'),
                Conv2D(24, 7, activation='relu', name='conv2'),
                MaxPooling2D(2, name='pool2'),
                Conv2D(32, 5, activation='relu', name='conv3'),
                MaxPooling2D(2, name='pool3'),
                Flatten(name='flatten'),
                Dense(512, activation='relu',    name='dense1'),
                Dense(256, activation='relu',    name='dense2'),
                Dense(128, activation='relu',    name='dense3'),
                Dense(64,  activation='relu',    name='dense4'),
                Dense(9,   activation='softmax', name='out')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', 
                  metrics=['accuracy', fmeasure, recall, precision])
    return model