from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import regularizers

from metrics import fmeasure, recall, precision

def conv_model(l1=0.00000, l2=0.0001):
    model = Sequential([
                Conv2D(input_shape=(9, 128, 128),
                       name='conv1',
                       filters=16,
                       kernel_size=8,
                       strides=4,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                Dropout(name='conv1_dropout',
                        rate=0.2),
                BatchNormalization(name='batchnorm1', axis=1),

                Conv2D(name='conv2',
                       filters=32,
                       kernel_size=4,
                       strides=2,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                Dropout(name='conv2_dropout',
                        rate=0.2),
                BatchNormalization(name='batchnorm2', axis=1),

                Conv2D(name='conv3',
                       filters=48,
                       kernel_size=4,
                       strides=1,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                Dropout(name='conv3_dropout',
                        rate=0.2),
                BatchNormalization(name='batchnorm3', axis=1),

                Conv2D(name='conv4',
                       filters=64,
                       kernel_size=4,
                       strides=1,
                       activation='relu',
                       data_format='channels_first',
                       kernel_regularizer=regularizers.l2(l2)),
                Dropout(name='conv4_dropout',
                        rate=0.2),
                BatchNormalization(name='batchnorm4', axis=1),

                Flatten(name='flatten'),

                Dense(name='dense1',
                      units=1024,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2)),
                Dropout(name='dense1_dropout',
                        rate=0.3),

                Dense(name='dense2',
                      units=512,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2)),
                Dropout(name='dense2_dropout',
                        rate=0.3),

                Dense(name='out',
                      units=10,
                      activation='softmax',
                      kernel_regularizer=regularizers.l2(l2))
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

if __name__ == "__main__":
    conv_model().summary()