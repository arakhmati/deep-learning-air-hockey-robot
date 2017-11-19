import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from caffe2.python import workspace

import keras_to_caffe2

if __name__ == '__main__':

    # Create Keras Model usingSequential API
    keras_model = Sequential([
                Conv2D(5, 5, activation='relu', name='conv1', 
                       data_format='channels_first',
                       input_shape=(3, 128, 128)),
                BatchNormalization(axis=1, name='batchnorm1'),
                MaxPooling2D(2, name='pool1', data_format='channels_first'),
                Conv2D(10, 5, activation='relu', name='conv2', 
                       data_format='channels_first'),
                BatchNormalization(axis=1, name='batchnorm2'),
                MaxPooling2D(2, name='pool2', data_format='channels_first'),
                Conv2D(20, 5, activation='relu', name='conv3',  
                       data_format='channels_first'),
                BatchNormalization(axis=1, name='batchnorm3'),
                MaxPooling2D(2, name='pool3', data_format='channels_first'),
                Conv2D(30, 5, activation='relu', name='conv4',  
                       data_format='channels_first'),
                BatchNormalization(axis=1, name='batchnorm4'),
                MaxPooling2D(2, name='pool4', data_format='channels_first'),
                Dropout(0.4, name='conv4_dropout'),
                Flatten(name='flatten'),
                Dense(300, activation='relu',    name='dense1'),
                Dropout(0.3, name='dense1_dropout'),
                Dense(300, activation='relu',    name='dense2'),
                Dropout(0.3, name='dense2_dropout'),
                Dense(9,   activation='softmax', name='out')
                ])
    keras_model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    
    # Copy model from keras to caffe2
    caffe2_model = keras_to_caffe2.keras_to_caffe2(keras_model)
    
    # Generate random input data
    frame = np.random.random_sample((1, 3, 128, 128)).astype(np.float32)
    
    # Predict using Keras 
    keras_pred = keras_model.predict(frame)[0]
    
    # Predict using Caffe2 
    workspace.FeedBlob('in', frame)
    workspace.RunNet(caffe2_model.net)
    caffe2_pred = workspace.FetchBlob('softmax')[0]
    
    # Compare Predictions
    print(keras_pred.tolist())
    print(caffe2_pred.tolist())
    print('%d == %d' % (np.argmax(keras_pred), np.argmax(caffe2_pred)))
