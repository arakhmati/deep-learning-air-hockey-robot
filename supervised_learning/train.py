import os
import h5py
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

project_path = os.path.dirname(os.path.realpath(__file__))
data_file = project_path + '/data.h5'

batch_size = 2500

def build_model():
    image_input = Input(shape=(128, 128, 3), dtype=np.float32, name='input')
    conv1  = Conv2D(4, 10, activation='relu', name='conv1')(image_input)
    pool1  = MaxPooling2D(3, name='pool1')(conv1)
    conv2  = Conv2D(8, 10, activation='relu', name='conv2')(pool1)
    pool2  = MaxPooling2D(3, name='pool2')(conv2)
    flat   = Flatten(name='flatten')(pool2)
    dense1 = Dense(100, activation='relu', name='dense1')(flat)
    
    x0 = Dense(50, activation='relu',    name='x0')(dense1)
    x1 = Dense(3,  activation='softmax', name='x1')(x0)

    y0 = Dense(50, activation='relu',    name='y0')(dense1)
    y1 = Dense(3,  activation='softmax', name='y1')(y0)
  
    model = Model(inputs=[image_input], outputs=[x1, y1])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
model = build_model()

with h5py.File(data_file, 'r') as f:
    images = f['images'][:]
    labels = f['labels'][:]

enc = OneHotEncoder(3)
enc.fit([[0, 0], [1, 1], [2, 2]])
    
for _ in range(3):
    for i in range(images.shape[0]//batch_size):
        print(i)
    
        images_batch = ((images[i*batch_size:(i+1)*batch_size]).astype(np.float32)-128)/128
        labels_batch = labels[i*batch_size:(i+1)*batch_size] + 1
        
        images_train, images_test, labels_train, labels_test = train_test_split(images_batch, labels_batch, test_size=0.2)
        
        one_hot_encoded_labels_train = enc.transform(labels_train).toarray()
        x_train = one_hot_encoded_labels_train[:, :3]
        y_train = one_hot_encoded_labels_train[:, 3:]
        
        one_hot_encoded_labels_test = enc.transform(labels_test).toarray()
        x_test = one_hot_encoded_labels_test[:, :3]
        y_test = one_hot_encoded_labels_test[:, 3:]
        
        
        model.fit([images_train], [x_train, y_train], epochs=25, batch_size=64)
        
        x_pred, y_pred = model.predict([images_test], batch_size=64)
        
        print(confusion_matrix(np.argmax(x_test, axis=1), np.argmax(x_pred, axis=1)))
        print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

model.save('model.h5')