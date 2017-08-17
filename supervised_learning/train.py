import os
import h5py
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

project_path = os.path.dirname(os.path.realpath(__file__))

def build_model():
    image_input = Input(shape=(128, 128, 3), dtype=np.float32, name='input')
    conv1  = Conv2D(8, 16, activation='relu', name='conv1')(image_input)
    pool1  = MaxPooling2D(3, name='pool1')(conv1)
    conv2  = Conv2D(16, 16, activation='relu', name='conv2')(pool1)
    pool2  = MaxPooling2D(3, name='pool2')(conv2)
    flat   = Flatten(name='flatten')(pool2)
    dense1 = Dense(100, activation='relu', name='dense1')(flat)
    output = Dense(9, activation='softmax', name='output')(dense1)
  
    model = Model(inputs=[image_input], outputs=output)
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model

def load_data(data_file=project_path+'/data.h5'):
    with h5py.File(data_file, 'r') as f:
        images = f['images'][:]
        top_ai_labels = f['top_ai_labels'][:]
        bottom_ai_labels = f['bottom_ai_labels'][:] 
        
        if 'top' in data_file:
            return images, top_ai_labels
        elif 'bottom' in data_file:
            return images, bottom_ai_labels
            

def train_network(model_file, batch_size = 5000):
    
    if os.path.exists(model_file):
        print('Choose a different model file or delete the file with the chosen name')
        return
    
    model = build_model()
    print(model.summary())
    
    enc = OneHotEncoder(9)
    enc = enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
    enc.transform([[0], [1], [2], [3], [4], [5], [6], [7], [8]]).toarray()
    
    images, labels = load_data()
    
    for epoch in range(15):
        for i in range(images.shape[0]//batch_size):
            print(epoch, i)
        
            images_batch = ((images[i*batch_size:(i+1)*batch_size]).astype(np.float32)-128)/128
            labels_batch =  enc.transform(labels[i*batch_size:(i+1)*batch_size] + 1)
            
            images_train, images_test, labels_train, labels_test = train_test_split(images_batch, labels_batch, test_size=0.2)
            
            model.fit([images_train], [labels_train], epochs=1, batch_size=128, validation_split=0.2)
            
            labels_pred = model.predict([images_test], batch_size=64)
            
            print(confusion_matrix(np.argmax(labels_test, axis=1), np.argmax(labels_pred, axis=1)))
            
        model.save(model_file)
    
train_network('top_ai_model.h5')
train_network('bottom_ai_model.h5')