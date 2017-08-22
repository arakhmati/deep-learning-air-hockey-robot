import os
import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras.callbacks

from gym_air_hockey import build_model, DataProcessor


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

project_path = os.path.dirname(os.path.realpath(__file__))

def load_data(data_file=None, bottom=True):
    if data_file is None:
        data_file = project_path+'/data.h5'
    
    with h5py.File(data_file, 'r') as f:
        frames          = f['frames'][:]
        top_ai_moves    = f['top_ai_moves'][:]
        bottom_ai_moves = f['bottom_ai_moves'][:] 
        
        if bottom:
            labels = bottom_ai_moves
        else:
            labels = top_ai_moves
            
        return frames, labels
            
    
if __name__ == "__main__":
    batch_size = 10000
    model_file = 'bottom_ai_model.h5'
    n_epochs = 100
#def train_network(model_file, batch_size=5000):
        
#    if os.path.exists(model_file):
#        print('Choose a different model file or delete the file with the chosen name')
#        return
    
    processor = DataProcessor()
    
    model = build_model()
    print(model.summary())
    
    callbacks = []
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=25, verbose=1, mode='auto'))
    callbacks.append(keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True))
    
    frames, labels_indices = load_data()
    
    frames = processor.normalize_observation(frames)
    labels = np.zeros((labels_indices.shape[0], 9), dtype=np.float32)
    for i in range(labels.shape[0]):
        labels[i, labels_indices[i]] = 1
    
    frames_train, frames_test, labels_train, labels_test = train_test_split(frames, labels, test_size=0.1)
    
    model.fit(frames_train, labels_train, epochs=n_epochs, batch_size=32, 
          validation_split=0.1, verbose=1, callbacks=callbacks)
    labels_pred = model.predict(frames_test, batch_size=64)
    print(confusion_matrix(np.argmax(labels_test, axis=1), np.argmax(labels_pred, axis=1)))
    
            
    model.save(model_file)
    
#train_network('top_ai_model.h5')
#train_network('bottom_ai_model.h5')