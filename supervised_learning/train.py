import os
import h5py
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from gym_air_hockey import build_model

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
            
def train_network(model_file, batch_size=5000):
        
#    if os.path.exists(model_file):
#        print('Choose a different model file or delete the file with the chosen name')
#        return
    
    model = build_model()
    print(model.summary())
    
    frames, labels = load_data()
    
    for epoch in range(15):
        for i in range(frames.shape[0]//batch_size):
            print(epoch, i)
        
            frames_batch = frames[i*batch_size:(i+1)*batch_size]
            frames_batch = (frames_batch.astype(np.float32) - 0) / 256
            labels_batch_indices = labels[i*batch_size:(i+1)*batch_size]
            
            labels_batch = np.zeros((batch_size, 9), dtype=np.float32)
            for i in range(labels_batch.shape[0]):
                labels_batch[i, labels_batch_indices[i]] = 1
            
            frames_train, frames_test, labels_train, labels_test = train_test_split(frames_batch, labels_batch, test_size=0.2)
            
            model.fit(frames_train, labels_train, epochs=1, batch_size=128, validation_split=0.2)
            
            labels_pred = model.predict(frames_test, batch_size=64)
            
            print(confusion_matrix(np.argmax(labels_test, axis=1), np.argmax(labels_pred, axis=1)))
            
        model.save(model_file)
    
train_network('top_ai_model.h5')
train_network('bottom_ai_model.h5')