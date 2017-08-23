import os
import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras.callbacks

from gym_air_hockey import build_model, DataProcessor


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
#    batch_size = 10000
    model_file = 'bottom_ai_model.h5'
    n_epochs = 200
#def train_network(model_file, batch_size=5000, n_epochs=100):
        
#    if os.path.exists(model_file):
#        print('Choose a different model file or delete the file with the chosen name')
#        return
    
    processor = DataProcessor()
    
    model = build_model()
    print(model.summary())
    
    
    callbacks = []
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=1, mode='auto'))
    callbacks.append(keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True))
    
    frames, labels_indices = load_data()
    
    frames = processor.normalize_observation(frames)
    labels = np.zeros((labels_indices.shape[0], 9), dtype=np.float32)
    for i in range(labels.shape[0]):
        labels[i, labels_indices[i]] = 1
    
#    frames_train, frames_test, labels_train, labels_test = train_test_split(frames, labels, test_size=0.1)
#    
#    c = numpy.c_[frames_train.reshape(len(frames_train), -1), labels_train.reshape(len(labels_train), -1)]
#    size = frames_train.size
#    shape = frames_train.shape
#    length = len(frames_train)
#    frames_train = c[:, :size//length].reshape(shape)
#    labels_train = c[:, size//length:].reshape(labels_train.shape)
#    np.random.shuffle(c)
        
    def prepare_for_lstm(frames, labels, lookback=10):
        n_samples, height, width, depth = frames.shape
        new_frames = np.zeros((n_samples-lookback, lookback, height, width, depth), dtype=np.float32)
        for i in range(0, n_samples-lookback):
            for j in range(lookback):
                new_frames[i, j] = frames[i + j]
                
        new_labels = labels[lookback:]
        return new_frames, new_labels
    frames, labels = prepare_for_lstm(frames, labels)
        
        
    model.fit(frames, labels, epochs=n_epochs, batch_size=16, 
          validation_split=0.1, verbose=1, callbacks=callbacks)
    labels_pred = model.predict(frames, batch_size=16)
    print(confusion_matrix(np.argmax(labels, axis=1), np.argmax(labels_pred, axis=1)))
    
    model.save(model_file)
    
#train_network('top_ai_model.h5')
#train_network('bottom_ai_model.h5')