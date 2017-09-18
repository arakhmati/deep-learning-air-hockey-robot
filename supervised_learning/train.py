import h5py
import time
import numpy as np

import argparse

from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

from model import conv_model
from keras.models import load_model
from model import fmeasure, recall, precision

def load_data(data_file=None, bottom=True):
    if data_file is None:
        data_file = 'data/1505438557_40000.h5'
    
    with h5py.File(data_file, 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]
        
        return frames, labels
            
def prepare_for_lstm(frames, labels, lookback=5):
    n_samples, height, width, depth = frames.shape
    new_frames = np.zeros((n_samples-lookback, lookback, height, width, depth), dtype=np.float32)
    for i in range(0, n_samples-lookback):
        for j in range(lookback):
            new_frames[i, j] = frames[i + j]
    new_labels = labels[lookback:]
    return new_frames, new_labels
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='Name of the data file with training data')
    args = parser.parse_args()
    data_file = args.data_file
    print(data_file)
    
    batch_size_train = 128
    batch_size_test  = 32
    n_epochs = 200
    
    frames, labels = load_data(data_file)
    labels = to_categorical(labels)
        
#    frames, labels = prepare_for_lstm(frames, labels)
    
#    # Shuffle Data
#    c = np.c_[frames.reshape(len(frames), -1), labels.reshape(len(labels), -1)]
#    size = frames.size
#    shape = frames.shape
#    length = len(frames)
#    frames = c[:, :size//length].reshape(shape)
#    labels = c[:, size//length:].reshape(labels.shape)
#    np.random.shuffle(c)
         
    model = load_model('model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    print(model.summary())
    
    callbacks = []
    callbacks.append(TensorBoard(log_dir=str(time.time()), histogram_freq=1, write_graph=True, write_images=True))
    
    model.fit(frames, labels, 
              epochs=n_epochs, 
              batch_size=batch_size_train,
              callbacks=callbacks,
              shuffle=True,
              verbose=1)
    
    predictions = model.predict(frames, batch_size=batch_size_test)
    print(confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1)))
    
    model.save('model.h5')