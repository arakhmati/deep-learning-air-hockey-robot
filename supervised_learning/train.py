import os
import h5py
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils.np_utils import to_categorical

from model import conv_model, convlstm_model


project_path = os.path.dirname(os.path.realpath(__file__))

def load_data(data_file=None, bottom=True):
    if data_file is None:
        data_file = project_path+'/data_300.h5'
    
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
    batch_size_train = 16
    batch_size_test  = 32
    n_epochs = 1000
    
    frames, labels = load_data()
    labels = to_categorical(labels)
        
    frames, labels = prepare_for_lstm(frames, labels)
    
    frames_train, frames_test, labels_train, labels_test = train_test_split(frames, labels, test_size=0.1)
    
#    c = np.c_[frames_train.reshape(len(frames_train), -1), labels_train.reshape(len(labels_train), -1)]
#    size = frames_train.size
#    shape = frames_train.shape
#    length = len(frames_train)
#    frames_train = c[:, :size//length].reshape(shape)
#    labels_train = c[:, size//length:].reshape(labels_train.shape)
#    np.random.shuffle(c)
         
    model = convlstm_model()
    
#    from keras.models import load_model
#    from model import fmeasure, recall, precision
#    model = load_model('bottom_ai_model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    print(model.summary())
    
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=500, verbose=1, mode='auto'))
    callbacks.append(TensorBoard(log_dir=str(time.time()), histogram_freq=1, write_graph=True, write_images=True))
    
    model.fit(frames_train, labels_train, 
              epochs=n_epochs, 
              batch_size=batch_size_train,
              validation_split=0.1, 
              callbacks=callbacks,
              shuffle=True,
              verbose=1)
    
    labels_pred = model.predict(frames_test, batch_size=batch_size_test)
    
    print(confusion_matrix(np.argmax(labels_test, axis=1), np.argmax(labels_pred, axis=1)))
    
    model.save('model.h5')
    
#train_network('top_ai_model.h5')
#train_network('bottom_ai_model.h5')