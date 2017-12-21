import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(dir_path + '/../utils')

import time
from datetime import datetime
import numpy as np

import argparse

from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

from model import conv_model
from keras.models import load_model
from model import fmeasure, recall, precision
from data_utils import load_data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='Name of the file with training data', required=True)
    args = parser.parse_args()
    data_file = args.data_file
    
    batch_size_train = 128
    batch_size_test  = 128
    n_epochs = 20
    
    frames, labels, _ = load_data(data_file)
    labels = to_categorical(labels, num_classes=10)

    if os.path.exists('models/model.h5'):
        print('Model already exists. Loading...')
        model = load_model('models/model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    else:
        print('Creating new model')
        model = conv_model()
    model.summary() 
    
    def current_time():
        return datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d_%H%M%S')
    
    callbacks = []
    callbacks.append(TensorBoard(log_dir='tensorboard/' + current_time() + ' ' + data_file[:3], 
                                 histogram_freq=1, write_graph=True, write_images=True))
    
    history = model.fit(frames, labels, 
              epochs=n_epochs, 
              batch_size=batch_size_train,
              callbacks=callbacks,
              # validation_split=0.2,
              shuffle=True,
              verbose=1)
    
    predictions = model.predict(frames, batch_size=batch_size_test)
    print(confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1)))
    print(classification_report(np.argmax(labels, axis=1), np.argmax(predictions, axis=1)))
    
    accuracy = np.count_nonzero(np.argmax(labels, axis=1) == np.argmax(predictions, axis=1)) / labels.shape[0]
    
    model.save('models/model.h5')
    model.save('models/model_%s_%02d.h5' % (current_time(), int(accuracy * 100)))