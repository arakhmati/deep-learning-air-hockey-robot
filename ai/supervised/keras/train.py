import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(dir_path + '/../utils')

import time
from datetime import datetime
import numpy as np

import argparse

from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

from data_utils import load_data
from models import conv_model, fmeasure, recall, precision

models_dir = 'models/'
model_name = 'model'
adversarial_model_name = 'adv_model'

model_file = models_dir + model_name + '.h5'
adversarial_model_file = models_dir + adversarial_model_name + '.h5'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='Name of the file with training data', required=True)
    parser.add_argument('-e', '--n_epochs', help='Number of epochs', default=100)
    args = parser.parse_args()
    data_file = args.data_file
    n_epochs = args.n_epochs

    batch_size_train = 128
    batch_size_test  = 128

    frames, labels, adversarial_labels = load_data(data_file)
    labels = to_categorical(labels, num_classes=10)
    adversarial_labels = to_categorical(adversarial_labels, num_classes=10)

    # Robot model
    if os.path.exists(model_file):
        print('Model already exists. Loading.')
        model = load_model(model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    else:
        print('Creating new model.')
        model = conv_model()

    # Human model
    if os.path.exists(adversarial_model_file):
        print('Adversarial model already exists. Loading.')
        adversarial_model = load_model(adversarial_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    elif os.path.exists(model_file):
        print('Model already exists. Loading it as adversarial model.')
        adversarial_model = load_model(model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    else:
        print('Creating new adversarial model.')
        adversarial_model = conv_model()

    for _model, _labels in zip([model, adversarial_model],
                 [labels, adversarial_labels]):

        history = _model.fit(frames, _labels,
                  epochs=n_epochs,
                  batch_size=batch_size_train,
                  shuffle=True,
                  verbose=1)

        logits = _model.predict(frames, batch_size=batch_size_test)
        y = np.argmax(_labels, axis=1)
        p = np.argmax(logits,  axis=1)
        print(confusion_matrix(y, p))
        print(classification_report(y, p))

    model.save(model_file)
    adversarial_model.save(adversarial_model_file)