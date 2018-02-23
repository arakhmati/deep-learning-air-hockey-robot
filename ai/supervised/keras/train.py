import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../utils')
sys.path.append(dir_path + '/../../')

import config

import time
from datetime import datetime
import numpy as np

import argparse

from data_utils import load_data

models_dir = dir_path + '/models/' + config.mode + '/'
robot_model_name = 'robot_model'
human_model_name = 'human_model'

robot_model_file = models_dir + robot_model_name + '.h5'
human_model_file = models_dir + human_model_name + '.h5'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file',  type=str, required=True, help='file with training data')
    parser.add_argument('-e', '--n_epochs',   type=int, default=15,    help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=128,   help='batch size')
    args = parser.parse_args()
    data_file   = args.data_file
    n_epochs    = args.n_epochs
    batch_size  = args.batch_size

    # Exit if the file was generated using a different mode
    if config.mode not in data_file:
        quit()

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # These imports might be slow, so do them after the check above
    from keras.models import load_model
    from keras.callbacks import TensorBoard
    from keras.utils.np_utils import to_categorical
    from sklearn.metrics import confusion_matrix, classification_report
    from models import conv_model
    from metrics import fmeasure, recall, precision

    states, robot_actions, human_actions = load_data(data_file)
    robot_actions = to_categorical(robot_actions, num_classes=10)
    human_actions = to_categorical(human_actions, num_classes=10)

    # Robot model
    if os.path.exists(robot_model_file):
        print('Robot model already exists. Loading.')
        robot_model = load_model(robot_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    else:
        print('Creating new model.')
        robot_model = conv_model()

    model_list  =  [robot_model]
    actions_list = [robot_actions]

    # Human model
    if config.train_human_model:
        if os.path.exists(human_model_file):
            print('Human model already exists. Loading.')
            human_model = load_model(human_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
        elif os.path.exists(robot_model_file):
            print('Robot model already exists. Loading it as adversarial model.')
            human_model = load_model(robot_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
        else:
            print('Creating new adversarial model.')
            human_model = conv_model()
        model_list  += [robot_model]
        actions_list += [robot_actions]

    for model, actions in zip(model_list, actions_list):

        history = model.fit(states, actions,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             verbose=1)

        logits = model.predict(states, batch_size=batch_size)
        y = np.argmax(actions, axis=1)
        p = np.argmax(logits,  axis=1)
        print(confusion_matrix(y, p))
        print(classification_report(y, p))

    robot_model.save(robot_model_file)
    if config.train_human_model:
        human_model.save(human_model_file)