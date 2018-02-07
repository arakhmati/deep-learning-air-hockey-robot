import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import h5py
import argparse
import numpy as np
from keras.models import load_model

from unveiler import Model
from metrics import fmeasure, recall, precision

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', type=str, required=True, help='file with test data')
    parser.add_argument('-m', '--model_file', type=str, required=True, help='file containing keras model')
    args = parser.parse_args()
    data_file = args.data_file
    model_file = args.model_file

    keras_model = load_model(model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    with h5py.File(data_file, 'r') as f:
        states = f['states'][:]
        robot_actions = f['robot_actions'][:]

    model = Model(keras_model)

    indices = np.random.permutation(states.shape[0])
    for index in indices:
        state = states[index]

        print('Feeforwarding through the network')
        model.predict(state)

        print('Visualizing all activations')
        model.visualize(until=12, n_cols=4) # Stop on last BatchNorm

        print('Deconvolving first layer')
        model.deconvolve(index=0)

        print('Deconvolving second layer')
        model.deconvolve(index=1)

        print('Deconvolving third layer')
        model.deconvolve(index=2)

        print('Deconvolving fourth layer')
        model.deconvolve(index=3)
