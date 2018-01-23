import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import h5py
import argparse
from keras.models import load_model

from unveiler import Model
from metrics import fmeasure, recall, precision

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', default=None, help='file with test data')
    parser.add_argument('-m', '--model_file', default=None, help='file containing keras model')
    args = parser.parse_args()
    data_file = args.data_file
    model_file = args.model_file

    keras_model = load_model(model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    with h5py.File(data_file, 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]

    model = Model(keras_model)

    start, offset = 0, 1
    for frame in frames[start:start+offset]:
        print('Feeforwarding through the network')
        model.predict(frame)

        print('Visualizing all activations')
        model.visualize(until=12, n_cols=5) # Stop on last BatchNorm

        print('Deconvolving first layer')
        model.deconvolve(index=0)