import argparse
import numpy as np
from utils.data_utils import load_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='Name of the data file to analyze')
    args = parser.parse_args()
    data_file = args.data_file
    frames, labels, adversarial_labels = load_data(data_file)
    
    direction = ['NW', 'W', 'SW', 'N', 'Stand', 'S', 'NE', 'E', 'SE', 'Undefined']
    
    plt.ion()
    for frame_idx in range(frames.shape[0]):
        plt.title(direction[labels[frame_idx]] + ' ' + direction[adversarial_labels[frame_idx]])
        plt.imshow(frames[frame_idx], cmap='binary')
        plt.pause(0.0001)
