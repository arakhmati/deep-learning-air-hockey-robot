import os, sys
import argparse
from utils.data_utils import load_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='Name of the data file to analyze')
    args = parser.parse_args()
    data_file = args.data_file
    frames, labels = load_data(data_file)
    print(frames.max(), frames.min(), frames.mean())
    
    direction = ['NW', 'W', 'SW', 'N', '', 'S', 'NE', 'E', 'SE']
    
    for frame_idx in range(frames.shape[0]):
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(frames[frame_idx][0:3].transpose((1,2,0)))
        ax[1].imshow(frames[frame_idx][3:6].transpose((1,2,0)))
        ax[2].imshow(frames[frame_idx][6:9].transpose((1,2,0)))
        f.suptitle(direction[labels[frame_idx]])
        plt.show()
