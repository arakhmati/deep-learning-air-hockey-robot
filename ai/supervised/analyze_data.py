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
    frames = np.uint8(frames * 128 + 128)
    print(frames.max(), frames.min(), frames.mean())
    
    direction = ['NW', 'W', 'SW', 'N', 'Stand', 'S', 'NE', 'E', 'SE', 'Undefined']
    
    for frame_idx in range(frames.shape[0]):
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(frames[frame_idx][0:3].transpose((1,2,0)))
        ax[1].imshow(frames[frame_idx][3:6].transpose((1,2,0)))
        ax[2].imshow(frames[frame_idx][6:9].transpose((1,2,0)))
        f.suptitle(direction[labels[frame_idx]] + ' ' + direction[adversarial_labels[frame_idx]])
        plt.show()
