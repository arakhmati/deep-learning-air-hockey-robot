import argparse
import numpy as np
from utils.data_utils import load_data
import matplotlib.pyplot as plt

direction = ['NW', 'W', 'SW', 'N', 'Stand', 'S', 'NE', 'E', 'SE', 'Undefined']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='data file to analyze')
    args = parser.parse_args()
    data_file = args.data_file
    states, robot_actions, human_actions = load_data(data_file)

    if 'rgb' in data_file:
        states = np.uint8(states * 128 + 128)
    
    for i in range(states.shape[0]):

        if 'rgb' in data_file:
            f, ax = plt.subplots(1, 3)
            state = states[i].transpose((1,2,0))
            ax[0].imshow(state[:,:,0:3])
            ax[1].imshow(state[:,:,3:6])
            ax[2].imshow(state[:,:,6:9])
            f.suptitle(direction[robot_actions[i]] + ' ' + direction[human_actions[i]])
        elif 'gray-diff' in data_file:
            state = states[i][0]
            plt.imshow(state, cmap='binary')
            plt.title(direction[robot_actions[i]] + ' ' + direction[human_actions[i]])
        plt.show()
