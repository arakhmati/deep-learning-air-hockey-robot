#!/usr/bin/env python

import os
import glob
import random
import numpy as np
from PIL import Image

from gym_air_hockey import DataProcessor
from utils.data_utils import load_data, save_data

current_dir    = os.path.dirname(os.path.realpath(__file__))
images_dir     = current_dir + '/random_images/'
data_dir       = current_dir + '/data/'
mixed_data_dir = current_dir + '/mixed_data/'

if __name__ == "__main__":
    
    images = glob.glob(images_dir + '*.jpg')
    data_files = glob.glob(data_dir + '*.h5')
    
    if not os.path.isdir(mixed_data_dir):
        os.mkdir(mixed_data_dir)             
    
    for data_file in data_files:

        print('Mixing %s' % data_file)
        base_name = os.path.basename(data_file)
        base_name = base_name[:-3].split('_')
    
        size = int(int(base_name[-1]) * 1.1)
        mixed_data_file = mixed_data_dir + '_'.join(base_name[:-1] + [str(size)]) + '.h5'
        if os.path.exists(mixed_data_file):
            print('%s already exists and will not be overwritten' % mixed_data_file)
            continue

        if 'rgb' in data_file:
            processor = DataProcessor(mode='rgb')
        elif 'gray-diff' in data_file:
            processor = DataProcessor(mode='gray-diff')
        
        states, robot_actions, human_actions = load_data(data_file)
        n_states = states.shape[0]
        n_random = int(n_states * 0.1)

        random_states = np.zeros([n_random] + list(states.shape[1:]), dtype=np.float32)
        random_actions = np.zeros(n_random, dtype=np.int8) + 9
        
        random.shuffle(images)
        
        for idx in range(len(images[:n_random])):
            try:
                image = Image.open(images[idx])
            except:
                print('Failed on %s' % images[idx])
                os.remove(images[idx])
                continue
            image = image.convert('RGB')

            processor.process_observation(np.array(image))
            processor.process_observation(np.array(image))
            random_states[idx] = processor.process_observation(np.array(image))
        
        states = np.concatenate((states, random_states))
        robot_actions = np.concatenate((robot_actions, random_actions))
        human_actions = np.concatenate((human_actions, random_actions))
        
        permutation = np.random.permutation(states.shape[0])
        states = states[permutation]
        robot_actions = robot_actions[permutation]
        human_actions = human_actions[permutation]
            
        save_data(mixed_data_file, states, robot_actions, human_actions)
        print('Saved mixed states to %s\n' % mixed_data_file)
