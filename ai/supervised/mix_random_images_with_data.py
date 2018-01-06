#!/usr/bin/env python

import os
import glob
import random
import numpy as np
from PIL import Image

from gym_air_hockey import DataProcessor
from utils.data_utils import load_data, save_data

current_dir = os.path.dirname(os.path.realpath(__file__))
images_dir = current_dir + '/random_images/'
data_dir = current_dir + '/data/'
mixed_data_dir = current_dir + '/mixed_data/'

if __name__ == "__main__":

    lookback = 3
    processor = DataProcessor()
    
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
        
        frames, labels, adversarial_labels = load_data(data_file)
        n_frames = frames.shape[0]
        n_random = int(n_frames * 0.1)
        
        random_frames = np.zeros((n_random, lookback * 3, processor.dim, processor.dim), dtype=np.float32)
        random_labels = np.zeros(n_random, dtype=np.int8) + 9
        
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
            random_frames[idx] = processor.process_observation(np.array(image))
        
        frames = np.concatenate((frames, random_frames))
        labels = np.concatenate((labels, random_labels))
        adversarial_labels = np.concatenate((adversarial_labels, random_labels))
        
        permutation = np.random.permutation(frames.shape[0])
        frames = frames[permutation]
        labels = labels[permutation]
        adversarial_labels = adversarial_labels[permutation]
            
        save_data(mixed_data_file, frames, labels, adversarial_labels)
        print('Saved mixed frames to %s\n' % mixed_data_file)
