import os
import h5py
import time
import json
import pygame
import datetime
import numpy as np
import progressbar

from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + '/data/'

if __name__ == "__main__":
    n_steps = 4
    n_lookback = 3
    n_max_frames = 1000
    
    air_hockey = AirHockey()
    processor = DataProcessor()
    
    frames = np.zeros((n_max_frames, n_lookback, processor.dim, processor.dim), dtype=np.float32)
    labels = np.zeros(n_max_frames, dtype=np.int8)
    
    current_frames = np.zeros((n_lookback, processor.dim, processor.dim), dtype=np.float32)
    
    # Fill in the current_frames
    for i in range(n_lookback):
        game_info  = air_hockey.step(n_steps=n_steps)
        current_frames[2] = np.copy(current_frames[1])
        current_frames[1] = np.copy(current_frames[0])   
        current_frames[0] = np.copy(processor.process_observation(game_info.frame))

    bar = progressbar.ProgressBar(max_value=n_max_frames)
    for i in range(n_max_frames):
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        
        frames[i] = np.copy(current_frames)
        labels[i] = processor.action_to_label(air_hockey.bottom_ai.force)
        
        game_info  = air_hockey.step(n_steps=n_steps)
        current_frames[2] = np.copy(current_frames[1])
        current_frames[1] = np.copy(current_frames[0])
        current_frames[0] = np.copy(processor.process_observation(game_info.frame))

        bar.update(i)
        if game_info.scored:
            pygame.quit()
            break
        
    frames = frames[:i]
    labels = labels[:i]
    
    def current_time():
        return datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')
    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        
    data_file = data_dir + ('%s_%d.h5' % (current_time(), i))
    with h5py.File(data_file , 'w') as f:
        f.create_dataset('frames', data=frames)
        f.create_dataset('labels', data=labels)
    
        