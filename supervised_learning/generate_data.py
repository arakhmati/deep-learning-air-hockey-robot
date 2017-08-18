import os
import h5py
import pygame
import numpy as np
import progressbar

from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

project_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    number_of_frames = 10000
    
    air_hockey = AirHockey()
    processor = DataProcessor()
    
    frames = np.zeros((number_of_frames, processor.dim, processor.dim, 3), dtype=np.float32)
    top_ai_moves    = np.zeros(number_of_frames, dtype=np.int8)
    bottom_ai_moves = np.zeros(number_of_frames, dtype=np.int8)

    bar = progressbar.ProgressBar(max_value=number_of_frames)
    for i in range(number_of_frames):
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        frame, _ = air_hockey.step()
        frames[i]           = processor.resize_observation(frame)
        top_ai_moves[i]    = processor.action_to_label(air_hockey.top_ai.force)
        bottom_ai_moves[i] = processor.action_to_label(air_hockey.bottom_ai.force)
        bar.update(i)
        if i % 2500 == 0:
            air_hockey.reset()
    
    with h5py.File(project_path + '/data.h5', 'w') as f:
        f.create_dataset('frames',           data=frames)
        f.create_dataset('top_ai_moves',     data=top_ai_moves)
        f.create_dataset('bottom_ai_moves',  data=bottom_ai_moves)
    
    pygame.quit()
        