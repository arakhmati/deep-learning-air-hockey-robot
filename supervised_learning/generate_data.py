import os
import cv2
import h5py
import pygame
import numpy as np
import progressbar

from air_hockey import AirHockey

project_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    screen_mode = 0
    video_output = 0
    number_of_frames = 100000
    
    new_width  = 128
    new_height = 128
    
    images = np.zeros((number_of_frames, new_height, new_width, 3), dtype=np.uint8)
    top_ai_labels    = np.zeros((number_of_frames, 2), dtype=np.int8)
    bottom_ai_labels = np.zeros((number_of_frames, 2), dtype=np.int8)
    
    air_hockey = AirHockey()

    bar = progressbar.ProgressBar(max_value=number_of_frames)
    for i in range(number_of_frames):
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        image = air_hockey.step()
        image =  image[:, air_hockey.dim.vertical_margin:-air_hockey.dim.vertical_margin, :].transpose((1,0,2))
        images[i] = cv2.resize(image, (new_height, new_width)).reshape((1, new_height, new_width, 3))
        top_ai_labels[i]    = air_hockey.top_ai.force
        bottom_ai_labels[i] = air_hockey.bottom_ai.force
        bar.update(i)
        if i % 2500 == 0:
            air_hockey.reset()
    
    with h5py.File(project_path + '/data.h5', 'w') as f:
        f.create_dataset('images',            data=images)
        f.create_dataset('top_ai_labels',     data=top_ai_labels)
        f.create_dataset('bottom_ai_labels',  data=bottom_ai_labels)
    
    pygame.quit()
        