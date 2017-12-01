import os
import pygame
import numpy as np

from keras.models import load_model
from model import fmeasure, recall, precision
from air_hockey import AirHockey
from gym_air_hockey import DataProcessor


project_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    
    dt = 2
    n_lookback = 3
    
    air_hockey = AirHockey()
    processor = DataProcessor()
    
    model = load_model('models/model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    
    def step(action=4):
        action = processor.process_action(action)
        game_info  = air_hockey.step(action=action, dt=dt)
        frame = processor.process_observation(game_info.frame)
        return frame
        
    def reset():
        # Fill in current_frame
        for _ in range(n_lookback):
            frame = step()
        return frame
    
    frame = reset()
    while True:
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        action = model.predict(frame.reshape((1,9,128,128)))[0]
        action = np.asscalar(np.argmax(action))        
        frame = step(action)
    pygame.quit()
        