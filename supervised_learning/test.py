import os
import pygame
import numpy as np

from keras.models import load_model
from model import fmeasure, recall, precision
from air_hockey import AirHockey
from gym_air_hockey import DataProcessor


project_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    
    air_hockey = AirHockey()
    processor = DataProcessor()
    
    model = load_model('conv.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    
    action = None
    while True:
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        game_info = air_hockey.step(action)
        frame = processor.process_observation(game_info.frame) 
        action = processor.process_action(np.argmax(model.predict(frame.reshape(1,128,128,3))[0]))
    pygame.quit()
        