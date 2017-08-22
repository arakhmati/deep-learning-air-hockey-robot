import os
import pygame
import numpy as np

from keras.models import load_model
from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

project_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    
    air_hockey = AirHockey()
    processor = DataProcessor()
    
    model = load_model('bottom_ai_model.h5')
    
    action = [0, 0]
    while True:

        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        frame, _           = air_hockey.step(action)
        frame = processor.process_observation(frame).reshape((1,128,128,3))
        action = np.argmax(model.predict(frame)[0])
        print(action)
        
    
    pygame.quit()
        