import sys
sys.path.append('../supervised/keras')

import pygame
import numpy as np

from keras.models import load_model
from metrics import fmeasure, recall, precision
from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

models_dir = './'
model_name = 'rl_model'

model_file = models_dir + model_name + '.h5'

if __name__ == "__main__":

    n_lookback = 3

    air_hockey = AirHockey()
    processor = DataProcessor()

    model = load_model(model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    def step(action=4):
        print(action)
        action = processor.process_action(action)

        game_info  = air_hockey.step(action=action)
        
        if game_info.scored:
            game_info = air_hockey.reset()
            reset()

        frame = processor.process_observation(game_info.frame)
        frame = frame.reshape((1,9,128,128))
        return frame

    def reset():
        # Fill in current_frame
        for _ in range(n_lookback):
            frame = step()
        return frame

    frame = reset()
    while True:
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        action = model.predict(frame)[0]

        action = np.asscalar(np.argmax(action))

        frame = step(action=action)
    pygame.quit()
