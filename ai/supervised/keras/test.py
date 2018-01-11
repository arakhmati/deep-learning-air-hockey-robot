import pygame
import numpy as np

from keras.models import load_model
from metrics import fmeasure, recall, precision
from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

models_dir = 'models/'
model_name = 'model'
adversarial_model_name = 'adv_model'

model_file = models_dir + model_name + '.h5'
adversarial_model_file = models_dir + adversarial_model_name + '.h5'

if __name__ == "__main__":

    n_lookback = 3

    air_hockey = AirHockey()
    processor = DataProcessor()

    adversarial_model = load_model(adversarial_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    model = load_model(model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    def step(action=4, adversarial_action=4):
        action = processor.process_action(action)
        adversarial_action = processor.process_action(adversarial_action)

        game_info  = air_hockey.step(action=action, adversarial_action=adversarial_action)

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
        adversarial_action = adversarial_model.predict(frame)[0]

        action = np.asscalar(np.argmax(action))
        adversarial_action = np.asscalar(np.argmax(adversarial_action))

        frame = step(action=action, adversarial_action=adversarial_action)
    pygame.quit()
