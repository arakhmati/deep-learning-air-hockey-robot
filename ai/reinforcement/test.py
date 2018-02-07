import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../')
sys.path.append(dir_path + '/../supervised/keras')

import config

import pygame
import numpy as np

from keras.models import load_model
from metrics import fmeasure, recall, precision

import gym
import gym_air_hockey

models_dir = dir_path + '/models/' + config.mode + '/'
robot_model_name = 'robot_model'
human_model_name = 'human_model'

robot_model_file = models_dir + robot_model_name + '.h5'
human_model_file = models_dir + human_model_name + '.h5'

if __name__ == "__main__":

    env = gym.make('AirHockey-v0')
    env.update(mode=config.mode)

    robot_model = load_model(robot_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    if config.train_human_model:
        human_model = load_model(human_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    
    state = env.reset()
    while True:
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break

        state = np.expand_dims(state, axis=0)

        robot_action = robot_model.predict(state)[0]
        robot_action = np.asscalar(np.argmax(robot_action))

        if config.train_human_model:
            human_action = human_model.predict(state)[0]
            human_action = np.asscalar(np.argmax(human_action))
        else:
            human_action = None

        state, _, terminal, game_info = env.step(robot_action=robot_action, human_action=human_action)

    pygame.quit()
