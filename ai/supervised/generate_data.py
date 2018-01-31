import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../')

import config

import os
import h5py
import time
import json
import pygame
import argparse
import datetime
import numpy as np
import progressbar
from collections import deque

import gym
import gym_air_hockey

from utils.data_utils import save_data

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + '/data/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_states', type=int, default=2000, help='number of states to be saved')

    args = parser.parse_args()
    n_states = args.n_states
    mode     = config.mode

    env = gym.make('AirHockey-v0')
    env.update(mode=mode)

    states        = deque(maxlen=n_states)
    robot_actions = deque(maxlen=n_states)
    human_actions = deque(maxlen=n_states)

    reset = True
    bar = progressbar.ProgressBar(max_value=n_states)
    for i in range(n_states):
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break

        if reset:
            reset = False
            env.reset()

        state, _, terminal, game_info = env.step()

        states.append(state)
        robot_actions.append(game_info['robot_action'])
        human_actions.append(game_info['human_action'])

        if terminal:
            reset = True
        
        bar.update(i)

    def current_time():
        return datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    states = np.array(states, dtype=np.float32)
    robot_actions = np.array(robot_actions, dtype=np.int8)
    human_actions = np.array(human_actions, dtype=np.int8)

    data_file = data_dir + ('%s_%s_%d.h5' % (current_time(), mode, n_states))
    save_data(data_file, states, robot_actions, human_actions)
    print('Saved generated states to %s' % data_file)

