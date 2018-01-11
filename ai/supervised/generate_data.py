import os
import h5py
import time
import json
import pygame
import argparse
import datetime
import numpy as np
import progressbar

from air_hockey import AirHockey
from gym_air_hockey import DataProcessor
from utils.data_utils import save_data

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + '/data/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lookback', type=int, default=3,
                        help='Number of frames to look in the past')
    parser.add_argument('-n', '--n_frames', type=int, default=5000,
                        help='Number of frames to be saved')
    args = parser.parse_args()

    lookback = args.lookback
    n_frames = args.n_frames
    print('lookback %d, n_frames %d' % (lookback, n_frames))

    air_hockey = AirHockey()
    processor = DataProcessor()

    frames = np.zeros((n_frames, lookback * 3, processor.dim, processor.dim), dtype=np.float32)
    labels = np.zeros(n_frames, dtype=np.int8)
    adversarial_labels = np.zeros(n_frames, dtype=np.int8)

    current_frame = np.zeros((lookback * 3, processor.dim, processor.dim), dtype=np.float32)

    def step():
        game_info  = air_hockey.step()
        frame = processor.process_observation(game_info.frame)
        action = processor.action_to_label(game_info.action)
        adversarial_action = processor.action_to_label(game_info.adversarial_action)
        return {'frame': frame,
                'action': action,
                'adversarial_action': adversarial_action,
                'scored': game_info.scored}

    def reset():
        # Fill in current_frame
        for _ in range(lookback):
            game_info = step()
        return game_info

    game_info = reset()
    bar = progressbar.ProgressBar(max_value=n_frames)
    for i in range(n_frames):
        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break

        frames[i] = game_info['frame']
        labels[i] = game_info['action']
        adversarial_labels[i] = game_info['adversarial_action']

        game_info = step()
        bar.update(i)

        if game_info['scored']:
            air_hockey.reset()
            game_info = reset()

    def current_time():
        return datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d-%H-%M-%S')

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    print(labels, adversarial_labels)

    data_file = data_dir + ('%s_%d_%d.h5' % (current_time(), lookback, n_frames))
    save_data(data_file, frames, labels, adversarial_labels)
    print('Saved generated frames to %s' % data_file)

