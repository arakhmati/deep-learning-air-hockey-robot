import os
import h5py
import argparse
import progressbar
import numpy as np

from skvideo.io import FFmpegWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='data file to visualize')
    args = parser.parse_args()
    data_file = args.data_file
    
    video_file = '%s.avi' % data_file[:-3]
    if os.path.isfile(video_file): 
        print('%s already exists' % video_file)
        quit()
    
    with h5py.File(data_file, 'r') as f:
        states = f['states'][:]
    
    states = np.uint8(states * 128 + 128)
    if 'rgb' in data_file:
        states = np.concatenate((states[:, :3], states[:, 3:6], states[:, 6:9]), axis=3)
    states = states.transpose((0, 2, 3, 1))
    n_states, height, width, _ = states.shape  
    
    writer = FFmpegWriter(video_file)
    
    bar = progressbar.ProgressBar(max_value=n_states)
    for i, state in enumerate(states):
        writer.writeFrame(state)
        bar.update(i)
    print('%s generated succesfully' % video_file)        
        