import os
import h5py
import argparse
import progressbar
import numpy as np

from skvideo.io import FFmpegWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_file', help='Name of the data file to visualize')
    args = parser.parse_args()
    data_file = args.data_file
    
    video_file = '%s.avi' % data_file[:-3]
    if os.path.isfile(video_file): 
        print('%s already exists' % video_file)
        quit()
    
    with h5py.File(data_file, 'r') as f:
        frames = f['frames'][:]
    
    frames = np.uint8(frames  * 128 + 128)
    n_frames, height, width = frames.shape  
    
    writer = FFmpegWriter(video_file)
    
    bar = progressbar.ProgressBar(max_value=n_frames)
    for i, frame in enumerate(frames):
        writer.writeFrame(frame)
        bar.update(i)
    print('%s generated succesfully' % video_file)
