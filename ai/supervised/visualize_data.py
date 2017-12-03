import os
import h5py
import argparse
import progressbar
import numpy as np

from skvideo.io import VideoWriter

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
    
    frames = np.uint8(frames * 128 + 128)
    frames = np.concatenate((frames[:, :3], frames[:, 3:6], frames[:, 6:9]), axis=3)
    frames = frames.transpose((0, 2, 3, 1))
    n_frames, height, width, _ = frames.shape  
    
    writer = VideoWriter(video_file, frameSize=(width, height))
    writer.open()
    
    bar = progressbar.ProgressBar(max_value=n_frames)
    for i, frame in enumerate(frames):
        writer.write(frame)
        bar.update(i)
    writer.release()
    print('%s generated succesfully' % video_file)
        
        