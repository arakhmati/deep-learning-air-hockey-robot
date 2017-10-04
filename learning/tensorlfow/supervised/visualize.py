import os
import cv2
import h5py
import argparse
import progressbar
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--data_file', help='Name of the data file with training data')
args = parser.parse_args()
data_file = args.data_file
print(data_file)

with h5py.File(data_file, 'r') as f:
    frames = f['frames'][:]

frames = np.uint8(frames*256)
    
number_of_frames, height, width, _ = frames.shape  

video_file = 'visualize.avi'
if os.path.isfile(video_file): os.remove(video_file)
writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'PIM1'), 30, (width, height))

bar = progressbar.ProgressBar(max_value=number_of_frames)
for i, frame in enumerate(frames):
    frame = frame[:, :, ::-1] # Flip from BGR to RGB
    writer.write(frame)
    bar.update(i)
        
        