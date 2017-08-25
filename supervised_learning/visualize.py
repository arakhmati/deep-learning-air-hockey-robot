import os
import cv2
import h5py
import progressbar
import numpy as np
import matplotlib.pyplot as plt

project_path = os.path.dirname(os.path.realpath(__file__))
data_file = project_path + '/data.h5'

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
        
        