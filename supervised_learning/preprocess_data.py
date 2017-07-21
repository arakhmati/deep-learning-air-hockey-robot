import cv2
import numpy as np
import matplotlib.pyplot as plot

from skimage.io import imread
import glob

import scipy.io as sio

files = glob.glob('data/*.jpg')
files.sort()
N = len(files)


image = imread(files[0])
image = image[25:675, 25:475, :3]
height, width, depth = image.shape

width = width//5
height = height//5

images = np.zeros(N, height, width, depth)
labels = np.zeros(N, 2)

for i, file_name in enumerate(files):
    
    r = re.search('\d*_([-10]{1,2})_([-10]{1,2}).jpg', file_name, re.IGNORECASE)
    labels = np.array([r.group(1), r.group(2)])
    
#    print(file_name)
    
    # Open Image
    image = imread(file_name)
    
    # Crop Image
    image = image[25:675, 25:475, :3]
    
    # Resize image
    height, width, depth = image.shape
    image=cv2.resize(image, (width//5, height//5))
    images[i] = image
    
#    plot.imshow(image)
#    plot.show()

sio.save_mat('preprocessed_data.mat', {'images': images, 'labels': labels})


0000001_-1.0.jpg