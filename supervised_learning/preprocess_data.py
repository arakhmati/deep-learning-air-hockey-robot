import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
import glob
import scipy.io as sio
import regex as re

import os
import threading

number_of_threads = 8

project_path = '/home/ahmed/Documents/41X'
games = glob.glob(project_path + '/supervised_data/*')
games.sort()

for game in [games[0]]:
    image_files = glob.glob(game + '/*.jpg')
    image_files.sort()
    number_of_examples = len(image_files)

    image = imread(image_files[0])
    image = image[25:675, 25:475, :]
    _, _, depth = image.shape
    
    width  = 128
    height = 128
    
    images = np.zeros((number_of_examples, height, width, depth), dtype=np.uint8)
    labels = np.zeros((number_of_examples, 2), dtype=np.int8)
    
    examples_per_thread = np.int(np.ceil(number_of_examples / number_of_threads))
    
    class DataThread (threading.Thread):
        def __init__(self, index):
            threading.Thread.__init__(self)
            self.index = index
        def run(self):
            process_image_batch(self.index)
    
    def process_image_batch(thread_index):
        try:
            start = thread_index * examples_per_thread
            end = min(start + examples_per_thread, number_of_examples)
            for i in range(start, end):
#                if i > 2:
#                    break
                if i % 1000 == 0:
                    print('%5d %5.2f%%\n' % (thread_index, (i-start)/(end-start)*100))
                file_name = image_files[i]
                r = re.search('\d*_([-10]{1,2})_([-10]{1,2}).jpg', file_name, re.IGNORECASE)
                labels[i] = np.array([np.int8(r.group(1)), np.int8(r.group(2))], dtype=np.int8)
                image = imread(file_name) # Open Image
                image = image[25:675, 25:475, :] # Crop Image
                images[i] = cv2.resize(image, (width, height)) # Resize image
#                plt.imshow(images[i])
#                plt.show()
        except Exception:
            import traceback
            print(traceback.format_exc())

    threadLock = threading.Lock()
    threads = []
    
    for i in range(number_of_threads):
        thread = DataThread(i)
        thread.start()
        threads.append(thread)
    
    for t in threads:
        t.join()
    
    np.savez_compressed(project_path + '/supervised_data/' + os.path.basename(os.path.normpath(game))+'_data.npz', images=images, labels=labels)