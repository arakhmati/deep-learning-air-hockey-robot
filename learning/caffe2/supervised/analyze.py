import h5py
import glob
import matplotlib.pyplot as plt

def load_data(data_file=None):
    if data_file is None:
        data_file = 'data/2017-10-06-18-14-31_448.h5'
    print(data_file)
    with h5py.File(data_file, 'r', driver='core') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]   
    return frames, labels

if __name__ == '__main__':
    frames, labels = load_data()
    
    print(frames)
    
    directions_map = ['NW', 'N', 'NE', 'W', '', 'E', 'SW', 'S', 'SE']
    
    for frame_idx in range(frames.shape[0]):
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(frames[frame_idx][0])
        axarr[1].imshow(frames[frame_idx][1])
        axarr[2].imshow(frames[frame_idx][2])
        f.suptitle(directions_map[labels[frame_idx]])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
