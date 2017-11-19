import h5py

def load_data(data_file):
    with h5py.File(data_file, 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]   
    return frames, labels