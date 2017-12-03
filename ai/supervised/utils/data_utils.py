import h5py

def save_data(data_file, frames, labels, adversarial_labels):
    with h5py.File(data_file , 'w') as f:
        f.create_dataset('frames', data=frames)
        f.create_dataset('labels', data=labels)
        f.create_dataset('adversarial_labels', data=adversarial_labels)

def load_data(data_file):
    with h5py.File(data_file, 'r') as f:
        frames = f['frames'][:]
        labels = f['labels'][:]  
        adversarial_labels = f['adversarial_labels'][:]  
    return frames, labels, adversarial_labels