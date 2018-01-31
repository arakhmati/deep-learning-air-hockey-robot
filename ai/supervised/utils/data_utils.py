import h5py

def save_data(data_file, states, robot_actions, human_actions):
    with h5py.File(data_file , 'w') as f:
        f.create_dataset('states',        data=states)
        f.create_dataset('robot_actions', data=robot_actions)
        f.create_dataset('human_actions', data=human_actions)

def load_data(data_file):
    with h5py.File(data_file, 'r') as f:
        states        = f['states'][:]
        robot_actions = f['robot_actions'][:]  
        human_actions = f['human_actions'][:]  
    return states, robot_actions, human_actions