import numpy as np
import matplotlib.pyplot as plt

direction = ['NW', 'W', 'SW', 'N', 'Stand', 'S', 'NE', 'E', 'SE', 'Undefined']

def plot_states(state, action, reward, next_state, done):
    
    state = np.uint8(np.copy(state) * 128 + 128)
    next_state = np.uint8(np.copy(next_state) * 128 + 128)
    print('{} {} {}'.format(direction[action], reward, done))
    
    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(state[0:3].transpose((1,2,0)))
    ax[0, 1].imshow(state[3:6].transpose((1,2,0)))
    ax[0, 2].imshow(state[6:9].transpose((1,2,0)))
    ax[1, 0].imshow(next_state[0:3].transpose((1,2,0)))
    ax[1, 1].imshow(next_state[3:6].transpose((1,2,0)))
    ax[1, 2].imshow(next_state[6:9].transpose((1,2,0)))
    
    plt.show()