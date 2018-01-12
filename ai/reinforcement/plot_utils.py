import numpy as np
import matplotlib.pyplot as plt

def plot_state(state):
    state = np.uint8(np.copy(state) * 128 + 128)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(state[0:3].transpose((1,2,0)))
    ax[1].imshow(state[3:6].transpose((1,2,0)))
    ax[2].imshow(state[6:9].transpose((1,2,0)))
    plt.show()