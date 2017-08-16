import numpy as np
import gym
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym_air_hockey


if __name__ == "__main__":
    
    env = gym.make('AirHockey-v0')
    
    model = load_model('../reinforcement_learning/model.h5')

    actions = np.zeros(6, dtype=np.float32)
    
#    while True:
#        if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
#        observations = env.step(actions)
#        observations =  observations[:, env.dim.vertical_margin:-env.dim.vertical_margin, :].transpose((1,0,2))
#        writer.write(observations[:,:,::-1])
#        observations = cv2.resize(observations, (128, 128)).reshape((1,128,128,3))
#        observations = (observations.astype(np.float32)-128)/128
#        
#        actions[:] = np.array(model.predict(observations)).flatten()
    
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=200, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=2, memory=memory, nb_steps_warmup=10,
    target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['accuracy'])
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
    dqn.fit(env, nb_steps=1, verbose=2)
    
    dqn.test(env, nb_episodes=5, visualize=True, nb_steps=50000)
    
#    pygame.quit ()