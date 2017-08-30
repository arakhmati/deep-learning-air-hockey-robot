from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym
import gym_air_hockey

import sys
sys.path.append('../supervised_learning')
from model import fmeasure, recall, precision

if __name__ == "__main__":

    env = gym.make('AirHockey-v0')
    
    model = load_model('../supervised_learning/conv.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    policy = EpsGreedyQPolicy(eps=0.1)
    memory = SequentialMemory(limit=2000, window_length=1)
    nb_steps_warm_up = 100
    target_model_update = 1
    enable_double_dqn = True
    
    nb_steps = 2000000
    nb_max_episode_steps = 150
    
    dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.nb_actions, 
                   nb_steps_warmup=nb_steps_warm_up, enable_double_dqn=enable_double_dqn,
                   target_model_update=target_model_update, processor=gym_air_hockey.DataProcessor())
    dqn.compile(Adam(), metrics=['mae', 'accuracy'])
    
    dqn.fit(env, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, verbose=9)
    
    dqn.model.save('rl_conv.h5')
    dqn.test(env, nb_episodes=5)
