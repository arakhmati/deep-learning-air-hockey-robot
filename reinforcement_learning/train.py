import numpy as np

#from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym
import gym_air_hockey

def build_model():
  
    model = Sequential([
                Conv2D(4, 16, activation='relu', name='conv1', input_shape=(128, 128, 3)),
                MaxPooling2D(3, name='pool1'),
                Conv2D(8, 16, activation='relu', name='conv2'),
                MaxPooling2D(3, name='pool2'),
                Flatten(name='flatten'),
                Dense(100, activation='relu', name='dense1'),
                Dense(50, activation='relu', name='dense2'),
                Dense(9,  activation='softmax', name='softmax')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
#    print(model.summary())
    return model


if __name__ == "__main__":

    env = gym.make('AirHockey-v0')
    
#    model = load_model('../reinforcement_learning/model.h5')
    model = build_model()
    policy = EpsGreedyQPolicy(eps=0.5)
    memory = SequentialMemory(limit=1000, window_length=1)
    nb_steps_warm_up = 1000
    target_model_update = 1e-2
    enable_double_dqn = True
    
    nb_steps = 100000
    nb_max_episode_steps = 50
    
    dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.nb_actions, 
                   nb_steps_warmup=nb_steps_warm_up, enable_double_dqn=enable_double_dqn,
                   target_model_update=target_model_update, processor=gym_air_hockey.AirHockeyProcessor())
    dqn.compile(Adam(), metrics=['mae', 'accuracy'])
    
    dqn.fit(env, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, verbose=9)
    
    dqn.test(env, nb_episodes=5, visualize=True)
