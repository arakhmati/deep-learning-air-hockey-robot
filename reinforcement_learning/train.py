import numpy as np

#from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym
import gym_air_hockey

from air_hockey_processor import AirHockeyProcessor

env = gym.make('AirHockey-v0')

def build_model():
    image_input = Input(shape=(128, 128, 3), dtype=np.float32, name='input')
    conv1  = Conv2D(8, 16, activation='relu', name='conv1')(image_input)
    pool1  = MaxPooling2D(3, name='pool1')(conv1)
    conv2  = Conv2D(16, 16, activation='relu', name='conv2')(pool1)
    pool2  = MaxPooling2D(3, name='pool2')(conv2)
    flat   = Flatten(name='flatten')(pool2)
    dense1 = Dense(100, activation='relu', name='dense1')(flat)
    output = Dense(9, activation='softmax', name='output')(dense1)
  
    model = Model(inputs=image_input, outputs=output)
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
#    print(model.summary())
    return model


if __name__ == "__main__":
    
#    model = load_model('../reinforcement_learning/model.h5')
    model = build_model()
    policy = EpsGreedyQPolicy(eps=0.3)
    memory = SequentialMemory(limit=1000, window_length=1)
    nb_steps_warm_up = 1000
    target_model_update = 1e-2
    enable_double_dqn = True
    
    nb_steps = 100000
    nb_max_episode_steps = 100
    
    dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.nb_actions, 
                   nb_steps_warmup=nb_steps_warm_up, enable_double_dqn=enable_double_dqn,
                   target_model_update=target_model_update, processor=AirHockeyProcessor())
    dqn.compile(Adam(), metrics=['mae', 'accuracy'])
    
    dqn.fit(env, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, verbose=9)
    
    dqn.test(env, nb_episodes=5, visualize=True)
