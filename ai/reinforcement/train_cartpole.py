import numpy as np

from keras.layers import Dense
from keras.models import  clone_model
from keras.models import Sequential
from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt

from collections import deque

import gym

from rl_agents import DDQNAgent, PGAgent

if __name__ == "__main__":

    n_episodes = 20000
    training_start = 100
    training_interval = 1
    copy_interval = 300
    
    batch_size = 32
    discount_rate = 0.99
    buffer_size = 50000
    eps_min = 0.0
    eps_max = 0.9
    eps_decay_steps = 1000
    
    reward_buffer = deque([], maxlen=100)
    
    model = Sequential([

                Dense(input_shape=[4],
                      name='dense1',
                      units=16,
                      activation='tanh'),
                Dense(name='dense2',
                      units=16,
                      activation='tanh'),
                Dense(name='out',
                      units=2,
                      activation='softmax',)
                ])
    model.compile(loss='mean_squared_error',  optimizer=Adam(),
                  metrics=['accuracy'])
            

    env = gym.make('CartPole-v0')
    agent = DDQNAgent(n_actions=2,
                      model=model,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      discount_rate=discount_rate,
                      eps_min=eps_min,
                      eps_max=eps_max,
                      eps_decay_steps=eps_decay_steps)
    
    
    def run_episode():
        
        reward_sum = 0
        state = env.reset()
        done = False
        
        while not done:
            
            try:
                env.render()
            except:
                pass
            
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience((state, action, reward, next_state, done))
            
            state = np.copy(next_state)
            
            reward_sum += reward
            
            # Train DDQNAgent
            if isinstance(agent, DDQNAgent):
                if agent.iteration > training_start and (agent.iteration % training_interval == 0):
                    agent.train()
                
                # Copy to target
                if agent.iteration > copy_interval and (agent.iteration % copy_interval == 0):
                    agent.update_target_weights()

        # Train PGAgent
        if isinstance(agent, PGAgent):
            agent.train()

        return reward_sum
    
    plt.ion()
    for episode_count in range(n_episodes):
        reward_sum = run_episode()
    
        reward_buffer.append(reward_sum)
        average = sum(reward_buffer) / len(reward_buffer)

        print("Episode Nr. {} Score: {} Average: {}".format(
            episode_count, reward_sum, average))
