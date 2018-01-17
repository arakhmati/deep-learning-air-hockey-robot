import numpy as np

from keras.layers import Dense
from keras.models import  clone_model
from keras.models import Sequential
from keras.optimizers import SGD

import matplotlib.pyplot as plt

from collections import deque

import gym

from ddqn import DDQNAgent

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
    
    def models():
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
                          activation='linear',)
                    ])
        model.compile(loss='mean_squared_error',  optimizer=SGD(lr=0.0075, momentum=0.5, decay=1e-6, clipnorm=2),
                      metrics=['accuracy'])
    
        target_model = clone_model(model)
        target_model.compile(loss='mean_squared_error',  optimizer=SGD(lr=0.0075, momentum=0.5, decay=1e-6, clipnorm=2),
                      metrics=['accuracy'])
        target_model.set_weights(model.get_weights())
        
        model.summary()
        target_model.summary()
        
        return model, target_model

        for model_weight, target_model_weight in zip(model.get_weights(), target_model.get_weights()):
            equals = np.allclose(model_weight, target_model_weight)
            assert equals
            

    env = gym.make('CartPole-v0')
    print(dir(env))
    agent = DDQNAgent(models=models(),
                      nb_actions=2,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      discount_rate=discount_rate,
                      eps_min=eps_min,
                      eps_max=eps_max,
                      eps_decay_steps=eps_decay_steps)
    
    
    def run_episode():
        
        reward_sum = 0
        observation = env.reset()
        state = agent.process_observation(observation)
        done = False
        
        while not done:
            
            try:
                env.render()
            except:
                pass
            
            action = agent.compute_action(state)
            observation, reward, done, _ = env.step(agent.process_action(action))
            
            # Get next state and store data to experience buffer
            next_state = agent.process_observation(observation)
            agent.store_memory((state, action, reward, next_state, done))
            state = next_state
            
            reward_sum += reward
            
            # Train
            if agent.iteration > training_start and (agent.iteration % training_interval == 0):
               agent.train_q()
            
               plt.clf()
               plt.plot(agent.loss_buffer)
               plt.pause(0.00001)
                
            # Copy to target
            if agent.iteration > copy_interval and (agent.iteration % copy_interval == 0):
                agent.update_target_weights()

        return reward_sum
    
    plt.ion()
    for episode_count in range(n_episodes):
        reward_sum = run_episode()
    
        reward_buffer.append(reward_sum)
        average = sum(reward_buffer) / len(reward_buffer)

        print("Episode Nr. {} Score: {} Average: {}".format(
            episode_count, reward_sum, average))
