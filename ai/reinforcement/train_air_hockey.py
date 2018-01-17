import sys
sys.path.append('../supervised/keras')
        
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from keras.layers import Dense
from keras.models import  clone_model, load_model, Model
from keras.optimizers import SGD

import gym
import gym_air_hockey

from ddqn import DDQNAgent
from metrics import fmeasure, recall, precision
from plot_utils import plot_states

if __name__ == "__main__":

    n_episodes = 20000
    training_start = 100
    training_interval = 100
    copy_interval = 3000
    
    batch_size = 128
    discount_rate = 0.99
    buffer_size = 5000
    eps_min = 0.0
    eps_max = 0.9
    eps_decay_steps = 100000
    
    reward_buffer = deque([], maxlen=100)
    
    def models():
        model = load_model('../supervised/keras/models/model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

        # Replace softmax layer with linear activation
        # By giorgiop on https://github.com/keras-team/keras/issues/3465 commented on 10 Oct 2016
        model.layers.pop()
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
        output = model.get_layer('dense1').output
        output = Dense(units=10, name='output', activation='linear')(output) # your newlayer Dense(...)
        model = Model(model.input, output)
        model.compile(loss='mean_squared_error',  optimizer=SGD(lr=0.0075, momentum=0.5, decay=1e-6, clipnorm=2),
                      metrics=['accuracy'])
    
        target_model = clone_model(model)
        target_model.compile(loss='mean_squared_error',  optimizer=SGD(lr=0.0075, momentum=0.5, decay=1e-6, clipnorm=2),
                      metrics=['accuracy'])
        target_model.set_weights(model.get_weights())
        
        model.summary()
        target_model.summary()
    
        for model_weight, target_model_weight in zip(model.get_weights(), target_model.get_weights()):
            equals = np.allclose(model_weight, target_model_weight)
            assert equals
            
        return model, target_model
            

    env = gym.make('AirHockey-v0')
    agent = DDQNAgent(models=models(),
                      nb_actions=env.nb_actions,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      discount_rate=discount_rate,
                      eps_min=eps_min,
                      eps_max=eps_max,
                      eps_decay_steps=eps_decay_steps,
                      processor=gym_air_hockey.DataProcessor())
    
    
    def run_episode():
        
        reward_sum = 0
        observation = env.reset()
        _ = agent.process_observation(observation)
        _ = agent.process_observation(observation)
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
#            plot_states(state, action, reward, next_state, done)
            agent.store_memory((state, action, reward, next_state, done))
            state = next_state
            
            reward_sum += reward
            
            # Train
            if agent.iteration > training_start and (agent.iteration % training_interval == 0):
                print('Iteration: {}'.format(agent.iteration))
                
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

