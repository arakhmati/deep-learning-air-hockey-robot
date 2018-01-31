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

models_dir = dir_path + '/models/' + config.mode + '/'
robot_model_name = 'robot_model'
human_model_name = 'human_model'

robot_model_file = models_dir + robot_model_name + '.h5'
human_model_file = models_dir + human_model_name + '.h5'

if __name__ == "__main__":

    n_episodes = 20000
    episode_length = 200
    training_start = 100
    training_interval = 25
    copy_interval = 2500
    
    batch_size = 64
    discount_rate = 0.99
    buffer_size = 10000
    eps_min = 0.0
    eps_max = 0.99
    eps_decay_steps = 100000
    
    reward_buffer = deque([], maxlen=100)
    
    def models():
        model = load_model('../supervised/keras/models/rgb/robot_model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

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
    
    def rl_models():
        model = load_model('rl_model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    
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
                      n_actions=env.n_actions,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      discount_rate=discount_rate,
                      eps_min=eps_min,
                      eps_max=eps_max,
                      eps_decay_steps=eps_decay_steps,
                      processor=gym_air_hockey.DataProcessor())
    
    
    
    programmed_action = True
    action_distribution = np.zeros((env.n_actions), dtype=np.uint32)
    def run_episode():
        global programmed_action
        
        reward_sum = 0
        state = env.reset()
        done = False
        
        episode_step = 0
        
        while not done and episode_step < episode_length:
            
            try:
                env.render()
            except:
                pass
            
            action = agent.compute_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            # Store data to experience buffer
            agent.store_memory((state, action, reward, next_state, done))
            state = np.copy(next_state)
            
            action_distribution[action] += 1
            reward_sum += reward
            
            # Train
            if agent.iteration > training_start and agent.iteration % training_interval == 0:
                if agent.iteration % 100 == 0:
                    print('Iteration: {}'.format(agent.iteration))
                    print('Action Distribution: ', end='')
                    print(action_distribution / sum(action_distribution))
                
                agent.train_q()
            
                plt.clf()
                plt.plot(agent.loss_buffer)
                plt.pause(0.00001)
                
            # Copy to target
            if agent.iteration > copy_interval and agent.iteration % copy_interval == 0:
                agent.update_target_weights()
                
            episode_step += 1

        return reward_sum
    
    plt.ion()
    for episode_count in range(n_episodes):
        reward_sum = run_episode()
    
        reward_buffer.append(reward_sum)
        average = sum(reward_buffer) / len(reward_buffer)

        print("Episode Nr. {} Score: {} Average: {}".format(
            episode_count, reward_sum, average))

