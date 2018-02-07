import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../')
sys.path.append(dir_path + '/../supervised/keras')

import config
        
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gym
import gym_air_hockey

from rl_agents import DDQNAgent, PGAgent
from plot_utils import plot_states

models_dir = dir_path + '/models/' + config.mode + '/'
robot_model_name = 'robot_model'
human_model_name = 'human_model'

robot_model_file = models_dir + robot_model_name + '.h5'
human_model_file = models_dir + human_model_name + '.h5'

if __name__ == "__main__":

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--pretrained_model_file', type=str, required=True, help='file with the pretrained model')
    parser.add_argument('-a', '--agent', type=str, default='ddqn', choices=['ddqn', 'pg'], help='reinforcement learning agent')
    args = parser.parse_args()
    pretrained_model_file = args.pretrained_model_file
    print(pretrained_model_file)
    agent_class = args.agent

    n_episodes = 200000
    episode_length = 200
    training_start = 100
    training_interval = 25
    copy_interval = 2500
    
    batch_size = 64
    discount_rate = 0.99
    buffer_size = 10000 if config.mode == 'rgb' else 60000
    eps_min = 0.1
    eps_max = 0.99
    eps_decay_steps = 1000000   

    env = gym.make('AirHockey-v0')
    env.update(mode=config.mode)


    agents = {
                'ddqn': DDQNAgent,
                'pg': PGAgent
             }
    Agent = agents[agent_class]

    agent = Agent(n_actions=env.n_actions,
                  pretrained_model_file=pretrained_model_file,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  discount_rate=discount_rate,
                  eps_min=eps_min,
                  eps_max=eps_max,
                  eps_decay_steps=eps_decay_steps,
                  model_file=robot_model_file)

    
    reward_buffer = deque([], maxlen=1000)  

    def run_episode():
        
        reward_sum = 0
        state = env.reset()
        done = False
        
        episode_step = 0
        while not done and episode_step < episode_length:
            
            try:
                env.render()
            except:
                pass
            
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)
            reward_sum += reward
            
            # Store data to experience buffer
            agent.store_experience((state, action, reward, next_state, done))
            state = np.copy(next_state)
            
            # Train DDQNAgent
            if isinstance(agent, DDQNAgent):
                if agent.iteration > training_start and (agent.iteration % training_interval == 0):
                    agent.train()
                
                # Copy to target
                if agent.iteration > copy_interval and (agent.iteration % copy_interval == 0):
                    agent.update_target_weights()
                
            episode_step += 1

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

