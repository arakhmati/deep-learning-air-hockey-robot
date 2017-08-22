from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym
import gym_air_hockey

if __name__ == "__main__":

    env = gym.make('AirHockey-v0')
    
    model = load_model('../supervised_learning/bottom_ai_model.h5')
#    model = gym_air_hockey.build_model()
    policy = EpsGreedyQPolicy(eps=0.01)
    memory = SequentialMemory(limit=2000, window_length=1)
    nb_steps_warm_up = 2000
    target_model_update = 1e-2
    enable_double_dqn = False
    
    nb_steps = 200000
    nb_max_episode_steps = 75
    
    dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.nb_actions, 
                   nb_steps_warmup=nb_steps_warm_up, enable_double_dqn=enable_double_dqn,
                   target_model_update=target_model_update, processor=gym_air_hockey.DataProcessor())
    dqn.compile(Adam(), metrics=['mae', 'accuracy'])
    
    dqn.fit(env, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps, verbose=9)
    
    dqn.model.save('rl_model.h5')
    dqn.test(env, nb_episodes=5)
