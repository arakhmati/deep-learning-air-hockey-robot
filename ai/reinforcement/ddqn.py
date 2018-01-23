import numpy as np
from collections import deque

class Processor(object):
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        return observation.flatten()

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, label):
        return label

class DDQNAgent(object):
    
    def __init__(self,
                 models,
                 nb_actions,
                 buffer_size=1000,
                 batch_size=128,
                 discount_rate=0.99,
                 policy=None,
                 processor=Processor(),
                 eps_min=0.0,
                 eps_max=0.9,
                 eps_decay_steps=1000):
        
        self.iteration = 0
        
        self.model, self.target_model = models
        self.nb_actions = nb_actions
        self.use_target = False
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        
        self.experience_buffer = deque([], maxlen=buffer_size)
        self.loss_buffer = deque([], maxlen=1000)
        
        self.processor = processor
        
        if policy is None:
            
            self.eps_min = eps_min
            self.eps_max = eps_max
            self.eps_decay_steps = eps_decay_steps
            
            def epsilon_greedy(q_values):
                epsilon = max(eps_min, eps_max - (eps_max - eps_min) * self.iteration / eps_decay_steps)
                if self.iteration % 100 == 0:
                    print('Epsilon %f' % epsilon)
                if np.random.uniform(0, 1) < epsilon:
                    return np.random.randint(self.nb_actions)
                return np.argmax(q_values)
            
            self.policy = epsilon_greedy

    def sample_memories(self, batch_size):
        
#        for state, action, reward, next_state, done in self.experience_buffer:
#            from plot_utils import plot_states
#            plot_states(state, action, reward, next_state, done)
        
        from random import sample
        min_len = min(batch_size, len(self.experience_buffer))
        random_memories = sample(self.experience_buffer, min_len)
        
        transposed = [[], [], [], [], []]
        for memory in random_memories:
            for transposed_row, value in zip(transposed, memory):
                transposed_row.append(value)
        return [np.array(transposed_row) for transposed_row in transposed]
        
        return random_memories
    
    def store_memory(self, memory):
        state, action, reward, next_state, done = memory
        self.experience_buffer.append((np.copy(state), action, reward, np.copy(next_state), done))
    
    def predict(self, X, use_target=False, expand_dims=False):
        if expand_dims:
            X = np.expand_dims(X, axis=0)
        if use_target:
            return self.target_model.predict(X)
        return self.model.predict(X)
    
    def compute_action(self, state):
        self.iteration += 1
        q_values = self.predict(state, expand_dims=True)
        action = self.policy(q_values)
        return action
    
    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())
        self.use_target = True
        self.model.save('rl_model.h5')
        print('Copied weights from target_model to model')
        
    def process_action(self, action):
        return self.processor.process_action(action)
    
    def process_observation(self, observation):
        return self.processor.process_observation(observation)

    def train_q(self):
            states, actions, rewards, next_states, done = self.sample_memories(self.batch_size)
            
            targets = self.predict(states)
            next_q_values = self.predict(next_states, use_target=self.use_target)
            next_actions = np.argmax(self.predict(next_states), axis=1)
            
            row_indices = np.arange(targets.shape[0])
            targets[row_indices, actions] = rewards + (1 - done) * self.discount_rate * next_q_values[row_indices, next_actions]
            
            targets = np.clip(targets, -1, 1)

            loss = self.model.fit(states, targets, epochs=1, verbose=0, shuffle=True)
            self.loss_buffer.append(loss.history['loss'])