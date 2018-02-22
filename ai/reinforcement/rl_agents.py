import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../')
sys.path.append('../supervised/keras')

import numpy as np
from abc import ABC, abstractmethod
from collections import deque

import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import  clone_model, load_model, Model
from keras.optimizers import SGD, Adam

from metrics import fmeasure, recall, precision

class Agent(ABC):
    
    def __init__(self,
                 n_actions,
                 buffer_size=1000,
                 batch_size=128,
                 discount_rate=0.99,
                 policy=None,
                 eps_min=0.0,
                 eps_max=0.9,
                 eps_decay_steps=1000,
                 model_file='rl_model.h5'):

        self.iteration = 0
        self.use_target = False
        
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.discount_rate = discount_rate

        self.action_distribution = np.zeros((self.n_actions), dtype=np.uint32)
        
        self._reset_experience_buffer()
        self.loss_buffer = deque([], maxlen=100000)
        
        self.model_file = model_file
        
        if policy is None:
            self.eps_min = eps_min
            self.eps_max = eps_max
            self.eps_decay_steps = eps_decay_steps
            def epsilon_greedy(q_values):
                epsilon = max(eps_min, eps_max - (eps_max - eps_min) * self.iteration / eps_decay_steps)
                if self.iteration % 100 == 0:
                    print('Epsilon %f' % epsilon)
                if np.random.uniform(0, 1) < epsilon:
                    return np.random.randint(self.n_actions)
                return np.argmax(q_values)
            self.policy = epsilon_greedy
            
    def _reset_experience_buffer(self):
        self.experience_buffer = deque([], maxlen=self.buffer_size)
            
    def _sample_experiences(self, batch_size):
        
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

    def _predict(self, X, use_target=False, expand_dims=False):
        if expand_dims:
            X = np.expand_dims(X, axis=0)
        if use_target:
            return self.target_model.predict(X)
        return self.model.predict(X)
        
    def store_experience(self, memory):
        state, action, reward, next_state, done = memory
        self.experience_buffer.append((np.copy(state), action, reward, np.copy(next_state), done))
        
    def act(self, state):
        
        self.iteration += 1
        prediction = self._predict(state, expand_dims=True)
        action = self.policy(prediction)
            
        self.action_distribution[action] += 1
        self.prediction = prediction[0]

        return action
        
    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())
        self.use_target = True
        self.model.save(self.model_file)
        print('Copied weights from model to target_model')

    def _set_target_model(self):
        self.target_model = clone_model(self.model)
        self.target_model.compile(
                               loss='mean_squared_error',  
                               optimizer=SGD(lr=0.0075, momentum=0.5, decay=1e-6, clipnorm=2),
                               metrics=['accuracy'])
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.summary()
   
        for model_weight, target_model_weight in zip(self.model.get_weights(), self.target_model.get_weights()):
            equals = np.allclose(model_weight, target_model_weight)
            assert equals

    def train(self):
        if self.iteration % 1000 == 0:
            print('Iteration: {}'.format(self.iteration))
            print('Action Distribution: ', end='')
            print(self.action_distribution / sum(self.action_distribution))

class DDQNAgent(Agent):
    
    def __init__(self,
                 n_actions,
                 pretrained_model_file=None,
                 model=None,
                 buffer_size=1000,
                 batch_size=128,
                 discount_rate=0.99,
                 policy=None,
                 eps_min=0.0,
                 eps_max=0.9,
                 eps_decay_steps=1000,
                 model_file='ddqn_model.h5',
                 replace_softmax=True):
                 
        super().__init__(
            n_actions=n_actions,
            buffer_size=buffer_size,
            batch_size=batch_size,
            discount_rate=discount_rate,
            policy=policy,
            eps_min=eps_min,
            eps_max=eps_max,
            eps_decay_steps=eps_decay_steps,
            model_file=model_file
        )

        if pretrained_model_file != None:
            self.model = load_model(pretrained_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
        elif model != None:
            self.model = model
        else:
            raise ValueError('A file containing model or a model object needs to specified')

        if replace_softmax:
            # Replace softmax layer with linear activation
            # By giorgiop on https://github.com/keras-team/keras/issues/3465 commented on 10 Oct 2016
            self.model.layers.pop()
            self.model.layers[-1].outbound_nodes = []
            self.model.outputs = [self.model.layers[-1].output]
            output = self.model.layers[-1].output
            output = Dense(units=self.n_actions, name='output', activation='linear')(output) # your newlayer Dense(...)
            self.model = Model(self.model.input, output)
            self.model.compile(
                          loss='mean_squared_error',  
                          optimizer=SGD(lr=0.0075, momentum=0.5, decay=1e-6, clipnorm=2),
                          metrics=['accuracy'])
        
        self.model.summary()
        self._set_target_model()
        
    def train(self, plot=True):
        super().train()
        states, actions, rewards, next_states, done = self._sample_experiences(self.batch_size)
        
        targets = self._predict(states)
        next_q_values = self._predict(next_states, use_target=self.use_target)
        next_actions = np.argmax(self._predict(next_states), axis=1)
        
        row_indices = np.arange(targets.shape[0])
        targets[row_indices, actions] = rewards + (1 - done) * self.discount_rate * next_q_values[row_indices, next_actions]

        targets = np.clip(targets, -1.0, 1.0)

        loss = self.model.fit(states, targets, epochs=1, verbose=0, shuffle=True)
        self.loss_buffer.append(loss.history['loss'])

        if plot:
            plt.clf()
            plt.plot(self.loss_buffer)
            plt.pause(0.1)
            
class PGAgent(Agent):
    
    def __init__(self,
                 n_actions,
                 pretrained_model_file=None,
                 model=None,
                 buffer_size=1000,
                 batch_size=128,
                 discount_rate=0.99,
                 policy=None,
                 eps_min=0.0,
                 eps_max=0.9,
                 eps_decay_steps=1000,
                 model_file='pg_model.h5',
                 learning_rate=0.01):
                 
                 
        super().__init__(
            n_actions=n_actions,
            buffer_size=buffer_size,
            batch_size=batch_size,
            discount_rate=discount_rate,
            policy=policy,
            eps_min=eps_min,
            eps_max=eps_max,
            eps_decay_steps=eps_decay_steps,
            model_file=model_file
        )
        self.learning_rate = learning_rate

        if pretrained_model_file != None:
            self.model = load_model(pretrained_model_file, {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
        elif model != None:
            self.model = model
        else:
            raise ValueError('A file containing model or a model object needs to specified')
        self.model.summary()

    def _sample_experiences(self):

        memories = list(self.experience_buffer)

        transposed = [[], [], [], []]
        for memory in memories:
            for transposed_row, value in zip(transposed, memory):
                transposed_row.append(value)
        return [np.array(transposed_row) for transposed_row in transposed]
        
    def store_experience(self, memory):
        state, action, reward, _, _ = memory
        
        y = np.zeros_like(self.prediction, dtype=np.float32)
        y[action] = 1.0

        # self.prediction is set in act() which should always be called before this method
        self.experience_buffer.append((np.copy(state), reward, self.prediction, y - self.prediction))
        
    def train(self, plot=True):
        super().train()

        def discount_rewards(rewards):
            rewards = np.array(rewards)
            # Discount episode rewards
            for i in reversed(range(1, len(rewards))):
                rewards[i-1] += self.discount_rate * rewards[i]
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            rewards -= np.mean(rewards)
            # Fix division by zero bug by using np.divide when scaling by std
            rewards_std = np.std(rewards)
            rewards = np.divide(rewards, rewards_std, out=np.zeros_like(rewards), where=rewards_std!=0)

            return rewards
            
        states, rewards, predictions, gradients = self._sample_experiences()
        rewards = discount_rewards(rewards)
        gradients *= rewards.reshape((rewards.shape[0], 1))
        
        targets = predictions + self.learning_rate * gradients
        
        loss = self.model.fit(states, targets, epochs=1, verbose=0, shuffle=True)
        self.loss_buffer.append(loss.history['loss'])

        self.model.save(self.model_file)
        
        self._reset_experience_buffer()

        if plot:
            plt.clf()
            plt.plot(self.loss_buffer)
            plt.pause(0.00001)