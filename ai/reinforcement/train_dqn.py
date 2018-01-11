import sys
sys.path.append('../supervised/keras')

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import load_model, clone_model
from keras.utils.np_utils import to_categorical

from collections import deque

from metrics import fmeasure, recall, precision

import gym
import gym_air_hockey
from plot_utils import plot

if __name__ == "__main__":

    K.set_learning_phase(1)

    n_steps = 1000000
    training_start = 100
    training_interval = 100
    save_steps = 10000
    copy_steps = 1000
    discount_rate = 0.95
    batch_size = 128
    iteration = 0
    done = True
    default_reset = 1000
    replay_memory_size = 2000
    eps_min = 0.0
    eps_max = 0.0
    eps_decay_steps = n_steps * 0.6

    env = gym.make('AirHockey-v0')
    processor = gym_air_hockey.DataProcessor()

    actor = load_model('../supervised/keras/models/model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    critic = clone_model(actor)
    critic.set_weights(actor.get_weights())

    for actor_weight, critic_weight in zip(actor.get_weights(), critic.get_weights()):
        assert np.allclose(actor_weight, critic_weight)

    def get_vars(model):
        return {var.name: var for var in model.trainable_weights}
    actor_q_values = actor.output
    critic_q_values = critic.output
    actor_vars = get_vars(actor)
    critic_vars = get_vars(critic)

    X_action = tf.placeholder(tf.int32, shape=[None])
    q_value = tf.reduce_sum(actor_q_values * tf.one_hot(X_action, 10), axis=1, keep_dims=True)

    y = tf.placeholder(tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(cost, global_step=global_step)

    replay_memory = deque([], maxlen=replay_memory_size)

    def sample_memories(batch_size):

        def get_experiences(filter_func):
            from random import shuffle
            experiences = [row for row in replay_memory if filter_func(row['reward'])]
            if len(experiences) == 0:
                print('Get experiences from replay_memory instead')
                experiences = [row for row in replay_memory]
            shuffle(experiences)
            experiences = experiences[:batch_size]

            return experiences

        rewarded_experiences = get_experiences(lambda x: x != 0.0)
        typical_experiences  = get_experiences(lambda x: x == 0.0)
#
        mixed_experiences = []
        for a, b in zip(rewarded_experiences, typical_experiences):
            mixed_experiences.append(a)
            mixed_experiences.append(b)
#
##            print('Rewarded: current')
##            plot(a['state'])
##            print('Rewarded: next')
##            plot(a['next_state'])
##
##            print('Typical: current')
##            plot(b['state'])
##            print('Typical: next')
##            plot(b['next_state'])
##        print('Press any button')
##        input()

        indices = np.random.permutation(len(mixed_experiences))[:batch_size]
        cols = [[], [], [], [], []]
        for idx in indices:
            memory = mixed_experiences[idx]
            for col, key in zip(cols, memory):
                col.append(memory[key])
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


    def epsilon_greedy(q_values, step):
        epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
#        if step % 100 == 0:
#            print('Epsilon %f' % epsilon)
        if np.random.randn() < epsilon:
            return np.random.randint(10)
        return np.argmax(q_values)

    reset = default_reset
    sess = K.get_session()
    while True:
        reset -= 1
        step = global_step.eval(session=sess)
        iteration += 1

        if step > n_steps:
            break

        if done or not reset:
            reset = default_reset
            observation = env.reset()
            _ = processor.process_observation(observation)
            _ = processor.process_observation(observation)
            state = np.copy(processor.process_observation(observation))

        q_values = actor_q_values.eval(session=sess, feed_dict={actor.input: [state]})
        action = epsilon_greedy(q_values, step)

        observation, reward, done, info = env.step(processor.process_action(action))
        if abs(reward) > 0.0:
            print('Reward %f' % reward)
        next_state = processor.process_observation(observation)

        replay_memory.append({'state': np.copy(state),
                              'action': action,
                              'reward': reward,
                              'next_state': np.copy(next_state),
                              'continue': 1.0 - done})
        state = np.copy(next_state)

        if iteration < training_start or iteration % training_interval != 0:
            continue

        print('Updated %d %d' % (iteration, step))
        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)

#        for s_t, a_t, r_t, s_t_1, c in zip(X_state_val, X_action_val, rewards, X_next_state_val, continues):
#            print(a_t, r_t, c)
#            plot(s_t)
#            plot(s_t_1)

        next_q_values = critic_q_values.eval(session=sess, feed_dict={critic.input: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1).reshape(-1, 1)
        y_val = rewards + continues * discount_rate * max_next_q_values
        training_op.run(session=sess, feed_dict={actor.input: X_state_val, X_action: X_action_val, y: y_val})

        if step % copy_steps == 0:
            critic.set_weights(actor.get_weights())

        if step % save_steps == 0:
            actor.save('rl_model.h5')


    actor.save('rl_model.h5')
