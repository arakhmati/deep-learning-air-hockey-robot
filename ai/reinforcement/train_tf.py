import sys
sys.path.append('../supervised/keras')

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import load_model, clone_model
from keras.utils.np_utils import to_categorical

from collections import deque

from models import fmeasure, recall, precision

import gym
import gym_air_hockey


# Set learning phase to make sure all placeholders have values
K.set_learning_phase(1)

if __name__ == "__main__":

    env = gym.make('AirHockey-v0')
    processor = gym_air_hockey.DataProcessor()

    actor = load_model('../supervised/keras/models/model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})

    critic = clone_model(actor)
    critic.set_weights(actor.get_weights())

    def get_vars(model):
        return {var.name: var for var in model.trainable_weights}
    actor_q_values = actor.output
    critic_q_values = critic.output
    actor_vars = get_vars(actor)
    critic_vars = get_vars(critic)

    X_action = tf.placeholder(tf.int32, shape=[None])
    q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, 10), axis=1, keep_dims=True)

    y = tf.placeholder(tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(cost, global_step=global_step)

    replay_memory_size = 10000
    replay_memory = deque([], maxlen=replay_memory_size)

    def sample_memories(batch_size):
        indices = np.random.permutation(len(replay_memory))[:batch_size]
        cols = [[], [], [], [], []]
        for idx in indices:
            memory = replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    eps_min = 0.05
    eps_max = 1.0
    eps_decay_steps = 50000

    def epsilon_greedy(q_values, step):
        epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
        if np.random.randn() < epsilon:
            return np.random.randint(10)
        return np.argmax(q_values)

    n_steps = 1000000
    training_start = 10
    training_interval = 50
    save_steps = 3
    copy_steps = 25
    discount_rate = 0.95
    skip_start = 90
    batch_size = 128
    iteration = 0
    done = True

    sess = K.get_session()
    while True:
        step = global_step.eval(session=sess)
        if iteration >= n_steps:
            break
        iteration += 1
        if done:
            observation = env.reset()
            state = processor.process_observation(observation)

        q_values = actor_q_values.eval(session=sess, feed_dict={actor.input: [state]})
        action = epsilon_greedy(q_values, step)

        observation, reward, done, info = env.step(processor.process_action(action))
        if reward != 0.0:
            print(reward)
        next_state = processor.process_observation(observation)

        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if iteration < training_start or iteration % training_interval != 0:
            continue

        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)
        next_q_values = actor_q_values.eval(session=sess, feed_dict={actor.input: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1).reshape(-1, 1)
        y_val = rewards + continues * discount_rate * max_next_q_values
        training_op.run(session=sess, feed_dict={critic.input: X_state_val, X_action: X_action_val, y: y_val})

        if iteration % copy_steps == 0:
            actor.set_weights(critic.get_weights())

    actor.save('rl_model.h5')
