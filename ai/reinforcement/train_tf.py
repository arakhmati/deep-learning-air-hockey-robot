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


# Set learning phase to make sure all placeholders have values
K.set_learning_phase(1)

if __name__ == "__main__":

    n_steps = 1000000
    training_start = 10
    training_interval = 20
    default_reset = 500
    copy_steps = 200
    discount_rate = 0.97
    batch_size = 128
    iteration = 0
    done = True

    eps_min = 0.1
    eps_max = 1.0
    eps_decay_steps = n_steps * 0.2

    replay_memory_size = 1000

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
    q_value = tf.reduce_sum(actor_q_values * tf.one_hot(X_action, 10), axis=1, keep_dims=True)

    y = tf.placeholder(tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(cost, global_step=global_step)

    replay_memory = deque([], maxlen=replay_memory_size)

    def sample_memories(batch_size):
        # memorable_experiences = [row for row in replay_memory if row[2] != 0.0]
        # if len(memorable_experiences) == 0:
        memorable_experiences = replay_memory
        indices = np.random.permutation(len(memorable_experiences))[:batch_size]
        cols = [[], [], [], [], []]
        for idx in indices:
            memory = memorable_experiences[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


    def epsilon_greedy(q_values, step):
        epsilon = max(eps_min, eps_max - (eps_max - eps_min) * iteration / eps_decay_steps)
        if iteration % 100 == 0:
            print('Epsilon %f' % epsilon)
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
            state = processor.process_observation(observation)

        q_values = actor_q_values.eval(session=sess, feed_dict={actor.input: [state]})
        action = epsilon_greedy(q_values, step)

        observation, reward, done, info = env.step(processor.process_action(action), dt=np.random.randint(2, 4))
        if abs(reward) > 0.0:
            print('Reward %f' % reward)
        next_state = processor.process_observation(observation)

        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if iteration < training_start or iteration % training_interval != 0:
            continue

        print('Updated %d' % iteration)
        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)
        next_q_values = critic_q_values.eval(session=sess, feed_dict={critic.input: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1).reshape(-1, 1)
        y_val = rewards + continues * discount_rate * max_next_q_values
        training_op.run(session=sess, feed_dict={actor.input: X_state_val, X_action: X_action_val, y: y_val})

        if step % copy_steps == 0:
            critic.set_weights(actor.get_weights())


    actor.save('rl_model.h5')
