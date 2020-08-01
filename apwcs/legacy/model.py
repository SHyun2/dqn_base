import random

import numpy as np
from collections import deque
import tensorflow as tf


class DQN:
    REPLAY_MEMORY = 10000
    BATCH_SIZE = 32
    DISCOUNT_RATE = 0.99
    STATE_LEN = 5

    # Consider
    MAX_EPISODES = 5000

    def __init__(self, session, num_of_ue, n_action):
        self.session = session
        self.n_action = n_action
        self.num_of_ue = num_of_ue

        # Save the result
        self.memory = deque()
        # Now state
        self.state = None

        # Input of the game state
        # [Nodes of UE, the number of game states]
        self.input_X = tf.placeholder(tf.float32, [None, num_of_ue, self.STATE_LEN])
        # Action value which made each state
        self.input_A = tf.placeholder(tf.int64, [None])
        # Input for calculating loss
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        # Target Q value network
        self.target_Q = self._build_network('target')

    def _build_network(self, name):
        with tf.variable_scope(name):
            # Hidden layer dimension
            h_size = 16
            # Number of discrete actions
            output_size = 512

            net = self.input_X
            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.layers.dense(net, output_size)

            Q = net

        return Q

    def _build_op(self):
        # Perform a gradient descent step on (y_j-Q(ð_j,a_j;θ))^2
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])
        return action

    def init_state(self, state):
        # Initialize current state
        state = [state for _ in range(self.STATE_LEN)]

        self.state = np.stack(state, axis=2)

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.num_of_ue, 1))
        next_state = np.append(self.state[:, 1:], next_state, axis=2)

        # Save the state and reward to the memory
        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        # Get data in the memory which is stored game play
        # And sample the data as batch size
        state, next_state, action, reward, terminal = self._sample_memory()

        # Calculate target Q value inserting next state to target network
        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        # These calculated values are used for the loss function of DQN
        #
        # if episode is terminates at step j+1 then r_j
        # otherwise r_j + γ*max_a'Q(ð_(j+1),a';θ')
        Y = []

        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.DISCOUNT_RATE * np.max(target_Q_value[i]))

        self.session.run(
            self.train_op,
            feed_dict={
                self.input_X: state,
                self.input_A: action,
                self.input_Y: Y
            }
        )
