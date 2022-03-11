from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input


class GaussianActor(Model):
    def __init__(self, hidden_layers: List, n_actions: int):
        """Create actor model that outputs mean and sd for all actions"""
        # todo maybe remove sampling to outer methods, than we can sample and return mean dependend on training or test
        super(GaussianActor, self).__init__()
        self.model = Sequential()
        for hidden_units in hidden_layers:
            self.model.add(Dense(hidden_units, activation="relu"))
        self.mean = Dense(
            n_actions,
            kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
            bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
        )
        self.log_std = Dense(
            n_actions,
            kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
            bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
        )

    @tf.function
    def call(self, state, training=True):

        # get mean and std
        net_out = self.model(state)
        mean = self.mean(net_out)
        log_std = self.log_std(net_out)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)

        # Action sampling and squashing of actions to -1 to 1
        pi_distribution = tfp.distributions.Normal(mean, std)
        if training:
            action = tf.stop_gradient(pi_distribution.sample())
        else:
            action = mean

        squashed_action = tf.tanh(action)

        # Get log probabilities
        logp = pi_distribution.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_action, 2) + 1e-6)
        logp = tf.reduce_sum(logp, axis=-1, keepdims=True)

        return squashed_action, logp


class Critic(Model):
    def __init__(self, hidden_layers: List, state_dim: int, n_actions: int):
        super(Critic, self).__init__()
        self.model = Sequential()
        self.model.add(Input(shape=(state_dim + n_actions), dtype=tf.float32))
        for hidden_units in hidden_layers:
            self.model.add(Dense(hidden_units, activation="relu"))
        self.model.add(
            Dense(
                1,
                kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
            )
        )

    @tf.function
    def call(self, state, action):
        net_out = tf.concat([state, action], 1)
        return self.model(net_out)
