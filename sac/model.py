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
    def call(self, state):
        x = self.model(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        # todo are logprobs calculated correctly?
        logprob = normal_dist.log_prob(action) - tf.math.log(
            1.0 - tf.pow(squashed_actions, 2) + 1e-6
        )  # epsilon for numerical stability
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob


class QCritic(Model):
    def __init__(self, hidden_layers: List, state_dim: int, n_actions: int):
        super(QCritic, self).__init__()
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
    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        return self.model(x)
