from abc import ABC
from typing import Sequence

import tensorflow as tf
from sac.model import GaussianActor, QCritic
from sac.utils import ReplayBuffer

# todo use mean action for testing -> better results than sampling from distribution


class SACAgent(ABC):
    def __init__(self, observation_space: int, n_actions: int):

        # Hyperparameters
        self.replay_size = int(1e6)
        self.lr = 0.001
        self.temperature = 0.3
        self.gamma = 0.99
        self.polyak_coef = 0.01
        self.batch_size = 128

        # replay buffer and networks
        self.replay_buffer = ReplayBuffer(state_dim=observation_space, n_actions=n_actions, size=self.replay_size)
        self.actor = GaussianActor(hidden_layers=[64, 64], n_actions=n_actions)
        self.qnetwork_1 = QCritic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)
        self.target_qnetwork_1 = QCritic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)
        self.qnetwork_2 = QCritic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)
        self.target_qnetwork_2 = QCritic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)

        # Optimizers
        # todo could be replaced by recitified adam (maybe better performance)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.qnetwork_1_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.qnetwork_2_optimizer = tf.optimizers.Adam(learning_rate=self.lr)

        # Reset weights of the networks with hard update: polyak=1.0
        soft_update(self.qnetwork_1.variables, self.target_qnetwork_1.variables, 1.0)
        soft_update(self.qnetwork_2.variables, self.target_qnetwork_2.variables, 1.0)

    def act(self, obs) -> float:
        """Get the action for a single state."""

        # unpack observation
        # state = obs["state"]
        # action_mask = obs["action_mask"]
        state = obs

        return self.actor(state[None, :])[0][0]

    def learn(self, obs, action, reward, next_obs, done):

        # unpack observation
        # state = obs["state"]
        # next_state = next_obs["state"]
        # next_action_mask = next_obs["action_mask"]

        # store to buffer and draw sample batch
        self.replay_buffer.store(obs, action, reward, next_obs, done)
        batch = self.replay_buffer.sample_batch(self.batch_size)

        # update actor and critic
        # todo should they update at the same time?
        self.train(
            obs=batch["obs1"], action=batch["acts"], rew=batch["rews"], next_obs=batch["obs2"], done=batch["done"]
        )

    def train(self, obs, action, rew, next_obs, done):

        # Computing action and a_tilde
        next_action, logp = self.actor(next_obs)

        # Taking the minimum of the q-functions values and add entropy
        value_target_1 = self.target_qnetwork_1(next_obs, action)
        value_target_2 = self.target_qnetwork_2(next_obs, action)
        value_target = tf.math.minimum(value_target_1, value_target_2) - self.temperature * logp

        # Computing target for q-functions
        target = rew + self.gamma * (1 - done) * tf.reshape(value_target, [-1])
        target = tf.reshape(target, [self.batch_size, 1])

        # Gradient descent for the two local critic networks
        with tf.GradientTape() as q_1_tape:
            qvalue_1 = self.qnetwork_1(obs, action)
            loss_1 = tf.reduce_mean(tf.square(qvalue_1 - target))
        with tf.GradientTape() as q_2_tape:
            qvalue_2 = self.qnetwork_2(obs, action)
            loss_2 = tf.reduce_mean(tf.square(qvalue_2 - target))

        # gradient descent for actor
        # todo deviated from example here
        with tf.GradientTape() as actor_tape:
            action, logp = self.actor(obs)
            qvalue = tf.math.minimum(self.qnetwork_1(obs, action), self.qnetwork_2(obs, action))
            # loss = tf.reduce_mean(self.temperature * logp - qvalue)

            # New actor_loss -> works better
            advantage = tf.stop_gradient(logp - qvalue)
            loss = tf.reduce_mean(logp * advantage)

        # Computing the gradients and applying them
        actor_gradients = actor_tape.gradient(loss, self.actor.trainable_weights)
        gradients_1 = q_1_tape.gradient(loss_1, self.qnetwork_1.trainable_weights)
        gradients_2 = q_2_tape.gradient(loss_2, self.qnetwork_2.trainable_weights)
        self.qnetwork_1_optimizer.apply_gradients(zip(gradients_1, self.qnetwork_1.trainable_weights))
        self.qnetwork_2_optimizer.apply_gradients(zip(gradients_2, self.qnetwork_2.trainable_weights))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))

        # Update the weights of the soft q-function target networks
        soft_update(self.qnetwork_1.variables, self.target_qnetwork_1.variables, self.polyak_coef)
        soft_update(self.qnetwork_2.variables, self.target_qnetwork_2.variables, self.polyak_coef)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon

    def set_learning_rate(self, alpha):
        self.alpha = alpha

    def get_learning_rate(self):
        return self.alpha


def soft_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable], tau: float) -> None:
    """Move each source variable by a factor of tau towards the corresponding target variable.
    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
        tau {float} -- How much to change to source var, between 0 and 1.
    """
    if len(source_vars) != len(target_vars):
        raise ValueError("source_vars and target_vars must have the same length.")
    for source, target in zip(source_vars, target_vars):
        target.assign((1.0 - tau) * target + tau * source)
