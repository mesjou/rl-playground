from abc import ABC
from typing import Sequence

import tensorflow as tf
from sac.model import Critic, GaussianActor
from sac.utils import ReplayBuffer, TrainingInfo


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
        self.qnet1 = Critic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)
        self.qnet1_target = Critic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)
        self.qnet2 = Critic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)
        self.qnet2_target = Critic(hidden_layers=[64, 64], state_dim=observation_space, n_actions=n_actions)

        # Optimizers
        self.optimizer_actor = tf.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_q1 = tf.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_q2 = tf.optimizers.Adam(learning_rate=self.lr)

        # Reset weights of the networks with hard update: polyak=1.0
        soft_update(self.qnet1.variables, self.qnet1_target.variables, 1.0)
        soft_update(self.qnet2.variables, self.qnet2_target.variables, 1.0)

    def act(self, obs) -> float:
        """Get the action for a single state."""

        # unpack observation for action masking
        # state = obs["state"]
        # action_mask = obs["action_mask"]
        state = obs

        return self.actor(state[None, :])[0][0]

    def learn(self, obs, action, reward, next_obs, done):

        # unpack observation for action masking
        # state = obs["state"]
        # next_state = next_obs["state"]
        # next_action_mask = next_obs["action_mask"]

        # store to buffer and draw sample batch
        self.replay_buffer.store(obs, action, reward, next_obs, done)
        batch = self.replay_buffer.sample_batch(self.batch_size)

        # update actor and critic
        # todo should they update at the same time?
        log = self.train(
            obs=batch["obs1"], action=batch["acts"], rew=batch["rews"], next_obs=batch["obs2"], done=batch["done"]
        )

        return log

    def train(self, obs, action, rew, next_obs, done):

        # Computing action and a_tilde
        next_action, next_logp = self.actor(next_obs)

        # Taking the minimum of the q-functions values and add entropy
        target_q1 = self.qnet1_target(next_obs, action)
        target_q2 = self.qnet2_target(next_obs, action)
        target_q = tf.math.minimum(target_q1, target_q2) - self.temperature * next_logp

        # Computing target for q-functions
        backup = rew + self.gamma * (1 - done) * tf.reshape(target_q, [-1])
        backup = tf.reshape(backup, [self.batch_size, 1])

        # Gradient descent for the two local critic networks
        with tf.GradientTape() as tape_q1:
            q1 = self.qnet1(obs, action)
            loss_q1 = tf.reduce_mean(tf.square(q1 - backup))
        with tf.GradientTape() as tape_q2:
            q2 = self.qnet2(obs, action)
            loss_q2 = tf.reduce_mean(tf.square(q2 - backup))

        # gradient descent for actor
        with tf.GradientTape() as tape_a:
            action, logp = self.actor(obs)
            q_pi = tf.math.minimum(self.qnet1(obs, action), self.qnet2(obs, action))

            # New actor_loss -> works better than: loss = tf.reduce_mean(self.temperature * logp - q_pi)
            advantage = tf.stop_gradient(logp - q_pi)
            loss_actor = tf.reduce_mean(logp * advantage)

        # Computing the gradients
        gradients_actor = tape_a.gradient(loss_actor, self.actor.trainable_weights)
        gradients_q1 = tape_q1.gradient(loss_q1, self.qnet1.trainable_weights)
        gradients_q2 = tape_q2.gradient(loss_q2, self.qnet2.trainable_weights)

        # Backprop
        self.optimizer_q1.apply_gradients(zip(gradients_q1, self.qnet1.trainable_weights))
        self.optimizer_q2.apply_gradients(zip(gradients_q2, self.qnet2.trainable_weights))
        self.optimizer_actor.apply_gradients(zip(gradients_actor, self.actor.trainable_weights))

        # Update the weights of the soft q-function target networks
        soft_update(self.qnet1.variables, self.qnet1_target.variables, self.polyak_coef)
        soft_update(self.qnet2.variables, self.qnet2_target.variables, self.polyak_coef)

        # logging
        q1_mean, q1_variance = tf.nn.moments(q1, axes=[0])
        q2_mean, q2_variance = tf.nn.moments(q2, axes=[0])

        return TrainingInfo(
            q1=q1_mean[0],
            q2=q2_mean[0],
            q1_var=q1_variance[0],
            q2_var=q2_variance[0],
            logp=tf.reduce_mean(logp),
            loss_actor=loss_actor,
            loss_q1=loss_q1,
            loss_q2=loss_q2,
        )

    def set_learning_rate(self, lr):
        self.lr = lr

    def get_learning_rate(self):
        return self.lr

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_temperature(self):
        return self.temperature


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
