import numpy as np
import tensorflow as tf
from algorithms.ppo1.utils import normc_initializer, PPOLossInfo
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class PPOAgent(tf.keras.Model):
    def __init__(self, state_shape, action_size):
        super().__init__()

        self.state_shape = state_shape
        self.action_size = action_size

        # hyperparameters
        self.clip_coef = 0.2  # the surrogate clipping coefficient
        self.ent_coef = 0.01  # coefficient of the entropy
        self.vf_coef = 0.5  # coefficient of the value function
        self.learning_rate = 2.5e-4  # the learning rate of the optimizer

        # agent state
        self.base_model = self.build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-5)

    def build_model(self):
        inputs = Input(shape=(int(np.product(self.state_shape)),), name="observations")

        critic_1 = Dense(64, activation="tanh", kernel_initializer=normc_initializer(1.0))(inputs)
        critic_2 = Dense(64, activation="tanh", kernel_initializer=normc_initializer(1.0))(critic_1)
        value = Dense(1, kernel_initializer=normc_initializer(1.0))(critic_2)

        actor_1 = Dense(64, activation="tanh", kernel_initializer=normc_initializer(1.0))(inputs)
        actor_2 = Dense(64, activation="tanh", kernel_initializer=normc_initializer(1.0))(actor_1)
        logits = Dense(self.action_size, activation=None, kernel_initializer=normc_initializer(0.01))(actor_2)

        return tf.keras.Model(inputs, [logits, value])

    def forward(self, obs):
        logits, values = self.base_model(obs)
        return logits, values

    def logp(self, logits, action):
        """Get the log-probability based on the action drawn from prob-distribution"""
        logp_all = tf.nn.log_softmax(logits)
        one_hot = tf.one_hot(action, depth=self.action_size)
        logp = tf.reduce_sum(one_hot * logp_all, axis=-1)
        return logp

    def entropy(self, logits=None):
        """Entropy term for more exploration based on OpenAI Baseline openai/baselines/common/distributions.py"""
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis=-1, keepdims=True)
        p0 = exp_a0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def action_and_value(self, obs, actions=None):
        logits, values = self.base_model(obs)
        if actions is None:
            actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        log_probs = self.logp(logits, actions)
        return np.squeeze(actions), tf.squeeze(values), tf.squeeze(log_probs), self.entropy(logits)

    def policy_loss(self, advantages, ratio):
        """Normalize advantages and calculate policy loss."""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        min_advantages = tf.where(advantages > 0, (1 + self.clip_coef) * advantages, (1 - self.clip_coef) * advantages)
        return -tf.reduce_mean(tf.math.minimum(ratio * advantages, min_advantages))

    def value_loss(self, values, values_new, returns):
        """Calculate clipped value function loss"""
        v_clipped = values + tf.clip_by_value(values_new - values, -self.clip_coef, self.clip_coef,)
        v_loss_clipped = tf.square(v_clipped - returns)
        v_loss_unclipped = tf.square(values_new - returns)

        v_loss_max = tf.math.maximum(v_loss_unclipped, v_loss_clipped)
        return 0.5 * tf.reduce_mean(v_loss_max)

    def loss(self, obs, actions, returns, advantages, values, log_probs):

        # get new prediciton of logits, entropy and value
        actions_new, values_new, log_probs_new, entropy = self.action_and_value(obs, actions)
        log_ratio = log_probs_new - log_probs
        ratio = tf.exp(log_ratio)

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        approx_kl = tf.reduce_mean((ratio - 1) - log_ratio)
        clip_fracs = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_coef), tf.int32))

        # calculate losses
        policy_loss = self.policy_loss(advantages, ratio)
        value_loss = self.value_loss(values, values_new, returns)
        entropy_loss = -tf.reduce_mean(entropy)

        total_loss = policy_loss + self.ent_coef * entropy_loss + value_loss * self.vf_coef

        return PPOLossInfo(
            total_loss=total_loss,
            value_loss=value_loss,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            approx_kl=approx_kl,
            clip_fracs=clip_fracs,
        )
