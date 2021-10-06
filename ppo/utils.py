import collections

import gym
import numpy as np
import tensorflow as tf


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def normc_initializer(std=1.0):
    """Custom  kernel initalizer copied from OpenAI baselines"""

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


PPOLossInfo = collections.namedtuple(
    "LossInfo", ("total_loss", "value_loss", "policy_loss", "entropy_loss", "approx_kl", "clip_fracs",)
)
