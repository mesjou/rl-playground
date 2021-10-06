import random

import numpy as np
import tensorflow as tf
from ppo.agent import PPOAgent
from ppo.runner import GymRunner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    seed = 1
    set_seed(seed)

    gym = GymRunner("CartPole-v1", seed)
    agent = PPOAgent(gym.obs_shape(), gym.action_size())

    gym.run(agent)
