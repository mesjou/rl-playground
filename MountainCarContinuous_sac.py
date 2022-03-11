import random

import numpy as np
import tensorflow as tf
from sac.agent import SACAgent
from sac.runner import GymRunner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    seed = 8
    set_seed(seed)

    gym = GymRunner("MountainCarContinuous-v0", seed)
    agent = SACAgent(gym.obs_shape(), gym.action_shape())

    gym.run(agent)
