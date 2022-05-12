import random

import numpy as np
from qlearning.agent import QLearnAgent
from qlearning.runner import GymRunner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    seed = 1
    set_seed(seed)

    gym = GymRunner("MountainCar-v0", seed)
    agent = QLearnAgent(gym.n_actions(), gym.obs_shape())

    gym.run(agent)
