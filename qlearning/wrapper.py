from gym.core import ObservationWrapper
import numpy as np


class ObservationDiscretizer(ObservationWrapper):
    def __init__(self, env, num_states):
        super().__init__(env)
        self.num_states = num_states

    def observation(self, observation):
        state_adj = (observation - self.env.observation_space.low) * (
            self.num_states / np.array([self.env.observation_space.high - self.env.observation_space.low])
        )
        return np.round(state_adj, 0).astype(int)[0]
