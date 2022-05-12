from gym.core import ObservationWrapper
import numpy as np


class ObservationDiscretizer(ObservationWrapper):
    def __init__(self, env, n_states):
        super().__init__(env)
        self.n_states = n_states

    def observation(self, observation):
        state_adj = (observation - self.env.observation_space.low) * (
            self.n_states / np.array([self.env.observation_space.high - self.env.observation_space.low])
        )
        return np.round(state_adj, 0).astype(int)[0]
