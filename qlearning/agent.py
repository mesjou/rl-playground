from typing import Tuple

import numpy as np


class QLearnAgent():
    def __init__(self, n_actions: int, n_states: Tuple):
        self.n_actions = n_actions
        self.n_states = n_states
        self.qtable = np.random.uniform(low=-1, high=1, size=self.n_states + (self.n_actions,))

    def get_max_action(self, next_state: np.array) -> int:
        """return action with highest q-value. If values equal choose randomly."""
        smallq = self.qtable[tuple(next_state)]
        listmaxactions = np.argwhere(smallq == np.amax(smallq))
        if len(listmaxactions) == 1:
            return listmaxactions[0][0]
        else:
            return np.random.choice(listmaxactions.flatten())

    def act(self, next_state: np.array, epsilon: float = 0) -> int:
        """epsilon greedy policy"""
        if np.random.random() > epsilon:
            return self.get_max_action(next_state)
        else:
            return np.random.choice(self.n_actions)  # take a random action

    def learn(self, action, reward, state, next_state, done, alpha: float = 0.8, gamma: float = 0.9):
        """update policy via q-learning"""
        oldq = self.qtable[tuple(state)][action]
        newq = round(oldq + (alpha * (1.0 - done) * (reward + gamma * np.max(self.qtable[tuple(next_state)]) - oldq)), 4)
        self.qtable[tuple(state)][action] = newq
