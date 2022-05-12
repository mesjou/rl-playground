from typing import Tuple

import numpy as np


class QLearnAgent():
    def __init__(self, n_actions: int, n_states: Tuple):
        self.n_actions = n_actions
        self.n_states = n_states
        self.qtable = np.random.uniform(low=-1, high=1, size = self.n_states + (self.n_actions,) )

    def maxaction(self, next_state: np.array) -> int:
        """return action with highest q-value. If values equal choose randomly."""
        tuple_next_state = tuple(obj for obj in next_state)
        smallq = self.qtable[tuple_next_state]
        listmaxactions = np.argwhere(smallq == np.amax(smallq))
        if len(listmaxactions) == 1:
            return listmaxactions[0][0]
        else:
            listmaxactions2 = np.array([ob[0] for ob in listmaxactions])
            return np.random.choice(listmaxactions2)

    def act(self, next_state: np.array, epsilon: float = 0) -> int:
        """epsilon greedy policy"""
        if np.random.random() > epsilon:
            self.action = self.maxaction(next_state)
        else:
            self.action = np.random.choice(self.n_actions)  # take a random action
        return self.action

    def learn(self, action, reward, state, next_state, done, alpha = 0.8, gamma: float = 0.9):
        """update policy via q-learning"""
        oldq = self.qtable[state[0]][state[1]][action]
        newq = round(oldq + (alpha * (1.0 - done) * (reward + gamma * np.max(self.qtable[next_state[0]][next_state[1]]) - oldq)), 4)
        self.qtable[state[0]][state[1]][action] = newq
