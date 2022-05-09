import model
import numpy as np


class QLearnAgent():
    def __init__(self, obs = [0,0], action = None, reward = None, numberactions = 3, num_states = [30,10]):
            self.obs= obs
            self.action = action
            self.reward = reward
            self.numberactions = numberactions
            self.num_states = num_states
            self.qtable = np.random.uniform(low=-1, high=1,
                                  size=(self.num_states[0], self.num_states[1],
                                        model.env.action_space.n))
            #self.qtable[self.num_states[0]-1] = np.array([[0 for i in range(numberactions)] for j in range(self.num_states[1])])#terminal state should have value zero



    def maxaction(self,obs):  # if there are actions with equal values a random action of those will be choosen
        smallq = self.qtable[obs[0]][obs[1]]
        listmaxactions = np.argwhere(smallq == np.amax(smallq))
        if len(listmaxactions) == 1:
            return listmaxactions[0][0]
        else:
            listmaxactions2 = np.array([ob[0] for ob in listmaxactions])
            return np.random.choice(listmaxactions2)

    def act(self,obs,epsilon = 0.3):# epsilon greedy policy
        if np.random.random() > epsilon:
            #self.action = self.maxaction(obs)
            self.action = np.argmax(self.qtable[obs[0]][obs[1]])
        else:
            self.action = np.random.choice(self.numberactions)  # take a random action
        return self.action


    def learn(self, action, reward, state, obs, done, alpha = 0.8, gamma = 0.9):#qlearning
                #print(action, reward, state, obs, done)
                oldq = self.qtable[state[0]][state[1]][action]
                newq = round(oldq + ( alpha * (1- done) * (reward + gamma * np.max(self.qtable[obs[0]][obs[1]]) - oldq)), 4)
                self.qtable[state[0]][state[1]][action] = newq
