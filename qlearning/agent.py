import model
import numpy as np
#print("HELLO")


class QLearnAgent():
    def __init__(self, obs = [0,0], action = None, reward = None, numberactions = 3, numberpos = 20, numbervelo = 15):
            self.obs= obs
            self.action = action
            self.reward = reward
            self.numberactions = numberactions
            self.done = False
            num_states = (model.env.observation_space.high - model.env.observation_space.low) * \
                         np.array([10, 100])# diskretize from another programm
            num_states = np.round(num_states, 0).astype(int) + 1
            print(num_states)
            self.qtable = np.random.uniform(low=-1, high=1,
                                  size=(num_states[0], num_states[1],
                                        model.env.action_space.n))
            self.qtable[num_states[0]-1] = np.array([[0 for i in range(model.env.action_space.n)] for j in range(num_states[1])])#terminal state should have value zero
            #self.qtable = np.array([[[ 0.5 for k in range(numberactions)] for j in range(numbervelo)] for i in range(numberpos)], float)
            #self.qtable = np.random.uniform(low=-1, high=1,size=(numberpos, numbervelo, numberactions))
            #self.qtable = [[[ None for k in range(numbervelo)] for j in range(numberpos)] for i in [0,1,2]] hier Actions außerhalb, für Vergleich effizienter innerhalb

    def act(self,epsilon = 0.3):# epsilon greedy policy
        if np.random.random() > epsilon:
            self.action = np.argmax(self.qtable[self.obs[0]][self.obs[1]])
        else:
            self.action = np.random.choice(self.numberactions)  # take a random action

    def learn(self, alpha = 0.2, gamma = 0.9, epsilon = 0.1):#qlearning
                oldstate = self.obs
                oldq = self.qtable[oldstate[0]][oldstate[1]][self.action]
                observation, self.reward, self.done, info = model.env.step(self.action)# step, action decided by act()
                if self.done and observation[0] >= 0.5:#terminal state
                    self.discretize_state(observation)
                    newq = oldq + round(( alpha * (- oldq)), 4)#in terminal state reward is zero and qtable value is initialized zero
                else:
                    self.discretize_state(observation)
                    newq = oldq + round(( alpha * (self.reward + gamma * np.max(self.qtable[self.obs[0]][self.obs[1]]) - oldq)), 4)
                self.qtable[oldstate[0]][oldstate[1]][self.action] = newq

    def discretize_state(self, observation, numbervelo = 15,numberpos = 20):#from another project
        state_adj = (observation - model.env.observation_space.low) * np.array([10, 100])
        self.obs = np.round(state_adj, 0).astype(int)




    """
    Ursprüngliche Diskretisierung
    
    def discretize_pos(observation, numberpos = 20):#von -1.2 bis 0.6
        # Input Array of length two with two observations
        # Output discretized value of position
        if observation[0] < -1:#eigentlich -1,2
            return 0
        if observation[0] > 0.5:#eigentlich 0.5
            return numberpos - 1
        else:
            return round(((observation[0]+1)/1.5)*(numberpos-2))


    def discretize_velo(observation, numbervelo = 15):#von -0.07 bis 0.07
        # Input Array of length two with two observations
        # Output discretized value of velocity
        if observation[1] < -0.05:
            return 0
        if observation[1] > 0.05:
            return numbervelo - 1
        else:
            return round(((observation[1]+0.05)/0.1)*(numbervelo-2))
            
    def maxaction(self):#if there are actions with equal values a random action of those will be choosen
        smallq = self.qtable[self.obs[0]][self.obs[1]]
        listmaxactions = np.argwhere(smallq == np.amax(smallq))
        if len(listmaxactions) == 1:
            return listmaxactions[0][0]
        else:
            listmaxactions2 = np.array([ob[0] for ob in listmaxactions])
            return np.random.choice(listmaxactions2)
    

    def discretize_state(self, observation, numbervelo = 15,numberpos = 20):
        #self.obs[0] = QLearnAgent.discretize_pos(observation, numberpos)
        #self.obs[1] = QLearnAgent.discretize_velo(observation, numbervelo)
    
    """
