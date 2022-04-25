import numpy as np
import agent
import qtablememory
import gym
import numpy as np
import runner


#runner.qlearning()

Q = qtablememory.B

env = gym.make('MountainCar-v0')
for i in range(100000):
    #print('next Episode', i)
    done = False
    env.reset()
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
        if reward == 0:
            print('---------------------------------')
            print('---------------------------------')
            print('---------------------------------')
            print('REWARD==0')
            print('---------------------------------')
            print('---------------------------------')
        #env.render()


testagent = agent.QLearnAgent()
def maxaction(obs):
    smallq = obs
    listmaxactions = np.argwhere(smallq == np.amax(smallq))
    if len(listmaxactions) == 1:
        return listmaxactions[0][0]
    else:
        listmaxactions2 = np.array([ob[0] for ob in listmaxactions])
        return np.random.choice(listmaxactions2)

"""""
testagent = agent.QLearnAgent()
observation = model.env.reset()
print('obs1', observation)
print('obs2', agent.QLearnAgent.discretize_pos(observation))
print('obs3', testagent.discretize_state(observation))
print('obs4', testagent.observation)
#testagent.observation = testagent.discretize_state(observation)
#print('obs4', testagent.observation)
testagent.act()
print('obs5', testagent.observation)
print('obs5', testagent.reward)
print('obs5', testagent.observation)


def discretize_pos(observation, numberpos):
    #Input Array of length two with two observations
    #Output discretized value of position
    if observation[0] < -1.2:
        return 0
    if observation[0] >= 0.5:
        return numberpos - 1
    else:
        stepsize = round(1.7 / (numberpos - 2),2)
        print(stepsize)
        for N in range(1, numberpos):
            if (-1.2 + ((N - 1) * stepsize)) <= observation[0] < (-1.2 + (N * stepsize)):
                return N
        return numberpos - 2

print(discretize_pos([0.33,0], 10))
"""""

"""
        else:
            stepsize = round(1.7 / (numbervelo - 2), 3)
            print(stepsize)
            for N in range(1, numbervelo):
                if (-1.2 + ((N - 1) * stepsize)) <= observation[1] < (-1.2 + (N * stepsize)):
                    return N
            return numbervelo - 2
        """

"""
        else:
            stepsize = round(1.7 / (numberpos - 2), 3)
            #print(stepsize)
            for N in range(1, numberpos):
                if (-1.2 + ((N - 1) * stepsize)) <= observation[0] < (-1.2 + (N * stepsize)):
                    return N
            return numberpos - 2
        """


"""""""""
def discretize_pos(observation, numberpos=100):
    # Input Array of length two with two observations
    # Output discretized value of position
    if observation[0] < -1.2:
        return 0
    if observation[0] >= 0.5:
        return numberpos - 1
    else:
        stepsize = round(1.7 / (numberpos - 2), 3)
        print(stepsize)
        for N in range(1, numberpos):
            if (-1.2 + ((N - 1) * stepsize)) <= observation[0] < (-1.2 + (N * stepsize)):
                return N
        return numberpos - 2


def discretize_velo(observation, numbervelo=100):
    # Input Array of length two with two observations
    # Output discretized value of velocity
    if observation[1] < -1.2:
        return 0
    if observation[1] >= 0.5:
        return numbervelo - 1
    else:
        stepsize = round(1.7 / (numbervelo - 2), 3)
        print(stepsize)
        for N in range(1, numbervelo):
            if (-1.2 + ((N - 1) * stepsize)) <= observation[1] < (-1.2 + (N * stepsize)):
                return N
        return numbervelo - 2
        
      Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 8]
Oldq is -9.998560550421862 Newq is -9.998560550421862 at position [8, 8, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 8]
Oldq is -9.998560550421862 Newq is -9.998560550421862 at position [8, 8, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 8]
Oldq is -9.998560550421862 Newq is -9.998560550421862 at position [8, 8, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 9, 7]
Oldq is -9.998582304549629 Newq is -9.998582304549629 at position [9, 7, 0, 9, 7]
Oldq is -9.998582304549629 Newq is -9.998582304549629 at position [9, 7, 0, 9, 7]
Oldq is -9.998582304549629 Newq is -9.998582304549629 at position [9, 7, 0, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998603528882102 Newq is -9.998603528882102 at position [8, 7, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998603528882102 Newq is -9.998603528882102 at position [8, 7, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.99857217256974 Newq is -9.99857217256974 at position [8, 7, 0, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998603528882102 Newq is -9.998603528882102 at position [8, 7, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.99857217256974 Newq is -9.99857217256974 at position [8, 7, 0, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998603528882102 Newq is -9.998603528882102 at position [8, 7, 1, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.99857217256974 Newq is -9.99857217256974 at position [8, 7, 0, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.99857217256974 Newq is -9.99857217256974 at position [8, 7, 0, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.99857217256974 Newq is -9.99857217256974 at position [8, 7, 0, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]
Oldq is -9.998557800053637 Newq is -9.998557800053637 at position [8, 7, 2, 8, 7]  
        
"""""""""
