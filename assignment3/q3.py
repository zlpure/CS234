import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt
from tqdm import *


def rmax(env, gamma, m, R_max, epsilon, num_episodes, max_step = 6, e = 0.7):
    """Learn state-action values using the Rmax algorithm

    Args:
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    gamma: float
        Discount factor. Number in range [0, 1)
    m: int
        	Threshold of visitance
    R_max: float 
        The estimated max reward that could be obtained in the game
    epsilon: 
        accuracy paramter
    num_episodes: int 
        Number of episodes of training.
    max_step: Int
        max number of steps in each episode

    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state-action values
    """

    Q = np.ones((env.nS, env.nA)) * R_max / (1 - gamma)
    R = np.zeros((env.nS, env.nA))
    nSA = np.zeros((env.nS, env.nA))
    nSASP = np.zeros((env.nS, env.nA, env.nS))
    ########################################################
    #                   YOUR CODE HERE                     #
    ########################################################
    total_score = 0
    average_score = np.zeros(num_episodes)
    for time in range(num_episodes):
        is_done = False
        cur_state = env.reset()
        for _ in range(max_step):
            if is_done:
                break
            if np.random.rand() > e:
                action = np.argmax(Q[cur_state])
            else:
                action = np.random.randint(env.nA)
            (next_state, reward, is_done, _) = env.step(action)
            total_score += reward
            if nSA[cur_state][action] < m:
                nSA[cur_state][action] += 1
                R[cur_state][action] += reward
                nSASP[cur_state][action][next_state] +=1
                if nSA[cur_state][action] == m:
                    up_bound = int(np.ceil(np.log(1.0/(epsilon*(1.0-gamma)))/(1.0-gamma)))
                    for i in range(up_bound):
                        for s in range(env.nS):
                            for a in range(env.nA):
                                if nSA[s][a] >= m:
                                    q_temp = R[s][a] / nSA[s][a]
                                    for j in range(env.nS):
                                        prob = nSASP[s][a][j] / nSA[s][a]    
                                        q_temp += gamma*prob*np.max(Q[j])
                                    Q[s][a] = q_temp
            cur_state = next_state
        average_score[time] = total_score / (time+1)
    ########################################################
    #                    END YOUR CODE                     #
    ########################################################
    return (Q, average_score)


def main():
    env = FrozenLakeEnv(is_slippery=False)
    print env.__doc__
    (Q, average_score) = rmax(env, gamma = 0.99, m=1, R_max = 1, epsilon = 0.1, num_episodes = 1000)
    render_single_Q(env, Q)
    plt.plot(np.arange(1000),np.array(average_score))
    plt.title('The running average score of the R-max with e-greedy learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    #plt.show()
    plt.savefig('r-max+e_greedy.jpg')

if __name__ == '__main__':
    print "haha"
    main()