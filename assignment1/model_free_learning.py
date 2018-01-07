### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *
import matplotlib.pyplot as plt
from tqdm import *

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [env.nS x env.nA] representing state, action values
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    q_value = np.zeros([env.nS, env.nA])  
    for i in range(num_episodes):
        done = False
        state = env.reset()
        while not done:
            if np.random.rand() > e:
                action = np.argmax(q_value[state])
            else:
                action = np.random.randint(env.nA)
            nextstate, reward, done, _ = env.step(action)
            q_value[state][action] = (1-lr)*q_value[state][action]+lr*(reward+gamma*np.max(q_value[nextstate]))
            state = nextstate
        if i%10 == 0:
            e *= decay_rate        
    '''
    print np.mean(q_value)
    
    plt.plot(np.arange(num_episodes),np.array(score))
    plt.title('The running average score of the Q-learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    #plt.show()
    plt.savefig('c.jpg')
    '''
    ############################
    return q_value

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [env.nS x env.nA] representing state-action values
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    q_value = np.zeros([env.nS, env.nA])  
    for i in range(num_episodes):
        done = False
        state = env.reset()
        if np.random.rand() > e:
            action = np.argmax(q_value[state])
        else:
            action = np.random.randint(env.nA)
        while not done:
            nextstate, reward, done, _ = env.step(action)
            if np.random.rand() > e:
                nextaction = np.argmax(q_value[nextstate])
            else:
                nextaction = np.random.randint(env.nA)
            q_value[state][action] = (1-lr)*q_value[state][action]+lr*(reward+gamma*q_value[nextstate][nextaction])
            state = nextstate
            action = nextaction
        if i%10 == 0:
            e *= decay_rate
    ############################

    return q_value

def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play Q function on. Must have nS, nA, and P as
        attributes.
    Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        #env.render()  #show frames 
        #time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    #print "Episode reward: %f" % episode_reward
    return episode_reward
    
# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    score1 = []
    score2 = []
    average_score1 = []
    average_score2 = []
    for i in tqdm(range(4000)):
        Q1 = learn_Q_QLearning(env, num_episodes=i+1)
        Q2 = learn_Q_SARSA(env, num_episodes=i+1)
        episode_reward1 = render_single_Q(env, Q1)
        episode_reward2 = render_single_Q(env, Q2)
        score1.append(episode_reward1)
        score2.append(episode_reward2)
    for i in range(4000):
        average_score1.append(np.mean(score1[:i+1]))
        average_score2.append(np.mean(score2[:i+1]))
    plt.plot(np.arange(4000),np.array(average_score1))
    plt.plot(np.arange(4000),np.array(average_score2))
    plt.title('The running average score of the Q-learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    plt.legend(['q-learning', 'sarsa'], loc='upper right')
    #plt.show()
    plt.savefig('model-free.jpg')
           
if __name__ == '__main__':
    main()
