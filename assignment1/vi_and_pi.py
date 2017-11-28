### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    idx = 1
    new_V = V.copy()
    #print P[14][2]
    while idx<=max_iteration or np.sum(np.sqrt(np.square(new_V-V)))>tol:
        idx += 1
        V = new_V
        for state in range(nS):
            max_result = -10
            max_idx = 0
            for action in range(nA):
                result = P[state][action]
                temp = np.array(result)[:,2].mean()
                #temp = result[0][2]
                for num in range(len(result)):
                    (probability, nextstate, reward, terminal) = result[num]
                    temp += gamma*probability*V[nextstate]
                    if max_result < temp:
                        max_result = temp
                        max_idx = action
            new_V[state] = max_result
            policy[state] = max_idx
        #print new_V
        #print policy
    ############################
    return V, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=100, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
    The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    value_function = np.zeros(nS)
    new_value_function = value_function.copy()
    i = 0
    while i<=max_iteration or np.sum(np.sqrt(np.square(new_value_function-value_function)))>tol:
        i += 1
        value_function = new_value_function.copy()
        for state in range(nS):
            result = P[state][policy[state]]
            new_value_function[state] = np.array(result)[:,2].mean()
            for num in range(len(result)):
                (probability, nextstate, reward, terminal) = result[num]
                new_value_function[state] += (gamma * probability * value_function[nextstate])
    ############################
    return new_value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        	number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """    
    ############################
    # YOUR IMPLEMENTATION HERE #
    q_function = np.zeros([nS,nA])
    for state in range(nS):
        for action in range(nA):
            result = P[state][action]
            for num in range(len(result)):
                (probability, nextstate, reward, terminal) = result[num]
                q_function[state][action] = reward
                q_function[state][action] += (gamma*probability*value_from_policy[nextstate])
    new_policy = np.argmax(q_function, axis=1)
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=200, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    i = 0 
    new_policy= policy.copy()
    while i<=max_iteration or np.sum(np.sqrt(np.square(new_policy-policy)))>tol:
        i += 1
        policy = new_policy
        V = policy_evaluation(P, nS, nA, policy)
        new_policy = policy_improvement(P, nS, nA, V, policy)
    ############################
    return V, policy



def example(env):
    """Show an example of gym
    Parameters
    	----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
    """
    env.seed(0); 
    from gym.spaces import prng; prng.seed(10) # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render();

def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        #time.sleep(0.5) # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render();
    print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    print env.__doc__
    #print "Here is an example of state, action, reward, and next state"
    #example(env)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    #V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    render_single(env, p_vi)
	