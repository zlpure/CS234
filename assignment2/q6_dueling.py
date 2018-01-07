import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q6_bonus_question import config


class MyDQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks 
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
       
        L1: 32 8x8 filters with stride 4  +  RELU
        L2: 64 4x4 filters with stride 2  +  RELU
        L3: 64 3x3 fitlers with stride 1  +  RELU
        L4a: 512 unit Fully-Connected layer  +  RELU
        L4b: 512 unit Fully-Connected layer  +  RELU
        L5a: 1 unit FC  (State Value)
        L5b: #actions FC (Advantage Value)
        L6: Aggregate V(s)+A(s,a)
        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        
        with tf.variable_scope(scope, reuse=reuse) as _:
            out = layers.conv2d(out, num_outputs=32, kernel_size=8, stride=4)
            out = layers.conv2d(out, num_outputs=64, kernel_size=4, stride=2)
            out = layers.conv2d(out, num_outputs=64, kernel_size=3, stride=1)
            out = layers.flatten(out)
            out = layers.fully_connected(out, num_outputs=512)
            out1 = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            out2 = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            out = out2 - tf.tile(tf.expand_dims(tf.reduce_mean(out2, axis=1),-1), [1,num_actions])
            out = out + tf.tile(out1, [1,num_actions])
            
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
