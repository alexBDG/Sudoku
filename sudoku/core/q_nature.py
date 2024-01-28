import tensorflow as tf

from utils.test_env import EnvTest

from core.q_linear import Linear
from core.q_schedule import LinearSchedule
from core.q_schedule import LinearExploration

from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
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

        with tf.variable_scope(scope, reuse=reuse):
            # ( batch_size x 84 x 84 x 4 )
            out = tf.layers.conv2d(state,
                                   filters=32,
                                   kernel_size=8,
                                   strides=4,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 19 x 19 x 32 )
            out = tf.layers.conv2d(out,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 8 x 8 x 64 )
            out = tf.layers.conv2d(out,
                                   filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 6 x 6 x 64 )
            out = tf.layers.flatten(out)
            # ( batch_size x 2304 )
            out = tf.layers.dense(out,
                                  512,
                                  activation="relu")
            # ( batch_size x 512 )
            out = tf.layers.dense(out, 
                                  num_actions)
            # ( batch_size x num_actions )

        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
