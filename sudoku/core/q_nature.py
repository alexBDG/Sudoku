# System imports.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')

# Package imports.
from .q_linear import Linear



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
        # Get output shape
        num_actions = self.env.action_space.shape

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
