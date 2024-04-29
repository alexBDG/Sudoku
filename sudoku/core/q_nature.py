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

    def _get_q_values_op(self, state, scope, reuse=False):
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
            # ( batch_size x 9 x 9 x 40 )
            out_1 = tf.layers.conv2d(state,
                                     filters=32,
                                     kernel_size=(9, 1),
                                     strides=2,
                                     padding="same",
                                     activation="relu")
            # ( batch_size x 5 x 5 x 32 )

            # ( batch_size x 9 x 9 x 40 )
            out_2 = tf.layers.conv2d(state,
                                     filters=32,
                                     kernel_size=(1, 9),
                                     strides=2,
                                     padding="same",
                                     activation="relu")
            # ( batch_size x 5 x 5 x 32 )

            # ( batch_size x 9 x 9 x 40 )
            out_3 = tf.layers.conv2d(state,
                                     filters=32,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding="same",
                                     activation="relu")
            # ( batch_size x 5 x 5 x 32 )

            out = tf.concat([out_1, out_2, out_3], -1)

            # ( batch_size x 5 x 5 x 96 )
            out = tf.layers.conv2d(out,
                                   filters=128,
                                   kernel_size=4,
                                   strides=2,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 3 x 3 x 128 )

            out = tf.layers.flatten(out)

            # ( batch_size x 1152 )
            out = tf.layers.dense(out,
                                  512,
                                  activation="relu")

            # ( batch_size x 512 )
            out = tf.layers.dense(out,
                                  num_actions)
            # ( batch_size x num_actions )

        return out


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
            # ( batch_size x 18 x 18 x 4 ) => Snake
            # ( batch_size x 9 x 9 x 8 ) => Sudoku
            out = tf.layers.conv2d(state,
                                   filters=32,
                                   kernel_size=8,
                                   strides=2,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 9 x 9 x 32 ) => Snake
            # ( batch_size x 5 x 5 x 32 ) => Sudoku

            out = tf.layers.conv2d(out,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 5 x 5 x 64 ) => Snake
            # ( batch_size x 3 x 3 x 64 ) => Sudoku

            out = tf.layers.conv2d(out,
                                   filters=128,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   activation="relu")
            # ( batch_size x 5 x 5 x 128 ) => Snake
            # ( batch_size x 3 x 3 x 128 ) => Sudoku

            out = tf.layers.flatten(out)

            # ( batch_size x 3200 ) => Snake
            # ( batch_size x 1152 ) => Sudoku
            out = tf.layers.dense(out,
                                  512,
                                  activation="relu")

            # ( batch_size x 512 )
            out = tf.layers.dense(out,
                                  num_actions)
            # ( batch_size x num_actions )

        return out
