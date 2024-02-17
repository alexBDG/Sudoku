# System imports.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')

# Package imports.
from .deep_q_learning import DQN



class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = self.env.observation_space.shape

        self.s = tf.placeholder(tf.uint8,
                                shape=(None,
                                       state_shape[0],
                                       state_shape[1],
                                       state_shape[2] * self.config.state_history),
                                name="states")
        self.a = tf.placeholder(tf.int32,
                                shape=(None),
                                name="actions")
        self.r = tf.placeholder(tf.float32,
                                shape=(None),
                                name="rewards")
        self.sp = tf.placeholder(tf.uint8,
                                 shape=(None,
                                        state_shape[0],
                                        state_shape[1],
                                        state_shape[2] * self.config.state_history),
                                 name="next_states")
        self.done_mask = tf.placeholder(tf.bool,
                                        shape=(None),
                                        name="done")
        self.lr = tf.placeholder(tf.float32,
                                 shape=(),
                                 name="learning_rate")


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
            shape = (batch_size, img height, img width, nchannels x .state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # Get output shape
        num_actions = self.env.action_space.shape

        with tf.variable_scope(scope, reuse=reuse):
            out = tf.layers.flatten(state,
                                    name='flatten')
            out = tf.layers.dense(out,
                                  self.config.DIM_DENSE_1,
                                  activation="relu",
                                  name='fc1')
            out = tf.layers.dense(out,
                                  self.config.DIM_DENSE_2,
                                  activation="relu",
                                  name='fc2')
            out = tf.layers.dense(out,
                                  num_actions,
                                  activation="softmax",
                                  name='fc3')

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """

        q_coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        assigned = [tf.assign(t_q_el, q_el) for t_q_el, q_el in zip(target_q_coll, q_coll)]
        self.update_target_op = tf.group(*assigned)


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.shape

        q_samp = self.r + (1.-tf.cast(self.done_mask, tf.float32)) * \
                        self.config.gamma * tf.reduce_max(target_q, axis=1)
        q_sum = tf.reduce_sum(q*tf.one_hot(self.a, num_actions), axis=1)
        self.loss = tf.reduce_mean((q_samp - q_sum)**2)


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm

        Args:
            scope: (string) name of the scope whose variables we are
                   differentiating with respect to
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        gradient, variable = list(zip(*optimizer.compute_gradients(self.loss, var)))

        if self.config.grad_clip:
            gradient, _ = tf.clip_by_global_norm(gradient, self.config.clip_val)

        self.train_op = optimizer.apply_gradients(list(zip(gradient, variable)))
        self.grad_norm = tf.global_norm(gradient)
