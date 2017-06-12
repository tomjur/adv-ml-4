import tensorflow as tf

class Agent:
    def do_forward(self, observations):
        # given an arbitrary number of current states (rows of observations),
        # calculate the action probabilities for each state
        h1 = tf.nn.tanh(tf.matmul(observations, self.w1) + self.b1)
        h2 = tf.nn.tanh(tf.matmul(h1, self.w2) + self.b2)
        return tf.nn.softmax(tf.matmul(h2, self.w3) + self.b3)

    def __init__(self, env,  hidden1, hidden2, starting_learning_rate, learning_rate_decay_steps,
                 learning_rate_weight_decrease):
        # environment's parameters
        self.input_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n
        # variable to control the learn rate.
        self.global_step = tf.Variable(0, trainable=False)
        tf_learn_rate = tf.train.exponential_decay(starting_learning_rate, self.global_step, learning_rate_decay_steps,
                                                   learning_rate_weight_decrease, staircase=False)

        # set the placeholders
        self.observations = tf.placeholder(dtype=tf.float32, shape=(None, self.input_space_size))
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.rewards_per_action = tf.placeholder(dtype=tf.float32, shape=(None,))

        # network weights
        self.w1 = tf.get_variable("w1", shape=[self.input_space_size, hidden1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.get_variable("b1", shape=[hidden1], initializer=tf.zeros_initializer())
        self.w2 = tf.get_variable("w2", shape=[hidden1, hidden2],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable("b2", shape=[hidden2], initializer=tf.zeros_initializer())
        self.w3 = tf.get_variable("w3", shape=[hidden2, self.action_space_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = tf.get_variable("b3", shape=[self.action_space_size], initializer=tf.zeros_initializer())

        self.trainable_params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

        # use the following to generate a sequence
        self.single_prediction = self.do_forward(self.observations)

        # for loss: first encode the action as one hot
        one_hot_actions = tf.one_hot(self.actions_taken, self.action_space_size)
        # get the relevant softmax of each step (set to zero the non selected actions and sum)
        relevant_softmax = self.single_prediction * one_hot_actions
        predictions_per_action = tf.reduce_sum(relevant_softmax, axis=1)

        # negative log
        negative_log_per_action = -1.0 * tf.log(
            (tf.ones_like(predictions_per_action) * 0.001) + predictions_per_action
        )

        # multiply by rewards
        loss_with_rewards = negative_log_per_action * self.rewards_per_action
        # train step, assume that rows of all the place holder contain all the state in the batch.
        self.train_step = tf.train.AdamOptimizer(learning_rate=tf_learn_rate).minimize(loss_with_rewards,
                                                                                       global_step=self.global_step)

        # the following allows to load a set of weights
        self.w1_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.input_space_size, hidden1])
        self.b1_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[hidden1])
        self.w2_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[hidden1, hidden2])
        self.b2_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[hidden2])
        self.w3_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[hidden2, self.action_space_size])
        self.b2_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.action_space_size])

        # call the following steps with the above placeholders to initiate with custom weights.
        self.load_w1 = tf.assign(self.w1, self.w1_input_placeholder)
        self.load_w2 = tf.assign(self.w2, self.w2_input_placeholder)
        self.load_w3 = tf.assign(self.w3, self.w3_input_placeholder)
        self.load_b1 = tf.assign(self.b1, self.b1_input_placeholder)
        self.load_b2 = tf.assign(self.b2, self.b2_input_placeholder)
        self.load_b3 = tf.assign(self.b3, self.b3_input_placeholder)
