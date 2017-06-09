#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf
import timeit


class Agent:
    def do_forward(self, observations):
        # observation_reshaped = tf.reshape(observations, (-1, self.input_space_size))
        h1 = tf.nn.tanh(tf.matmul(observations, self.w1) + self.b1)
        h2 = tf.nn.tanh(tf.matmul(h1, self.w2) + self.b2)
        return tf.nn.softmax(tf.matmul(h2, self.w3) + self.b3)

    def __init__(self, env,  hidden1, hidden2, starting_learning_rate, learning_rate_decay_steps,
                 learning_rate_weight_decrease):
        self.input_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n
        self.global_step = tf.Variable(0, trainable=False)
        tf_learn_rate = tf.train.exponential_decay(starting_learning_rate, self.global_step, learning_rate_decay_steps,
                                                   learning_rate_weight_decrease, staircase=False)

        # set the placeholders
        self.observations = tf.placeholder(dtype=tf.float32, shape=(None, self.input_space_size))
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.rewards_per_action = tf.placeholder(dtype=tf.float32, shape=(None,))

        # network
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

        # # verify that all the first dimensions are batch sizes
        # observation_batch = tf.constant(self.observations.get_shape()[0], tf.int32, (1,))
        # action_batch = tf.constant(self.actions_taken.get_shape()[0], tf.int32, (1,))
        # rewards_batch = tf.constant(self.rewards_per_action.get_shape()[0], tf.int32, (1,))
        #
        # assert_sizes_1 = tf.assert_equal(observation_batch, action_batch)
        # assert_sizes_2 = tf.assert_equal(observation_batch, rewards_batch)
        # with tf.control_dependencies([assert_sizes_1, assert_sizes_2]):

        # for loss: first encode the action as one hot
        one_hot_actions = tf.one_hot(self.actions_taken, self.action_space_size)
        # one_hot_actions = tf.Print(one_hot_actions, [self.actions_taken[0], one_hot_actions[0,:]], summarize=4)
        # get the relevant softmax of each step
        relevant_softmax = self.single_prediction * one_hot_actions
        # relevant_softmax=tf.Print(relevant_softmax,
        # [self.single_prediction[0,:], one_hot_actions[0,:], relevant_softmax[0,:]], summarize=4)
        predictions_per_action = tf.reduce_sum(relevant_softmax, axis=1)

        # negative log
        negative_log_per_action = -1.0 * tf.log(predictions_per_action)
        # negative_log_per_action = tf.log(predictions_per_action)

        # init optimizer
        adam = tf.train.AdamOptimizer(learning_rate=tf_learn_rate)
        # adam = tf.train.GradientDescentOptimizer(learning_rate=tf_learn_rate)
        # adam = tf.train.RMSPropOptimizer(learning_rate=tf_learn_rate)
        # compute and apply the batch gradients
        grads_and_vars = adam.compute_gradients(loss=negative_log_per_action, var_list=self.trainable_params,
                                                grad_loss=self.rewards_per_action)
        self.train_step = adam.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.global_step)

        # # negative log
        # negative_log_per_action = -1.0 * tf.log(predictions_per_action)
        # # times the rewards
        #
        # # init optimizer
        # adam = tf.train.AdamOptimizer(learning_rate=tf_learn_rate)
        # # compute and apply the batch gradients
        # grads_and_vars = adam.compute_gradients(loss=negative_log_per_action, var_list=self.trainable_params,
        #                                         grad_loss=self.rewards_per_action)
        # self.train_step = adam.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.global_step)


def run_episode(sess, env, agent, render=False):
    done = False
    obsrv = env.reset()  # Obtain an initial observation of the environment
    states = [obsrv]
    rewards = []
    actions = []
    while not done:
        if render:
            env.render()
        # Run the policy network and get a distribution over actions
        action_probs = sess.run(agent.single_prediction,
                                feed_dict={agent.observations: np.reshape(obsrv, (-1, obsrv.shape[0]))})
        action_probs = action_probs.astype(np.float64)
        action_probs /= action_probs.sum()  # normalize
        # get most likely
        action = np.argmax(action_probs)
        # step the environment and get new measurements
        obsrv, reward, done, _ = env.step(action)
        # update states
        states.append(obsrv)
        rewards.append(reward)
        actions.append(action)
    return states, rewards, actions


def compute_future_gains(rewards_for_episode, gamma):
    reward_per_step = np.array(rewards_for_episode)
    reward_per_step = reward_per_step[::-1]
    next = 0.0
    for i in range(len(reward_per_step)):
        next = reward_per_step[i] + gamma * next
        reward_per_step[i] = next
    # reward_per_step = np.cumsum(reward_per_step)
    return reward_per_step[::-1]


def normalize_rewards_for_batch(reward_per_step):
    # normalize rewards
    reward_per_step -= reward_per_step.mean()
    rewards_std = reward_per_step.std()
    if rewards_std > 0.0:
        reward_per_step /= rewards_std
    return reward_per_step


def do_movie(sess, env, agent):
    _, rewards, _ = run_episode(sess, env, agent, render=True)
    print 'movie reward {}'.format(np.sum(rewards))


def run_for_parameter_set(hidden1, hidden2,
                          starting_learning_rate, learning_rate_decay_steps, learning_rate_weight_decrease,
                          total_episodes, episodes_per_update, rewards_discount_factor,
                          print_summary_identification=None):
    env_d = 'LunarLander-v2'
    # env_d = 'CartPole-v0'
    env = gym.make(env_d)
    env.reset()

    agent = Agent(env=env, hidden1=hidden1, hidden2=hidden2, starting_learning_rate=starting_learning_rate,
                  learning_rate_decay_steps=learning_rate_decay_steps,
                  learning_rate_weight_decrease=learning_rate_weight_decrease)
    init = tf.global_variables_initializer()
    initial_start_time = timeit.default_timer()

    with tf.Session() as sess:
        sess.run(init)
        for update_iteration in range(total_episodes / episodes_per_update):
            print 'update iteration {} out of {}'.format(update_iteration + 1, total_episodes / episodes_per_update)

            start_time = timeit.default_timer()

            batch_states = []
            batch_actions = []
            batch_rewards = np.zeros([0])

            avg_rewards = 0.0
            for episode in range(episodes_per_update):
                # do a single episode and get the states, rewards and actions
                states, rewards, actions = run_episode(sess, env, agent, render=False)
                avg_rewards += np.sum(rewards)
                # transform to future gains
                rewards = compute_future_gains(rewards, rewards_discount_factor)
                # add to memory
                batch_states += states[:-1]  # last state has no action
                batch_actions += actions
                batch_rewards = np.concatenate((batch_rewards, rewards))

            # normalize rewards
            batch_rewards = normalize_rewards_for_batch(batch_rewards) / episodes_per_update
            avg_rewards /= episodes_per_update

            # apply gradients:
            _, step = sess.run([agent.train_step, agent.global_step], feed_dict={
                agent.observations: np.vstack(batch_states),
                agent.actions_taken: np.stack(batch_actions),
                # agent.rewards_per_action: np.reshape(batch_rewards, (-1, 1)),
                agent.rewards_per_action: batch_rewards,
            })

            current_time = timeit.default_timer()
            elapsed = current_time - start_time
            elapsed_from_start = current_time - initial_start_time
            print 'iteration done, global_step {}, time {}, from start {}, avg rewards {}'\
                .format(step, elapsed, elapsed_from_start, avg_rewards)

        # save parameters:
        params_to_save = sess.run(agent.trainable_params)
        params_filename = 'ws.p' if print_summary_identification is None \
            else 'ws_{}.p'.format(print_summary_identification)
        with open(params_filename, 'wb') as fp:
            pickle.dump(params_to_save, fp)

        # do final movie - if not summary
        if print_summary_identification is None:
            do_movie(sess, env, agent)
        else:
            # use the parameter as a prefix for the file
            filename = '{}.txt'.format(print_summary_identification)
            avg_rewards = np.sum([run_episode(sess, env, agent) for _ in range(10)]) / 10.0
            with open(filename, 'w') as summary_file:
                def write_to_file(name, value):
                    summary_file.write('{} : {}\n'.format(name, value))

                write_to_file('avg_rewards', avg_rewards)
                summary_file.write('\n')
                write_to_file('hidden1', hidden1)
                write_to_file('hidden2', hidden2)
                write_to_file('starting_learning_rate', starting_learning_rate)
                write_to_file('learning_rate_decay_steps', learning_rate_decay_steps)
                write_to_file('learning_rate_weight_decrease', learning_rate_weight_decrease)
                write_to_file('total_episodes', total_episodes)
                write_to_file('episodes_per_update', episodes_per_update)
                write_to_file('rewards_discount_factor', rewards_discount_factor)


def main(argv):
    # hidden1 = 8
    # hidden2 = 8
    hidden1 = 15
    hidden2 = 15

    starting_learning_rate = 0.01
    learning_rate_decay_steps = 100
    learning_rate_weight_decrease = 1.0

    episodes_per_update = 10
    # episodes_per_update = 5

    total_episodes = 30000
    # total_episodes = 5

    rewards_discount_factor = 1.00

    run_for_parameter_set(hidden1, hidden2,
                          starting_learning_rate, learning_rate_decay_steps, learning_rate_weight_decrease,
                          total_episodes, episodes_per_update, rewards_discount_factor)

if __name__ == '__main__':
    tf.app.run()
