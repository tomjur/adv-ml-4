#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author: daniel
"""
import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf
import timeit


class Agent():
    def __init__(self, env,  hidden1, hidden2, starting_learning_rate, learning_rate_decay_steps,
                 learning_rate_weight_decrease):
        input_space_size = env.observation_space.shape[0]
        action_space_size = env.action_space.n
        global_step = tf.Variable(0, trainable=False)
        tf_learn_rate = tf.train.exponential_decay(starting_learning_rate, global_step, learning_rate_decay_steps,
                                                   learning_rate_weight_decrease, staircase=False)

        # set the placeholder
        self.observation = tf.placeholder(dtype=tf.float64, shape=(input_space_size, ))
        observation_reshaped = tf.reshape(self.observation, (1, input_space_size))
        # network
        w1 = tf.Variable(tf.random_normal([input_space_size, hidden1], dtype=tf.float64))
        b1 = tf.Variable(tf.random_normal([hidden1], dtype=tf.float64))
        h1 = tf.tan(tf.matmul(observation_reshaped, w1) + b1)
        w2 = tf.Variable(tf.random_normal([hidden1, hidden2], dtype=tf.float64))
        b2 = tf.Variable(tf.random_normal([hidden2], dtype=tf.float64))
        h2 = tf.tan(tf.matmul(h1, w2) + b2)
        w3 = tf.Variable(tf.random_normal([hidden2, action_space_size], dtype=tf.float64))
        b3 = tf.Variable(tf.random_normal([action_space_size], dtype=tf.float64))
        self.trainable_params = [w1, w2, w3, b1, b2, b3]
        self.softmax_step = tf.reshape(tf.nn.softmax(tf.matmul(h2, w3) + b3), (action_space_size,))

        # this placeholder will notify the network which action was taken
        self.selected_action_placeholder = tf.placeholder(dtype=tf.int32, shape=())
        # get the log of the gradients of that action with respect to the parameters
        selected_action_softmax = self.softmax_step[self.selected_action_placeholder]
        # since we are maximizing need to negate
        log_selected_action_softmax = tf.log(selected_action_softmax)
        # log_selected_action_softmax = -1.0 * tf.log(selected_action_softmax)
        adam = tf.train.AdamOptimizer(learning_rate=tf_learn_rate)
        gradients_for_action = adam.compute_gradients(loss=log_selected_action_softmax,
                                                           var_list=self.trainable_params)
        assert_w = tf.assert_equal(self.trainable_params[0], gradients_for_action[0][1])
        with tf.control_dependencies([assert_w]):
            self.gradients_for_action = gradients_for_action
        # self.gradients_for_action = tf.gradients(ys=log_selected_action_softmax, xs=self.trainable_params)

        # placeholder for the gradients with the total reward
        self.w1_gradients_placeholder = tf.placeholder(dtype=tf.float64, shape=([input_space_size, hidden1]))
        self.w2_gradients_placeholder = tf.placeholder(dtype=tf.float64, shape=([hidden1, hidden2]))
        self.w3_gradients_placeholder = tf.placeholder(dtype=tf.float64, shape=([hidden2, action_space_size]))
        self.b1_gradients_placeholder = tf.placeholder(dtype=tf.float64, shape=([hidden1]))
        self.b2_gradients_placeholder = tf.placeholder(dtype=tf.float64, shape=([hidden2]))
        self.b3_gradients_placeholder = tf.placeholder(dtype=tf.float64, shape=([action_space_size]))

        max_norm = 1.0
        prepare_inputs_for_optimization = [
            (tf.clip_by_norm(self.w1_gradients_placeholder, max_norm), w1),
            (tf.clip_by_norm(self.w2_gradients_placeholder, max_norm), w2),
            (tf.clip_by_norm(self.w3_gradients_placeholder, max_norm), w3),
            (tf.clip_by_norm(self.b1_gradients_placeholder, max_norm), b1),
            (tf.clip_by_norm(self.b2_gradients_placeholder, max_norm), b2),
            (tf.clip_by_norm(self.b3_gradients_placeholder, max_norm), b3),
        ]
        # prepare_inputs_for_optimization = [
        #     (self.w1_gradients_placeholder, w1),
        #     (self.w2_gradients_placeholder, w2),
        #     (self.w3_gradients_placeholder, w3),
        #     (self.b1_gradients_placeholder, b1),
        #     (self.b2_gradients_placeholder, b2),
        #     (self.b3_gradients_placeholder, b3),
        # ]

        self.train_step = adam.apply_gradients(prepare_inputs_for_optimization, global_step=global_step)


def multiply_gradients_by_scalar(grads, scalar):
    for i in range(len(grads)):
        grads[i] *= scalar
    return grads


def accumulate_gradients(total_grads, current_grads):
    if total_grads is None:
        total_grads = current_grads
    else:
        total_grads = [current_grads[i] + total_grads[i] for i in range(len(current_grads))]
    return total_grads


def get_grads_for_episode(env, sess, agent):
    reward_per_step = []
    grads_per_step = []
    done = False
    obsrv = env.reset()  # Obtain an initial observation of the environment
    while not done:
        # Run the policy network and get a distribution over actions
        action_probs = sess.run(agent.softmax_step, feed_dict={agent.observation: obsrv})
        # sample action from distribution
        action = np.argmax(np.random.multinomial(1, action_probs))
        # step the environment and get new measurements
        new_obsrv, reward, done, info = env.step(action)
        # calculate the new reward
        reward_per_step.append(reward)
        # calculate the resulting gradients
        new_grads = sess.run(agent.gradients_for_action,
                             feed_dict={agent.observation: obsrv,
                                        agent.selected_action_placeholder: action})
        # remove the second part (this is the variable)
        new_grads = [g[0] for g in new_grads]

        # sum the gradients
        grads_per_step.append(new_grads)
        # update the state
        obsrv = new_obsrv

    # calculate the reward of the tail
    reward_per_step = np.array(reward_per_step)
    reward_per_step = reward_per_step[::-1]
    next = 0.0
    for i in range(len(reward_per_step)):
        next = reward_per_step[i] + 0.96 * next
        reward_per_step[i] = next
    # reward_per_step = np.cumsum(reward_per_step)
    reward_per_step = reward_per_step[::-1]

    # # get the sum of rewards
    # sum_rewards = np.sum(reward_per_step)

    # multiply the gradients with the cumulative reward
    grads_for_episode = None
    for t in range(len(grads_per_step)):
        grads_in_t = grads_per_step[t]
        grads_in_t = multiply_gradients_by_scalar(grads_in_t, reward_per_step[t])
        grads_for_episode = accumulate_gradients(grads_for_episode, grads_in_t)
    return grads_for_episode, reward_per_step[0]

    # # multiply the gradients with the cumulative reward
    # grads_for_episode = None
    # for t in range(len(grads_per_step)):
    #     grads_for_episode = accumulate_gradients(grads_for_episode, grads_per_step[t])
    # grads_for_episode = multiply_gradients_by_scalar(grads_for_episode, sum_rewards)

    # return grads_for_episode, reward_per_step[0]
    # return grads_for_episode, sum_rewards


def do_movie(sess, env, agent):
    done = False
    obsrv = env.reset()  # Obtain an initial observation of the environment
    while not done:
        # render the movie
        env.render()
        # Run the policy network and get a distribution over actions
        action_probs = sess.run(agent.softmax_step, feed_dict={agent.observation: obsrv})
        # get most likely
        action = np.argmax(action_probs)
        # step the environment and get new measurements
        obsrv, _, done, _ = env.step(action)



def main(argv):
    hidden1 = 10
    # hidden1 = 15
    hidden2 = 10
    # hidden2 = 15

    starting_learning_rate = 0.001
    learning_rate_decay_steps = 10
    learning_rate_weight_decrease = 1.0

    # episodes_per_update = 10
    episodes_per_update = 3
    total_episodes = 30000

    one_movie_per_updates = 10  # if set to -1 only show at end

    # env_d = 'LunarLander-v2'
    env_d = 'CartPole-v0'
    env = gym.make(env_d)
    env.reset()

    agent = Agent(env=env, hidden1=hidden1, hidden2=hidden2, starting_learning_rate=starting_learning_rate,
                  learning_rate_decay_steps=learning_rate_decay_steps,
                  learning_rate_weight_decrease=learning_rate_weight_decrease)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for update_iteration in range(total_episodes/episodes_per_update):
            print 'update iteration {} out of {}'.format(update_iteration+1, total_episodes/episodes_per_update)

            start_time = timeit.default_timer()
            # this will hold the total changes in the gradients
            total_gradients = None
            # holds the avg rewards
            avg_rewards = 0.0

            for episode in range(episodes_per_update):
                # do a single episode and get the gradients
                grads_for_episode, complete_rewards = get_grads_for_episode(env, sess, agent)
                # aggregate to avg rewards
                avg_rewards += complete_rewards
                # updates the gradients in the total batch update
                total_gradients = accumulate_gradients(total_gradients, grads_for_episode)
            # calc rewards
            avg_rewards /= episodes_per_update
            # divide by episodes_per_update
            total_gradients = multiply_gradients_by_scalar(total_gradients, 1.0 / episodes_per_update)

            # apply gradients:
            sess.run(agent.train_step, feed_dict={
                agent.w1_gradients_placeholder: total_gradients[0],
                agent.w2_gradients_placeholder: total_gradients[1],
                agent.w3_gradients_placeholder: total_gradients[2],
                agent.b1_gradients_placeholder: total_gradients[3],
                agent.b2_gradients_placeholder: total_gradients[4],
                agent.b3_gradients_placeholder: total_gradients[5],
            })

            elapsed = timeit.default_timer() - start_time
            print 'iteration done, time {}, avg rewards {}'.format(elapsed, avg_rewards)

            # show movie if needed
            if one_movie_per_updates != -1 and (update_iteration + 1) % one_movie_per_updates == 0:
                do_movie(sess, env, agent)

        # do final movie
        do_movie(sess, env, agent)


if __name__ == '__main__':
    tf.app.run()
