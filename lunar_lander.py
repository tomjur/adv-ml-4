#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import gym
import numpy as np
import cPickle as pickle
import timeit
from agent import *


def run_episode(sess, env, agent, movie_mode=False):
    done = False
    obsrv = env.reset()  # Obtain an initial observation of the environment
    states = [obsrv]
    rewards = []
    actions = []
    while not done:
        if movie_mode:
            env.render()
        # Run the policy network and get a distribution over actions
        action_probs = sess.run(agent.single_prediction,
                                feed_dict={agent.observations: np.reshape(obsrv, (-1, obsrv.shape[0]))})
        action_probs = action_probs.astype(np.float64)
        action_probs /= action_probs.sum()  # normalize
        # get most likely
        if movie_mode:
            # in test time we want to take the most probable
            action = np.argmax(action_probs)
        else:
            action = np.argmax(np.random.multinomial(1, action_probs.reshape(-1)))
        # step the environment and get new measurements
        obsrv, reward, done, _ = env.step(action)
        # update states
        states.append(obsrv)
        rewards.append(reward)
        actions.append(action)
    return states, rewards, actions


def compute_future_gains(rewards_for_episode, gamma):
    # given the rewards in the episode and gamma, returns the rewards s.t each place contains the future gain
    reward_per_step = np.array(rewards_for_episode)
    reward_per_step = reward_per_step[::-1]
    next = 0.0
    for i in range(len(reward_per_step)):
        next = reward_per_step[i] + gamma * next
        reward_per_step[i] = next
    return reward_per_step[::-1]


def normalize_rewards(reward_per_step):
    # normalize rewards
    reward_per_step -= reward_per_step.mean()
    rewards_std = reward_per_step.std()
    if rewards_std > 0.0:
        reward_per_step /= rewards_std
    return reward_per_step


def do_movie(sess, env, agent):
    _, rewards, _ = run_episode(sess, env, agent, movie_mode=True)
    print 'movie reward {}'.format(np.sum(rewards))


def run_for_parameter_set(hidden1, hidden2,
                          starting_learning_rate, learning_rate_decay_steps, learning_rate_weight_decrease,
                          total_episodes, episodes_per_update, rewards_discount_factor,
                          print_summary_identification=None):
    env_d = 'LunarLander-v2'
    env = gym.make(env_d)
    env.reset()

    # init agent
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
                states, rewards, actions = run_episode(sess, env, agent, movie_mode=False)
                avg_rewards += np.sum(rewards)
                # transform to future gains
                rewards = compute_future_gains(rewards, rewards_discount_factor)
                # add to memory
                batch_states += states[:-1]  # last state has no action
                batch_actions += actions
                batch_rewards = np.concatenate((batch_rewards, rewards))

            # normalize rewards
            batch_rewards = normalize_rewards(batch_rewards) / episodes_per_update
            avg_rewards /= episodes_per_update

            # apply gradients:
            _, step = sess.run([agent.train_step, agent.global_step], feed_dict={
                agent.observations: np.vstack(batch_states),
                agent.actions_taken: np.stack(batch_actions),
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

    starting_learning_rate = 0.001
    # we didn't use the decay - therefore the decrease is by a factor of 1.0
    learning_rate_decay_steps = 100
    learning_rate_weight_decrease = 1.0

    episodes_per_update = 10
    # episodes_per_update = 5

    total_episodes = 30000
    # total_episodes = 5

    rewards_discount_factor = 0.99

    run_for_parameter_set(hidden1, hidden2,
                          starting_learning_rate, learning_rate_decay_steps, learning_rate_weight_decrease,
                          total_episodes, episodes_per_update, rewards_discount_factor)

if __name__ == '__main__':
    tf.app.run()
