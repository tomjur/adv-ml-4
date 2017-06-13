import gym
import numpy as np
import cPickle as pickle
from agent import *
from lunar_lander import run_episode
import re
import os


def eval_single_file(filename, sess, env, agent):
    with open(filename, 'rb') as fp:
        [w1, b1, w2, b2, w3, b3] = pickle.load(fp)
    # init env
    print 'estimating file {}'.format(filename)
    sess.run([agent.load_w1, agent.load_w2, agent.load_w3, agent.load_b1, agent.load_b2, agent.load_b3],
             feed_dict={
                 agent.w1_input_placeholder: w1,
                 agent.w2_input_placeholder: w2,
                 agent.w3_input_placeholder: w3,
                 agent.b1_input_placeholder: b1,
                 agent.b2_input_placeholder: b2,
                 agent.b3_input_placeholder: b3,
             })

    # run episodes
    cumulative_reward_per_episode = [np.sum(run_episode(sess, env, agent)[1]) for i in range(100)]
    min_rewards = np.min(cumulative_reward_per_episode)
    mean_rewards = np.mean(cumulative_reward_per_episode)
    max_rewards = np.max(cumulative_reward_per_episode)
    print 'file {}: min: {} avg: {} max: {}'.format(filename, min_rewards, mean_rewards, max_rewards)
    return min_rewards, mean_rewards, max_rewards


def show_movie_for_file(filename, sess, env, agent):
    with open(filename, 'rb') as fp:
        [w1, b1, w2, b2, w3, b3] = pickle.load(fp)
    # init env
    print 'estimating file {}'.format(filename)
    sess.run([agent.load_w1, agent.load_w2, agent.load_w3, agent.load_b1, agent.load_b2, agent.load_b3],
             feed_dict={
                 agent.w1_input_placeholder: w1,
                 agent.w2_input_placeholder: w2,
                 agent.w3_input_placeholder: w3,
                 agent.b1_input_placeholder: b1,
                 agent.b2_input_placeholder: b2,
                 agent.b3_input_placeholder: b3,
             })
    run_episode(sess, env, agent, movie_mode=True)


def main(argv):
    if len(argv)>1:
        directory = argv[1]
    else:
        directory = (os.path.dirname(os.path.realpath(__file__)))
    print 'using direcotry {}'.format(directory)
    files = [os.path.join(directory,f) for f in os.listdir(directory) if re.match(r'(ws(_.*)?\.p)', f)]

    hidden1 = 15
    hidden2 = 15

    starting_learning_rate = 0.001
    # we didn't use the decay - therefore the decrease is by a factor of 1.0
    learning_rate_decay_steps = 100
    learning_rate_weight_decrease = 1.0

    env_d = 'LunarLander-v2'
    env = gym.make(env_d)
    env.reset()

    # init agent
    agent = Agent(env=env, hidden1=hidden1, hidden2=hidden2, starting_learning_rate=starting_learning_rate,
                  learning_rate_decay_steps=learning_rate_decay_steps,
                  learning_rate_weight_decrease=learning_rate_weight_decrease)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # init agent
        sess.run(init)
        global_rewards = 0.0
        global_best_file = ''
        for f in files:
            min_rewards, mean_rewards, max_rewards = eval_single_file(f, sess, env, agent)
            # we select the model with the best worst performance
            # if global_mean_rewards < mean_rewards:
            #     global_mean_rewards = mean_rewards
            #     global_best_file = f
            if global_rewards < min_rewards:
                global_rewards = min_rewards
                global_best_file = f
        print 'best file is {} with score {}'.format(global_best_file, global_rewards)

        show_movie_for_file(global_best_file, sess, env, agent)


if __name__ == '__main__':
    tf.app.run()
