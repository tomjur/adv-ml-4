import numpy as np
from lunar_lander import run_for_parameter_set

import sys

if __name__ == '__main__':

    name = sys.argv[1]

    hidden1 = np.random.choice([10, 12, 15, 18, 20])
    hidden2 = np.random.choice([10, 12, 15, 18, 20])
    starting_learning_rate = np.power(10.0, np.random.choice([-2, -3, -4, -5]))
    learning_rate_decay_steps = np.random.choice([5, 10, 20, 100])
    learning_rate_weight_decrease = np.random.choice([0.96, 0.98, 0.99, 1.0])
    episodes_per_update = np.random.choice([3, 5, 10, 20, 100])
    # total_episodes = 10
    total_episodes = 30000
    rewards_discount_factor = np.random.choice([0.96, 0.98, 0.99, 1.0])
    one_movie_per_updates = -1

    run_for_parameter_set(hidden1, hidden2,
                          starting_learning_rate, learning_rate_decay_steps, learning_rate_weight_decrease,
                          total_episodes, episodes_per_update, rewards_discount_factor,
                          one_movie_per_updates, print_summary_identification=name)