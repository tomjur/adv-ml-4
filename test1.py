import gym
env = gym.make('CartPole-v0')
# env = gym.make('LunarLander-v2')

s1 = env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action