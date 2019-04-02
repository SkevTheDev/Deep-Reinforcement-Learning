import gym

# create cartpole environment
e = gym.make('CartPole-v0')

# reset cartpole environment
obs = e.reset()
print(obs)

# prints discrete(2) for pushing left of right
print(e.action_space)
# prints box(4,) for the 3 observation points
print(e.observation_space)


