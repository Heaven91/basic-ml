import gym
from gym import envs
# print ens.registry.all()
env = gym.make('CartPole-v0')
print env.action_space    #print the valid action space of the current environment
observation = env.reset()  # the reset() function returns an observation
for _ in range(1000):
	print observation
	observation, reward, is_terminal, info = env.step(env.action_space.sample())
	env.render()
	if is_terminal:
		print "the goal is reached, env is closed"
		break
