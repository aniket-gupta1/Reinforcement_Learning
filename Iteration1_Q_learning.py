import gym
import time
import numpy as np 
import matplotlib.pyplot as plt 
from collections import defaultdict
import itertools
import sys

env = gym.make("MountainCar-v0")
env.reset()

Discrete_size = [20] * len(env.observation_space.high)
Discrete_obs_space_size = (env.observation_space.high - env.observation_space.low) / Discrete_size

print(env.goal_position)
done = False

def get_discrete_states(state):
	return tuple(((state - env.observation_space.low)/Discrete_obs_space_size).astype(np.int))

def q_learning(env, num_episodes, discount_factor=0.95, alpha=0.1, epsilon=0.1):
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = np.zeros((20,20,3))
	#Q = np.random.uniform(low=-2, high=0, size=(20,20,3))
	#print(Q.shape)
	# Keeps track of useful statistics
	episode_lengths = np.zeros(num_episodes)
	episode_rewards = np.zeros(num_episodes)
	
	for i_episode in range(num_episodes):
		# Print out which episode we're on, useful for debugging.
		if (i_episode + 1) % 1 == 0:
			print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
			sys.stdout.flush()
		
		# Reset the environment
		state = get_discrete_states(env.reset())
		
		# One step in the environment
		# total_reward = 0.0
		for t in itertools.count():
			# Take a step
			rand_prob = np.random.rand()
			if epsilon>rand_prob:
				action = np.random.randint(0, env.action_space.n)
			else:
				action = np.argmax(Q[state])

			next_s, reward, done, _ = env.step(action)
			next_state = get_discrete_states(next_s)
			if i_episode%2000 == 0:
				env.render()

			# Update statistics
			episode_rewards[i_episode] += reward
			episode_lengths[i_episode] = t
			
			# TD Update   
			td_target = reward + discount_factor * np.max(Q[next_state])
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta
			
			if next_s[0] >= env.goal_position:
				print("We made it on episode: {}".format(i_episode))
				sys.stdout.flush()
				Q[state][action] = 0
			if done:
				break
			
				
			state = next_state
	
	return Q, stats

Q, stats = q_learning(env, 25000)

print("Closing environment")
env.close()
