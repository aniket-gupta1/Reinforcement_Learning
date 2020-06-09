import gym
import itertools
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# Making the observation space discrete
Discrete_size = [20] * len(env.observation_space.high)
Discrete_obs_space_size = (env.observation_space.high - env.observation_space.low) / Discrete_size

def get_discrete_states(state):
	return tuple(((state - env.observation_space.low)/Discrete_obs_space_size).astype(np.int))

def q_learning(env, num_episodes, discount_factor=0.95, alpha=0.1, epsilon=0.1):
	
	# Initialize the Q-table 
	"""
	Q-table contains the expected discounted sum of rewards for each state-action pair
	"""
	Q = np.zeros((20,20,3))

	done = False 								# Initialize the done signal
	episode_lengths = np.zeros(num_episodes)
	episode_rewards = np.zeros(num_episodes)
	
	for i_episode in range(num_episodes):

		# Print out which episode we're on
		if (i_episode + 1) % 1 == 0:
			print(f"\rEpisode {i_episode + 1}/{num_episodes}")
		
		# Reset the environment and observe the starting state
		state = get_discrete_states(env.reset())
		
		# Loop till done signal becomes true
		for t in itertools.count():

			# Select action through epsilon-greedy policy
			rand_prob = np.random.rand()
			if epsilon>rand_prob:
				action = np.random.randint(0, env.action_space.n)
			else:
				action = np.argmax(Q[state])

			# Observe next_state, reward and done signal
			next_state, reward, done, _ = env.step(action)

			# Very critical step!
			state = next_state

			# Make the observed state discrete
			next_state = get_discrete_states(next_state)

			# Render the environment to see the performance of the algorithm
			if i_episode%2000 == 0:
				env.render()

			# Good old book-keeping
			episode_rewards[i_episode] += reward
			episode_lengths[i_episode] = t
			
			# TD Update   
			td_target = reward + discount_factor * np.max(Q[next_state])
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta
			
			# Find out if we actually reached the goal position
			if next_s[0] >= env.goal_position:
				print("We made it on episode: {}".format(i_episode))
				Q[state][action] = 0
			
			if done:
				break

q_learning(env, 25000)

print("Closing environment")
env.close()
