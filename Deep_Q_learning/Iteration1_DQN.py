import gym
import time
import math
import random
import itertools
import numpy as np 
import tensorflow as tf 
from statistics import mean
from collections import deque, namedtuple
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from tensorflow.keras.callbacks import TensorBoard

# Initialize tensorboard object
name = f'DQN_logs_{time.time()}'
summary_writer = tf.summary.create_file_writer(logdir = f'logs/{name}/')

class Model(tf.keras.Model):
	"""
	Subclassing a multi-layered NN using Keras from Tensorflow
	"""
	def __init__(self, num_states, hidden_units, num_actions):
		super(Model, self).__init__() # Used to run the init method of the parent class
		self.input_layer = kl.InputLayer(input_shape = (num_states,))
		self.hidden_layers = []

		for hidden_unit in hidden_units:
			self.hidden_layers.append(kl.Dense(hidden_unit, activation = 'tanh')) # Left kernel initializer
		
		self.output_layer = kl.Dense(num_actions, activation = 'linear')

	@tf.function
	def call(self, inputs, **kwargs):
		x = self.input_layer(inputs)
		for layer in self.hidden_layers:
			x = layer(x)
		output = self.output_layer(x)
		return output

class ReplayMemory():
	"""
	Used to store the experience genrated by the agent over time
	"""
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.push_count = 0

	def push(self, experience):
		if len(self.memory)<self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
	"""
	Decaying Epsilon-greedy strategy
	"""
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)

class DQN_Agent():
	"""
	Used to take actions by using the Model and given strategy.
	"""
	def __init__(self, strategy, num_actions):
		self.current_step = 0
		self.strategy = strategy
		self.num_actions = num_actions

	def select_action(self, state, policy_net):
		rate = self.strategy.get_exploration_rate(self.current_step)
		self.current_step += 1

		if rate > random.random():
			return random.randrange(self.num_actions), rate, True
		else:
			return np.argmax(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), rate, False

def copy_weights(Copy_from, Copy_to):
	"""
	Function to copy weights of a model to other
	"""
	variables2 = Copy_from.trainable_variables
	variables1 = Copy_to.trainable_variables
	for v1, v2 in zip(variables1, variables2):
		v1.assign(v2.numpy())

if __name__ == "__main__":

	# Initialize the parameters
	batch_size = 64
	gamma = 0.99
	eps_start = 1
	eps_end = 0.000
	eps_decay = 0.001
	target_update = 25
	memory_size = 100000
	lr = 0.01
	num_episodes = 1000
	hidden_units = [200,200]

	# Initialize the environment
	env = gym.make('CartPole-v0')

	""" 
	Notice that we are not using any function to make the states discrete here as DQN 
	can handle discrete state spaces.
	"""

	# Initialize Class variables
	strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
	agent = DQN_Agent(strategy, env.action_space.n)
	memory = ReplayMemory(memory_size)

	# Experience tuple variable to store the experience in a defined format
	Experience = namedtuple('Experience', ['states','actions', 'rewards', 'next_states', 'dones'])
	
	# Initialize the policy and target network
	policy_net = Model(len(env.observation_space.sample()), hidden_units, env.action_space.n)
	target_net = Model(len(env.observation_space.sample()), hidden_units, env.action_space.n)
	
	# Copy weights of policy network to target network
	copy_weights(policy_net, target_net)

	optimizer = tf.optimizers.Adam(lr)
	epochs = 10000

	total_rewards = np.empty(epochs)

	for epoch in range(epochs):
		state = env.reset()
		ep_rewards = 0
		losses = []

		for timestep in itertools.count():
			# Take action and observe next_stae, reward and done signal
			action, rate, flag = agent.select_action(state, policy_net)
			next_state, reward, done, _ = env.step(action)
			ep_rewards += reward

			# Store the experience in Replay memory
			memory.push(Experience(state, action, next_state, reward, done))
			state = next_state

			if memory.can_provide_sample(batch_size):
				# Sample a random batch of experience from memory
				experiences = memory.sample(batch_size)
				batch = Experience(*zip(*experiences))

				# batch is a list of tuples, converting to numpy array here
				states, actions, rewards, next_states, dones = np.asarray(batch[0]),np.asarray(batch[1]),np.asarray(batch[3]),np.asarray(batch[2]),np.asarray(batch[4])
				
				# Calculate TD-target
				q_s_a_prime = np.max(target_net(np.atleast_2d(next_states).astype('float32')), axis = 1)
				q_s_a_target = np.where(dones, rewards, rewards+gamma*q_s_a_prime)
				q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')		
			
				# Calculate Loss function and gradient values for gradient descent
				with tf.GradientTape() as tape:
					q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(states).astype('float32')) * tf.one_hot(actions, env.action_space.n), axis=1)
					loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

				# Update the policy network weights using ADAM
				variables = policy_net.trainable_variables
				gradients = tape.gradient(loss, variables)
				optimizer.apply_gradients(zip(gradients, variables))

				losses.append(loss.numpy())

			else:
				losses.append(0)

			# If it is time to update target network
			if timestep%target_update == 0:
				copy_weights(policy_net, target_net)
		
			if done:
				break

		total_rewards[epoch] = ep_rewards 
		avg_rewards = total_rewards[max(0, epoch - 100):(epoch + 1)].mean() # Running average reward of 100 iterations
		
		# Good old book-keeping
		with summary_writer.as_default():
			tf.summary.scalar('Episode_reward', total_rewards[epoch], step = epoch)
			tf.summary.scalar('Running_avg_reward', avg_rewards, step = epoch)
			tf.summary.scalar('Losses', mean(losses), step = epoch)

		if epoch%1 == 0:
			print(f"Episode:{epoch} Episode_Reward:{total_rewards[epoch]} Avg_Reward:{avg_rewards: 0.1f} Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")

	env.close()


