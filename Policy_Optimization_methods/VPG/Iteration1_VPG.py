import gym
import time
import numpy as np
import scipy.signal
import tensorflow as tf 
from statistics import mean
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from tensorflow.keras.callbacks import TensorBoard

# Initialize tensorboard object
name = f'VPG_logs_{time.time()}'
summary_writer = tf.summary.create_file_writer(logdir = f'logs/{name}/')

class Model(tf.keras.Model):
	def __init__(self, num_states, hidden_units, num_actions, module_name):
		super(Model, self).__init__() # Used to run the init method of the parent class
		self.input_layer = kl.InputLayer(input_shape = (num_states,))
		self.hidden_layers = []
		self.module_name = module_name

		for hidden_unit in hidden_units:
			self.hidden_layers.append(kl.Dense(hidden_unit, activation = 'tanh')) # Left kernel initializer
		
		if module_name == 'policy_net':
			self.output_layer = kl.Dense(num_actions, activation = 'linear')
		elif module_name == 'value_net':
			self.output_layer = kl.Dense(1, activation = 'linear')

	@tf.function
	def call(self, inputs, **kwargs):
		x = self.input_layer(inputs)
		for layer in self.hidden_layers:
			x = layer(x)
		output = self.output_layer(x)

		# if self.module_name == 'policy_net':
		# 	return tf.nn.log_softmax(output)
		# elif self.module_name == 'value_net':
		return output

class Memory():
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

	def clear_memory(self):
		self.memory = []

	def return_func(self, rews, discount):
		n = len(rews)
		rtgs = np.zeros_like(rews, dtype = 'float32')
		for i in reversed(range(n)):
			rtgs[i] = rews[i] + (discount*rtgs[i+1] if i+1 < n else 0)
		return rtgs

	def advantage_func(self, rews, discount):
		return scipy.signal.lfilter([1], [1, float(-discount)], rews[::-1], axis=0)[::-1]

class Agent():
	def __init__(self, num_actions):
		self.current_step = 0
		self.num_actions = num_actions

	def select_action(self, state, policy_net):
		return tf.squeeze(tf.random.categorical(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32'))), 1), axis = 1)

def normalize(adv):
	g_n = len(adv)
	adv = np.asarray(adv)
	mean = np.mean(adv)
	std = np.std(adv)

	return (adv-mean)/std
	
if __name__ == "__main__":

	# Initialize the parameters
	gamma = 0.99
	p_lr = 0.01
	v_lr = 0.01
	lam = 0.97
	train_value_iterations = 80
	num_episodes = 1000
	local_steps_per_epoch = 2000
	epochs = 300
	render = False
	render_time = 100

	# Initialize the environment
	env = gym.make('CartPole-v0')

	# Initialize Class variables
	agent = Agent(env.action_space.n)
	memory = Memory(local_steps_per_epoch)
	temp_memory = Memory(local_steps_per_epoch)

	# Experience tuple variable to store the experience in a defined format
	Experience = namedtuple('Experience', ['states','actions', 'rewards'])
	temp_Experience = namedtuple('Experience', ['states','actions', 'rewards', 'values'])

	# Initialize the policy and target network
	policy_net = Model(len(env.observation_space.sample()), [64,64], env.action_space.n, 'policy_net')
	value_net = Model(len(env.observation_space.sample()), [32], 0, 'value_net')

	# Optimizers for the models
	optimizer_policy_net = tf.optimizers.Adam(p_lr)
	optimizer_value_net = tf.optimizers.Adam(v_lr)

	# Main Loop
	for epoch in range(epochs):
		# Reset the environment and observe the state
		state = env.reset()
		done = False
		ep_rewards = []
		returns = []
		advantage = []
		log_probs = []
		avg_rewards = []

		for t in range(local_steps_per_epoch):

			# To render environment
			if render and t%render_time == 0:
				env.render()
				
			# Select action using current policy
			action = agent.select_action(state, policy_net)

			# Find value of the state using the value function
			value = tf.squeeze(value_net(np.atleast_2d(np.array(state.reshape(1,-1))).astype('float32')))

			# Take action and observe next_stae, reward and done signal
			next_state, reward, done, _ = env.step(action.numpy()[0])
			
			# Critical Step
			state = next_state

			# Store the data in memory for policy update
			memory.push(Experience(state, action, reward))

			"""
			This variable is used for storing the data till the done signal is true. 
			True done signal marks the end of one episode and since we are collecting 
			multiple trajectories here, we need this variable to calculate the GAE update

			Try to find a better approach here!
			"""
			temp_memory.push(temp_Experience(state, action, reward, value))
			ep_rewards.append(reward)

			if done or (t+1 == local_steps_per_epoch):

				# Compute Rewards to Go
				returns += list(memory.return_func(ep_rewards, gamma))

				temp = temp_Experience(*zip(*temp_memory.memory))

				"""
				This step is critical as in the last trajectory that we are collecting 
				we are not waiting for the episdoe to be over, so we need to bootstrap 
				for the value of the state
				"""
				last_val = 0 if done else tf.squeeze(value_net(np.atleast_2d(np.array(state.reshape(1,-1)).astype('float32'))))

				temp_states, temp_actions, temp_rewards, temp_values = np.asarray(temp[0]),np.asarray(temp[1]),np.asarray(temp[2]),np.asarray(temp[3])
				temp_values = np.append(temp_values, last_val)
				
				# Compute TD-target
				delta = temp_rewards + gamma * temp_values[1:] - temp_values[:-1]
				advantage += list(memory.advantage_func(delta, gamma*lam))
				temp_memory.clear_memory()

				avg_rewards.append(sum(ep_rewards))

				# Reset environment to start another trajectory
				state, done, ep_rewards = env.reset(), False, []

		buf = Experience(*zip(*memory.memory))
		states, actions, rewards = np.asarray(buf[0]),np.asarray(buf[1]),np.asarray(buf[2])
		avg_rewards = np.mean(np.asarray(avg_rewards))

		# This helps to stabilize the training of the model
		advantage = normalize(advantage)

		# Calculate the Policy and Value gradients for gradient descent
		with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
			logits = tf.nn.log_softmax(policy_net(np.atleast_2d(np.array(states)).astype('float32')))

			"""
			Since we selected only one action out of the available ones, we need
			to identify that action using one_hot encoding
			"""
			one_hot_values = tf.squeeze(tf.one_hot(np.array(actions), env.action_space.n))
			log_probs = tf.math.reduce_sum(logits * one_hot_values, axis=1)
			policy_loss = -tf.math.reduce_mean(advantage * log_probs)
			value_loss = kls.MSE(returns,tf.squeeze(value_net(np.atleast_2d(np.array(states)).astype('float32'))))

		policy_variables = policy_net.trainable_variables
		value_variables = value_net.trainable_variables
		policy_gradients = policy_tape.gradient(policy_loss, policy_variables)
		value_gradients = value_tape.gradient(value_loss, value_variables)

		# Update the policy network weights using ADAM
		optimizer_policy_net.apply_gradients(zip(policy_gradients, policy_variables))
		"""
		Since we know the actual rewards that we got, value loss is pretty high.
		So we need to perform multiple iterations of gradient descent to achieve 
		a good performance
		"""
		for iteration in range(train_value_iterations):
			optimizer_value_net.apply_gradients(zip(value_gradients, value_variables))
		
		# Book-keeping
		with summary_writer.as_default():
			tf.summary.scalar('Episode_returns', sum(returns), step = epoch)
			tf.summary.scalar('Running_avg_reward', avg_rewards, step = epoch)
			tf.summary.scalar('Losses', policy_loss, step = epoch)

		if epoch%1 == 0:
			print(f"Episode: {epoch} Losses: {policy_loss: 0.2f} Avg_reward: {avg_rewards: 0.2f}")


	# To render the environment after the training to check how the model performs.
	# You can save the weights for further use using model.save_weights() function from TF2
	render_var = input("Do you want to render the env(Y/N) ?")
	if render_var == 'Y' or render_var == 'y':
		n_render_iter = int(input("How many episodes? "))
		
		for i in range(n_render_iter):
			state = env.reset()
			while True:
				action = agent.select_action(state, policy_net)
				env.render()
				n_state, reward, done, _ = env.step(action.numpy())
				if done:
					break
	else:
		print("Thankyou for using!")

	env.close()


