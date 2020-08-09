import cv2
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
		self.input_layer = kl.InputLayer(input_shape = (210,160,4))
		self.num_states = num_states
		self.hidden_layers = []
		self.module_name = module_name

		self.hidden_layers.append(kl.Conv2D(32, (3,3),strides=(4, 4), padding="valid", activation = 'relu'))
		self.hidden_layers.append(kl.MaxPooling2D(pool_size = (2,2)))

		self.hidden_layers.append(kl.Conv2D(64, (3,3), strides=(2, 2), padding="valid", activation = 'relu'))
		self.hidden_layers.append(kl.MaxPooling2D(pool_size = (2,2)))

		self.hidden_layers.append(kl.Conv2D(64, (3,3), strides=(1, 1), padding="valid", activation = 'relu'))
		self.hidden_layers.append(kl.MaxPooling2D(pool_size = (2,2)))

		self.hidden_layers.append(kl.Flatten())

		for hidden_unit in hidden_units:
			self.hidden_layers.append(kl.Dense(hidden_unit, activation = 'relu')) # Left kernel initializer
		
		if module_name == 'policy_net':
			self.output_layer = kl.Dense(num_actions, activation = 'linear')
		elif module_name == 'value_net':
			self.output_layer = kl.Dense(1, activation = 'linear')

	@tf.function
	def call(self, inputs, **kwargs):
		x = self.input_layer(inputs)
		for i,layer in enumerate(self.hidden_layers):
			x = layer(x)
		output = self.output_layer(x)

		return output

class State_Buffer():
	def __init__(self, frame_height, frame_width, total_frames):
		self.total_frames = total_frames
		self.buffer = np.zeros((frame_height, frame_width, total_frames)).astype("uint8")

	def add(self, frame):
		self.buffer[..., :-1] = self.buffer[..., 1:]
		self.buffer[..., -1] = frame

	def reset(self):
		self.buffer *= 0

	def get_state(self):
		return self.buffer

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
		return tf.squeeze(tf.random.categorical(policy_net((state).astype('float32')), 1), axis = 1)

def normalize(adv):
	g_n = len(adv)
	adv = np.asarray(adv)
	mean = np.mean(adv)
	std = np.std(adv)

	return (adv-mean)/std

def convert_to_grayscale(image):
	image = image[:,:,::-1]
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	
if __name__ == "__main__":

	# Initialize the parameters
	gamma = 0.99
	p_lr = 0.01
	v_lr = 0.01
	lam = 0.97
	train_value_iterations = 80
	num_episodes = 1000
	local_steps_per_epoch = 1000
	epochs = 100
	render = False
	render_time = 100

	# Initialize the environment
	env = gym.make('DemonAttack-v0')

	# Initialize Class variables
	agent = Agent(env.action_space.n)
	memory = Memory(local_steps_per_epoch)
	temp_memory = Memory(local_steps_per_epoch)
	buf_ = State_Buffer(210, 160, 4)

	# Experience tuple variable to store the experience in a defined format
	Experience = namedtuple('Experience', ['states','actions', 'rewards'])
	temp_Experience = namedtuple('Experience', ['states','actions', 'rewards', 'values'])

	# Initialize the policy and target network
	policy_net = Model(env.observation_space.sample(), [64,64], env.action_space.n, 'policy_net')
	value_net = Model(env.observation_space.sample(), [32], 0, 'value_net')

	# Optimizers for the models
	optimizer_policy_net = tf.optimizers.Adam(p_lr)
	optimizer_value_net = tf.optimizers.Adam(v_lr)

	# Main Loop
	for epoch in range(epochs):
		# Reset the environment and observe the state
		state = convert_to_grayscale(env.reset())

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
			state = buf_.get_state()
			temp_state = state.reshape(((1,)+state.shape))
			action = agent.select_action(temp_state, policy_net)

			# Find value of the state using the value function
			value = tf.squeeze(value_net(np.atleast_2d(np.array(temp_state).astype('float32'))))

			# Take action and observe next_stae, reward and done signal
			next_state, reward, done, _ = env.step(action.numpy()[0])

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
				last_val = 0 if done else tf.squeeze(value_net(np.array(temp_state).astype('float32')))

				temp_states, temp_actions, temp_rewards, temp_values = np.asarray(temp[0]),np.asarray(temp[1]),np.asarray(temp[2]),np.asarray(temp[3])
				temp_values = np.append(temp_values, last_val)
				
				# Compute TD-target
				delta = temp_rewards + gamma * temp_values[1:] - temp_values[:-1]
				advantage += list(memory.advantage_func(delta, gamma*lam))
				temp_memory.clear_memory()

				avg_rewards.append(sum(ep_rewards))

				# Reset environment to start another trajectory
				state, done, ep_rewards = env.reset(), False, []

			# Critical Step
			state = next_state
			next_state = convert_to_grayscale(next_state)
			buf_.add(next_state)

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
			print(log_probs)
			policy_loss = -tf.math.reduce_mean(advantage * log_probs)
			print(policy_loss)
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


