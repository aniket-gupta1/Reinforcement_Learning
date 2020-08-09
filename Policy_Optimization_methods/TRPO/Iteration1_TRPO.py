import gym
import sys
import time
import math
import logging
import random
import numpy as np
import tensorflow as tf 
from statistics import mean
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from tensorflow.keras.callbacks import TensorBoard
import scipy.signal

# Initialize tensorboard object
time_var = time.localtime()
name = f'{time_var.tm_mon}_{time_var.tm_mday}_{time_var.tm_hour}_{time_var.tm_min}_{time_var.tm_sec}'
#tensorboard = TensorBoard(log_dir = f'/logs/{name}')
summary_writer = tf.summary.create_file_writer(logdir = f'TRPO_logs/{name}/')

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

		if self.module_name == 'policy_net':
			return tf.nn.log_softmax(output)
		elif self.module_name == 'value_net':
			return output


class ReplayMemory():
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
		return tf.squeeze(tf.random.categorical(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32'))), 1))

def normalize(adv):
	g_n = len(adv)
	adv = np.asarray(adv)
	mean = np.mean(adv)
	std = np.std(adv)

	return (adv-mean)/std

# Kindly learn about the conjugate gradient algorithm
def conjugate_gradient(Ax, b):
	x = np.zeros_like(b)
	r = b.copy()
	p = r.copy()
	r_dot_old = np.dot(r,r)

	for _ in range(cg_iters):
		z = Ax(p)
		alpha = r_dot_old / (np.dot(p,z) + 1e-8)
		x += alpha * p
		r -= alpha * z
		r_dot_new = np.dot(r,r)
		p = r + (r_dot_new / r_dot_old) * p
		r_dot_old = r_dot_new
	
	return x

def hessian_vector_product(f, params, x_input):
	g = tf.concat([tf.reshape(x,(-1,)) for x in tf.gradients(params, f)])

	return tf.concat([tf.reshape(x,(-1,)) for x in tf.gradients(params, tf.reduce_sum(g*x_input))])

def step_and_eval(step):
	

if __name__ == "__main__":
	gamma = 0.99
	p_lr = 0.01
	v_lr = 0.001
	lam = 0.97
	epochs = 50
	delta = 0.01
	damping_coeff = 0.1
	cg_iters = 10
	backtrack_iters = 10
	backtrack_coeff = 0.8
	train_value_iterations = 80
	num_episodes = 1000
	local_steps_per_epoch = 2000

	info_shapes = ?? ## Still to be defined

	env = gym.make('CartPole-v0')
	agent = Agent(env.action_space.n)
	Experience = namedtuple('Experience', ['states','actions', 'rewards'])
	temp_Experience = namedtuple('Experience', ['states','actions', 'rewards', 'values'])
	policy_net = Model(len(env.observation_space.sample()), [64,64], env.action_space.n, 'policy_net')
	value_net = Model(len(env.observation_space.sample()), [32], 0, 'value_net')
	memory = ReplayMemory(local_steps_per_epoch)
	temp_memory = ReplayMemory(local_steps_per_epoch)

	optimizer_policy_net = tf.optimizers.Adam(p_lr)
	optimizer_value_net = tf.optimizers.Adam(v_lr)

	# Why define the number of local iterations ? We can also define the number of episode to run
	# and then update the policy paramaeters.
	for epoch in range(epochs):
		state = env.reset()
		done = False
		ep_rewards = []
		returns = []
		advantage = []
		log_probs = []
		avg_rewards = []

		finished_rendering_this_epoch = False
		for t in range(local_steps_per_epoch):

			# To render the gym env once every epoch
			if (not finished_rendering_this_epoch):
				pass #env.render()

			action = agent.select_action(state, policy_net)
			#log_probs = tf.math.reduce_sum(policy_net(np.atleast_2d(np.array(state.reshape(1,-1))).astype('float32')) * tf.one_hot(np.array(action), env.action_space.n), axis=1)
			value = tf.squeeze(value_net(np.atleast_2d(np.array(state.reshape(1,-1))).astype('float32')))

			next_state, reward, done, _ = env.step(action.numpy())
			state = next_state

			memory.push(Experience(state, action, reward))
			temp_memory.push(temp_Experience(state, action, reward, value))
			ep_rewards.append(reward)

			if done or (t+1 == local_steps_per_epoch):
				returns += list(memory.return_func(ep_rewards, gamma))
				temp = temp_Experience(*zip(*temp_memory.memory))
				last_val = 0 if done else tf.squeeze(value_net(np.atleast_2d(np.array(state.reshape(1,-1)).astype('float32'))))

				temp_states, temp_actions, temp_rewards, temp_values = np.asarray(temp[0]),np.asarray(temp[1]),np.asarray(temp[2]),np.asarray(temp[3])
				temp_values = np.append(temp_values, last_val)
				delta = temp_rewards + gamma * temp_values[1:] - temp_values[:-1]
				advantage += list(memory.advantage_func(delta, gamma*lam))
				temp_memory.clear_memory()

				# If trajectory ends and the episode does not, we should bootstrap for the remaining value				
				#memory.update(last_val)
				avg_rewards.append(sum(ep_rewards))
				state, done, ep_rewards = env.reset(), False, []
				finished_rendering_this_epoch = True

		# Updating the policy and value function
		buf = Experience(*zip(*memory.memory))
		states, actions, rewards = np.asarray(buf[0]),np.asarray(buf[1]),np.asarray(buf[2])
		avg_rewards = np.mean(np.asarray(avg_rewards))

		advantage = normalize(advantage)

		for iteration in range(backtrack_iters):
			k_l, a_l_new = set_and_eval(backtrack_coeff**iteration)

			if iteration == backtrack_iters-1:
				k_l, a_l_new = set_and_eval(0.)

		# Training the value function
		with tf.GradientTape() as value_tape:
			value_loss = kls.MSE(returns,tf.squeeze(value_net(np.atleast_2d(np.array(states)).astype('float32'))))

		value_variables = value_net.trainable_variables
		value_gradients = value_tape.gradient(value_loss, value_variables)

		for iteration in range(train_value_iterations):
			optimizer_value_net.apply_gradients(zip(value_gradients, value_variables))
		
		with summary_writer.as_default():
			tf.summary.scalar('Episode_returns', sum(returns), step = epoch)
			tf.summary.scalar('Running_avg_reward', avg_rewards, step = epoch)
			tf.summary.scalar('Losses', policy_loss, step = epoch)

		if epoch%1 == 0:
			print(f"Episode: {epoch} |Losses: {policy_loss: 0.2f}| Return: {sum(returns)}| Avg_reward: {avg_rewards: 0.2f}")
			sys.stdout.flush()

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


