import gym
import sys
import time
import logging
import random
import itertools
import threading
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models
from tensorflow.keras.callbacks import TensorBoard

# Implement multi-threading here
time_var = time.localtime()
name = f'Walker2d-v2 {time_var.tm_mon}_{time_var.tm_mday}_{time_var.tm_hour}_{time_var.tm_min}_{time_var.tm_sec}'
summary_writer = tf.summary.create_file_writer(logdir = f'TD3_logs/{name}/')

def Model(input_shape, output_shape, input_activation, output_activation, hidden_layers):
	model = models.Sequential()

	model.add(layers.Dense(hidden_layers[0], input_shape = input_shape, activation = input_activation))
	model.add(layers.Dense(hidden_layers[1], activation = input_activation))

	model.add(layers.Dense(output_shape, activation = output_activation))

	return model

def copy_weights(Copy_from, Copy_to, constant):
	variables2 = Copy_from.trainable_variables
	variables1 = Copy_to.trainable_variables
	for v1, v2 in zip(variables1, variables2):
		v1.assign((1-constant)*v2.numpy() + constant*v1.numpy())

def return_func(rews, discount):
	n = len(rews)
	rtgs = np.zeros_like(rews, dtype = 'float32')
	for i in reversed(range(n)):
		rtgs[i] = rews[i] + (discount*rtgs[i+1] if i+1 < n else 0)
	return rtgs

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

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

class DDPG():
	"""Class to train a policy network using the DDPG algorithm"""
	def __init__(self, env_name, env, Actor_net, Critic_net1, Critic_net2, act_dim, obs_dim ,lam = 0.97,
				Actor_lr = 0.0001, gamma = 0.99, delta = 0.01, Critic_lr = 0.001, render = False, epoch_steps = 2000, 
				value_train_iterations = 5, memory_size = 100000, polyak_const = 0.995, minibatch_size = 100):
		
		self.env_name = env_name
		self.env = env
		self.gamma = gamma
		self.Actor = Actor_net
		self.Critic1 = Critic_net1
		self.Critic2 = Critic_net2
		self.act_dim = act_dim
		self.obs_dim = obs_dim
		
		self.Actor_optimizer = optimizers.Adam(lr = Actor_lr)
		self.Critic_optimizer1 = optimizers.Adam(lr = Critic_lr)
		self.Critic_optimizer2 = optimizers.Adam(lr = Critic_lr)

		self.render = render
		self.lam = lam
		self.value_train_iterations = value_train_iterations

		self.Experience = namedtuple('Experience', ['states','actions', 'rewards', 'next_states', 'dones'])
		self.memory_size = memory_size
		self.memory = ReplayMemory(self.memory_size)

		self.Target_Actor = models.clone_model(self.Actor)
		self.Target_Critic1 = models.clone_model(self.Critic1)
		self.Target_Critic2 = models.clone_model(self.Critic2)

		self.minibatch_size = minibatch_size
		self.polyak_const = polyak_const
		self.act_limit = self.env.action_space.high[0]
		self.policy_delay = 2

	def select_action(self, state):
		state = np.atleast_2d(state).astype('float32')
		action = self.Actor(state) * self.act_limit
		return action

	def add_noise(self, action, step, decay = False, noise_scale = 0.1):
		"""Function to add OU noise to the action for exploration"""
		if decay:
			noise_scale *= 1/((step+1)**0.5)
		action += noise_scale * np.random.randn(self.act_dim)
		action = np.squeeze(np.clip(action, -self.act_limit, self.act_limit))
		return action

	def close(self):
		"""This function closes all the running environments""" 
		self.env.close()

	def render_episode(self, n=1):
		"""Renders n episodes using the current policy network"""
		for i in range(n):
			state = self.env.reset()
			total_reward = 0
			done = False
			while not done:
				self.env.render()
				action = self.select_action(state)
				next_state, reward, done, _ = self.env.step(action)
				state = next_state
				total_reward += reward
			#print(f"Total Reward accumulated: {total_reward}")

	def load_weights(self, path):
		# Load saved weights to test the algorithm
		self.Actor.load_weights(path)

	def update_critic(self, states, actions, rewards, next_states, dones, step):
		with tf.GradientTape() as critic_tape1, tf.GradientTape() as critic_tape2:
			# Represent the actions suggested by the target Actor network
			target_actions = self.Target_Actor(next_states)*self.act_limit
			# Adding noise to the target actions
			target_actions = self.add_noise(target_actions, step)
			# Represent the target Q values for the target actions by the target Critic network
			target_Q_values1 = self.Target_Critic1(tf.concat([next_states, target_actions], axis = -1))
			target_Q_values2 = self.Target_Critic2(tf.concat([next_states, target_actions], axis = -1))
			#Find minimum of the target Q values
			target_Q_values = tf.minimum(target_Q_values1, target_Q_values2)
			# Bellman backup	
			backup = tf.stop_gradient(rewards + self.gamma*(1-dones)*target_Q_values)
			# Represent the Q values suggested by the Critic Network
			Q_values1 = self.Critic1(tf.concat([states, actions], axis = -1))
			Q_values2 = self.Critic2(tf.concat([states, actions], axis = -1))
			# Loss function to update the critic network
			critic_loss1 = tf.reduce_mean((backup - Q_values1)**2)
			critic_loss2 = tf.reduce_mean((backup - Q_values2)**2)
		
		critic_loss = critic_loss1 + critic_loss2

		critic_gradients1 = critic_tape1.gradient(critic_loss1, self.Critic1.trainable_variables)
		self.Critic_optimizer1.apply_gradients(zip(critic_gradients1, self.Critic1.trainable_variables))
		critic_gradients2 = critic_tape2.gradient(critic_loss2, self.Critic2.trainable_variables)
		self.Critic_optimizer2.apply_gradients(zip(critic_gradients2, self.Critic2.trainable_variables))

		return critic_loss

	def update_actor(self, states, actions, rewards, next_states, dones):
		with tf.GradientTape() as actor_tape:
			# Represents the true actions the Actor network would take
			# Since the actions stored above also contains actions selected due to random noise function N
			true_actions = self.Actor(states)*self.act_limit
			# Represent the true Q_values using the Critic network
			true_Q_values = self.Critic1(tf.concat([states, true_actions], axis = -1))
			# Loss function to update the Actor network
			actor_loss = -tf.reduce_mean(true_Q_values)

		actor_gradients = actor_tape.gradient(actor_loss, self.Actor.trainable_variables)
		self.Actor_optimizer.apply_gradients(zip(actor_gradients, self.Actor.trainable_variables))

		return actor_loss

	def update_target_networks(self):
		# Updating the target Actor and target Critic networks
		copy_weights(self.Actor, self.Target_Actor, self.polyak_const)
		copy_weights(self.Critic1, self.Target_Critic1, self.polyak_const)
		copy_weights(self.Critic2, self.Target_Critic2, self.polyak_const)

	def train_step(self, episode):
		if self.render and episode%1000==0:
			self.render_episode()
		
		# Function to run multiple trajectories using multi-threading
		done = False
		state = self.env.reset()
		total_reward = 0
		actor_losses = np.array([])
		critic_losses = np.array([])
		for t in itertools.count():

			if episode<10:
				action = self.env.action_space.sample()
			else:
				action = self.select_action(state)
				action = self.add_noise(action, t)

			next_state, reward, done, _ = self.env.step(action)
			total_reward += reward
			self.memory.push(self.Experience(state, action, [reward], next_state, [done]))
			state = next_state

			if self.memory.can_provide_sample(self.minibatch_size):
				experiences = self.memory.sample(self.minibatch_size)
				batch = self.Experience(*zip(*experiences))

				states, actions, rewards, next_states, dones = np.asarray(batch[0]).astype('float32'),np.asarray(batch[1]),\
											np.asarray(batch[2]),np.asarray(batch[3]).astype('float32'),np.asarray(batch[4])

				critic_loss = self.update_critic(states, actions, rewards, next_states, dones, t)

				if t%self.policy_delay == 0:
					actor_loss = self.update_actor(states, actions, rewards, next_states, dones)
					self.update_target_networks()
				else:
					actor_loss = 0
					
				# Good old book-keeping
				actor_losses = np.append(actor_losses, actor_loss)
				critic_losses = np.append(critic_losses, critic_loss)

			if done:
				break

		with summary_writer.as_default():
			tf.summary.scalar("reward", total_reward, step=episode)
			tf.summary.scalar("actor_loss", np.mean(actor_losses), step=episode)
			tf.summary.scalar("critic_loss", np.mean(critic_losses), step=episode)

		print(f"Ep:{episode} total_reward:{total_reward:0.2f} critic_loss:{np.mean(critic_losses):0.2f} actor_loss:{np.mean(actor_losses):0.2f}")	

	def train(self, episodes):
		print(f"Starting training, saving checkpoints and logs to: {name}")

		for episode in range(episodes):
			self.train_step(episode)

			if episode%10 == 0 and episode != 0:
				self.Actor.save_weights(f"{name}/Episode{episode}.ckpt")

if __name__ == "__main__":
	env_name = str(sys.argv[1]) if sys.argv[1] else 'CartPole-v0'
	env = gym.make(env_name)
	obs_dim = tf.convert_to_tensor(len(env.observation_space.sample()))
	act_dim = tf.convert_to_tensor(len(env.action_space.sample()))

	# Policy Model
	Actor_net = Model((obs_dim,), len(env.action_space.sample()), 'relu', 'tanh', [400,300])
	#print(Actor_net.summary())

	# Q-value Model
	Critic_net1 = Model((obs_dim+act_dim,), 1, 'relu', 'linear', [400,300])
	Critic_net2 = Model((obs_dim+act_dim,), 1, 'relu', 'linear', [400,300])
	#print(Critic_net.summary())

	agent = DDPG(env_name, env, Actor_net, Critic_net1, Critic_net2, render=True, act_dim = act_dim, obs_dim = obs_dim)
	episodes = int(sys.argv[2]) if sys.argv[2] else int(200)

	if sys.argv[3] == "test":
		path = str(sys.argv[4])
		agent.load_weights(path)
		agent.render_episode(20)
	else:
		agent.train(episodes)

	agent.close()