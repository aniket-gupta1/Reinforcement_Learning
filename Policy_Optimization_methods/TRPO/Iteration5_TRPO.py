import gym
import time
import copy
import logging
import threading
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models
from tensorflow.keras.callbacks import TensorBoard

# Aim: TO implement a object oriented code for TRPO here using this link
# https://github.com/spideralessio/TRPO-Tensorflow2/blob/master/TRPO.py

# Trying the following things here:
# 1. observing the advantage values

# Initialize summary writer object to visualise data in the tensorboard
time_var = time.localtime()
name = f'{time_var.tm_mon}_{time_var.tm_mday}_{time_var.tm_hour}_{time_var.tm_min}_{time_var.tm_sec}'
summary_writer = tf.summary.create_file_writer(logdir = f'TRPO_logs/{name}/')

##@tf.funciton	
def flatgrad(loss_fn, var_list):
	with tf.GradientTape() as tape:
		loss = loss_fn()

	grads = tape.gradient(loss, var_list)
	return tf.concat([tf.reshape(g,[-1]) for g in grads], axis = 0)

#@tf.function
def Model(input_shape, output_shape, input_activation, output_activation, hidden_layers):
	model = models.Sequential()

	#model.add(layers.InputLayer(input_shape = input_shape))
	for i, hidden_layer in enumerate(hidden_layers): 
		if i==0:
			model.add(layers.Dense(hidden_layer, input_shape = input_shape, activation = input_activation))
		else:
			model.add(layers.Dense(hidden_layer, activation = input_activation))

	model.add(layers.Dense(output_shape, activation = output_activation))

	return model

def assign_vars(model, theta):
	shapes = [v.shape.as_list() for v in model.trainable_variables]	
	size_theta = np.sum([np.prod(shape) for shape in shapes])
	# self.assign_weights_op = tf.assign(self.flat_weights, self.flat_wieghts_ph)
	start = 0
	for i, shape in enumerate(shapes):
		size = np.prod(shape)
		param = tf.reshape(theta[start:start + size], shape)
		model.trainable_variables[i].assign(param)
		start += size
	assert start == size_theta, "messy shapes"

def flatvars(model):
	return tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0)

def return_func(rews, discount):
	n = len(rews)
	rtgs = np.zeros_like(rews, dtype = 'float32')
	for i in reversed(range(n)):
		rtgs[i] = rews[i] + (discount*rtgs[i+1] if i+1 < n else 0)
	return rtgs

def normalize(adv):
	g_n = len(adv)
	adv = np.asarray(adv)
	mean = np.mean(adv)
	std = np.std(adv)

	return (adv-mean)/(std+1e-8)

class TRPO(object):
	"""This class implements the Trust Region Policy Optimization algorithm. TRPO gurrantees monotnic
	positive increment in the policy parameters and thus overcomes the drawback of VPG which collapses
	if the optmizer takes a bad step"""
	def __init__(self, env_name, env, policy_net, value_net = None,lam = 0.97, value_lr = 0.01, gamma = 0.99, delta = 0.01, 
					cg_damping = 0.001, cg_iters = 10, residual_tol = 1e-5, backtrack_coeff = 0.6,
					backtrack_iters = 10, render = False,local_steps_per_epoch = 2000, value_train_iterations = 5):
		self.env_name = env_name
		self.envs = []
		self.gamma = gamma
		self.cg_iters = cg_iters
		self.cg_damping = cg_damping
		self.residual_tol = residual_tol
		self.model = policy_net
		self.tmp_model = models.clone_model(self.model)
		self.value_net = value_net
		if self.value_net:
			self.value_optimizer = optimizers.Adam(lr=value_lr)
			self.value_net.compile(self.value_optimizer, loss = losses.MSE)
		self.delta = delta
		self.backtrack_coeff = backtrack_coeff
		self.backtrack_iters = backtrack_iters
		self.render = render
		self.local_steps_per_epoch = local_steps_per_epoch
		self.lam = lam
		self.value_train_iterations = value_train_iterations
		self.N_PATHS = 15
		self.N_THREADS = 2
		for i in range(self.N_PATHS):
			self.envs.append(copy.deepcopy(env))
	
	def close(self):
		self.env.close()


	#Changed
	def __call__(self, state):
		state = np.atleast_2d(state).astype('float32')
		logits = self.model(state)
		action_prob = tf.nn.softmax(logits).numpy().ravel()
		action = np.random.choice(range(action_prob.shape[0]), p=action_prob)
		#action = tf.squeeze(tf.random.categorical(logits, 1))
		return action, action_prob
	
	def render_episode(self, n=1):
		for i in range(n):
			state = self.env.reset()

			while True:
				self.env.render()
				action, _ = self(state)
				#print(f"The action is : {action}")
				next_state, reward, done, _ = self.env.step(action.numpy())
				if done:
					break

	def load_weights(self,path):
		self.model.load_weights(path)
	
	def sample(self, episode):
		obs_all, actions_all, rs_all, action_probs_all, Gs_all = [None]*self.N_PATHS, [None]*self.N_PATHS, [None]*self.N_PATHS, [None]*self.N_PATHS, [None]*self.N_PATHS
		mean_total_reward = [None]*self.N_PATHS

		def generate_path(path):
			entropy = 0
			obs, actions, rs, action_probs, Gs = [], [], [], [], []
			ob = self.envs[path].reset()
			done = False
			
			while not done:
				action, action_prob = self(ob)
				new_ob, r, done, info = self.envs[path].step(action)
				last_action = action
				rs.append(r)
				obs.append(ob)
				actions.append(action)
				action_probs.append(action_prob)
				ob = new_ob
			G = 0
			for r in rs[::-1]:
				G = r + self.gamma*G
				Gs.insert(0, G)
			mean_total_reward[path] = sum(rs)
			entropy = entropy / len(actions)
			obs_all[path] = obs
			actions_all[path] = actions
			rs_all[path] = rs
			action_probs_all[path] = action_probs
			Gs_all[path] = Gs
		
		i = 0
		while i < self.N_PATHS:
			j = 0
			threads = []
			while j < self.N_THREADS and i < self.N_PATHS:
				thread = threading.Thread(target=generate_path, args=(i,))
				thread.start()
				threads.append(thread)
				j += 1
				i += 1
			for thread in threads:
				thread.join()

		mean_total_reward = np.mean(mean_total_reward)
		Gs_all = np.concatenate(Gs_all)
		obs_all = np.concatenate(obs_all)
		rs_all = np.concatenate(rs_all)
		actions_all = np.concatenate(actions_all)
		action_probs_all = np.concatenate(action_probs_all)
		return obs_all, Gs_all, mean_total_reward, actions_all, action_probs_all
	#@tf.function
	def train_step(self,episode, obs, Gs, actions, action_probs, total_reward, t0):
		def surrogate_loss(theta = None):
			if theta is None:
				model = self.model
			else:
				model = self.tmp_model
				assign_vars(self.tmp_model, theta)

			logits = model(np.atleast_2d(obs).astype('float32'))
			action_prob = tf.nn.softmax(logits)
			action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)
			old_logits = self.model(np.atleast_2d(obs).astype('float32'))
			old_action_prob = tf.nn.softmax(old_logits)
			old_action_prob = tf.reduce_sum(actions_one_hot * old_action_prob, axis=1).numpy() + 1e-8
			prob_ratio = action_prob / old_action_prob # pi(a|s) / pi_old(a|s)
			loss = tf.reduce_mean(prob_ratio * advantage)

			return loss
		
		def kl_fn(theta = None):
			if theta is None:
				model = self.model
			else:
				model = self.tmp_model
				assign_vars(self.tmp_model, theta)
			logits = model(np.atleast_2d(obs).astype('float32'))
			action_prob = tf.nn.softmax(logits).numpy() + 1e-8
			old_logits = self.model(np.atleast_2d(obs).astype('float32'))
			old_action_prob = tf.nn.softmax(old_logits)
			return tf.reduce_mean(tf.reduce_sum(old_action_prob * tf.math.log(old_action_prob / action_prob), axis=1))
		
		def hessian_vector_product(p):
			def hvp_fn(): 
				kl_grad_vector = flatgrad(kl_fn, self.model.trainable_variables)
				grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
				return grad_vector_product

			fisher_vector_product = flatgrad(hvp_fn, self.model.trainable_variables).numpy()
			return fisher_vector_product + (self.cg_damping * p)

		def conjugate_grad(Ax, b):
			x = np.zeros_like(b)
			r = b.copy()
			p = r.copy()
			old_p = p.copy()
			r_dot_old = np.dot(r,r)

			for _ in range(self.cg_iters):
				z = Ax(p)
				alpha = r_dot_old / (np.dot(p, z) + 1e-8)
				old_x = x
				x += alpha * p
				r -= alpha * z
				r_dot_new = np.dot(r,r)
				beta = r_dot_new / (r_dot_old + 1e-8)
				r_dot_old = r_dot_new
				if r_dot_old < self.residual_tol:
					break
				old_p = p.copy()
				p = r + beta * p
				if np.isnan(x).any():
					print("x is nan")
					print("z", np.isnan(z))
					print("old_x", np.isnan(old_x))
					print("kl_fn", np.isnan(kl_fn()))
			return x

		def linesearch(x, fullstep):
			fval = surrogate_loss(x)
			for (_n_backtracks, stepfrac) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):
				xnew = x + stepfrac * fullstep
				newfval = surrogate_loss(xnew)
				kl_div = kl_fn(xnew)
				if np.isnan(kl_div):
					print("kl is nan")
					print("xnew", np.isnan(xnew))
					print("x", np.isnan(x))
					print("stepfrac", np.isnan(stepfrac))
					print("fullstep",  np.isnan(fullstep))
				if kl_div <= self.delta and newfval >= 0:
					#print("Linesearch worked at ", _n_backtracks)
					return xnew
				if _n_backtracks == self.backtrack_iters - 1:
					pass#print("Linesearch failed.", kl_div, newfval)
			return x

		# ============================================================================
		# Policy update starts
		Vs = self.value_net(obs).numpy().flatten()
		advantage = Gs - Vs 
		advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
		
		actions_one_hot = tf.one_hot(actions, self.envs[0].action_space.n, dtype="float32")
		policy_loss = surrogate_loss()
		policy_gradient = flatgrad(surrogate_loss, self.model.trainable_variables).numpy()
		
		step_direction = conjugate_grad(hessian_vector_product, policy_gradient)
		shs = .5 * step_direction.dot(hessian_vector_product(step_direction).T)

		lm = np.sqrt(shs / self.delta) + 1e-8
		fullstep = step_direction / lm
		if np.isnan(fullstep).any():
			print("fullstep is nan")
			print("lm", lm)
			print("step_direction", step_direction)
			print("policy_gradient", policy_gradient)
		
		oldtheta = flatvars(self.model).numpy()

		theta = linesearch(oldtheta, fullstep)

		if np.isnan(theta).any():
			print("NaN detected. Skipping update...")
		else:
			assign_vars(self.model, theta)

		kl = kl_fn(oldtheta)
		# Policy update ends

		# Value update using model.fit() method
		history = self.value_net.fit(obs, Gs, epochs=self.value_train_iterations, verbose = 0)
		# history object contains the loss and accuracy metrics after each epoch of training
		value_loss = history.history["loss"][-1]

		print(f"Ep {episode}: Rw_mean: {total_reward:0.2f} VL: {value_loss: 0.2f} KL: {kl: 0.2f}")

		# Maintaning records for visualisation in tensorboard
		if self.value_net:
			with summary_writer.as_default():
				tf.summary.scalar("reward", total_reward, step=episode)
				tf.summary.scalar("value_loss", value_loss, step=episode)
				#tf.summary.scalar("policy_loss", policy_loss, step=episode)

	def train(self, episodes):
		assert self.value_net is not None
		print(f"Starting training, saving checkpoints and logs to: {name}")

		for episode in range(episodes):
			t0 = time.time()
			obs, Gs, avg_reward, actions, action_probs = self.sample(episode)
			#print(f"Sample time: {time.time() - t0}")

			total_loss = self.train_step(episode, obs, Gs, actions, action_probs, avg_reward, t0)

			if episode%10 == 0 and episode != 0 and self.value_net:
				self.model.save_weights(f"{name}/Episode{episode}.ckpt")

if __name__ == "__main__":
	env_name = 'CartPole-v0'
	env = gym.make(env_name)
	policy_model = Model((4,), env.action_space.n, 'tanh', 'linear', [64,64])
	value_net = Model((4,), 1, 'tanh', 'linear', [64,64])

	agent = TRPO(env_name, env, policy_model, value_net, render=False)
	episodes = 200
	agent.train(episodes)
	agent.close()