import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")

import time
import numpy as np
import tensorflow as tf
import pdb
import data_reader 

class mymodel(object):
	def __init__(self,config,data,seed_text_len,device= 'gpu'):
		self.config = config
		self.batch_size = config.batch_size
		self.num_steps = config.num_steps
		self.scope = "MyModel"
		self.data = data
		self.seed_text_len = seed_text_len
		self.create_placeholders()
		self.global_step = \
			tf.contrib.framework.get_or_create_global_step()
		self.metrics = {}
		if device == 'gpu':
			tf.device('/gpu:0')
		else:
			tf.device('/cpu:0')
		with tf.variable_scope(self.scope):
			self.build_model()
			if self.config.is_train:
				self.compute_loss_and_metrics()
				self.compute_gradients_and_train_op()

	def create_placeholders(self):
		batch_size = self.config.batch_size
		num_steps = self.config.num_steps
		
		# input data  
		self.x = tf.placeholder(tf.int64,[batch_size,num_steps],name="inputs")
		self.y = tf.placeholder(tf.int64,[batch_size,num_steps],name="outputs")
		self.predict_x = tf.placeholder(tf.int64,[1,self.seed_text_len],name="predict_seed")
		self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")
		self.phase_train = tf.placeholder(tf.bool,name="phase_train")

	def build_model(self):
		is_train = self.config.is_train
		config = self.config
		batch_size = config.batch_size
		num_layers = config.num_layers
		size = config.hidden_size
		vocab_size = config.vocab_size
		num_steps = config.num_steps
		input_size  = config.input_size

		rand_uni_initializer = \
			tf.random_uniform_initializer(
				-self.config.init_scale, self.config.init_scale)
		def attention_cell():
			if is_train:
				return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(size, forget_bias=1.0, state_is_tuple=True),output_keep_prob = self.keep_prob)
			else:
				return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(size, forget_bias=1.0, state_is_tuple=True),output_keep_prob = 1)
		
		cell = tf.contrib.rnn.MultiRNNCell([attention_cell() for _ in range(num_layers)],state_is_tuple=True)
		self.initial_state = cell.zero_state(batch_size, tf.float32)
		self.initial_state_for_predict = cell.zero_state(1,tf.float32)
		embedding = tf.get_variable(
			"embedding", [vocab_size, input_size], dtype=tf.float32)
		if is_train:
			inputs = tf.nn.embedding_lookup(embedding, self.x)
			inputs = tf.nn.dropout(inputs, self.keep_prob)
		else:
			inputs = tf.nn.embedding_lookup(embedding,self.predict_x)

		outputs = []
		if is_train:
			state = self.initial_state
			with tf.variable_scope("RNN", initializer=rand_uni_initializer):
				for time_step in range(num_steps):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(inputs[:, time_step, :], state)
					outputs.append(cell_output)
				self.metrics["final_state"] = state

			
		else:
			state = self.initial_state_for_predict
			with tf.variable_scope("RNN"):
				for time_step in range(self.seed_text_len):
					if time_step >0 :
						tf.get_variable_scope().reuse_variables()
					(cell_output,state) = cell(inputs[:,time_step,:], state)
					outputs.append(cell_output)
					if time_step == 0:
						self.metrics["final_state"] = state

		full_conn_layers = [tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])]
		#full_conn_layers = [tf.stack(outputs, name='stacked_output')]
		with tf.variable_scope("output_layer"):
			self.model_logits = tf.contrib.layers.fully_connected(
				inputs=full_conn_layers[-1],
				num_outputs=vocab_size,
				activation_fn=None,
				weights_initializer=rand_uni_initializer,
				biases_initializer=rand_uni_initializer,
				trainable=True)
			self.metrics["model_prob"] = tf.nn.softmax(self.model_logits)

	def compute_loss_and_metrics(self):
		entropy_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[self.model_logits],
			[tf.reshape(self.y, [-1])],
			[tf.ones([self.batch_size * self.num_steps])],
			average_across_timesteps=False)

		self.metrics["loss"] = tf.reduce_sum(entropy_loss)

	def compute_gradients_and_train_op(self):
		train_vars = self.train_vars = tf.trainable_variables()
		# my_lib.get_num_params(tvars)
		grads = tf.gradients(self.metrics["loss"], train_vars)
		grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

		self.metrics["grad_sum"] = tf.add_n([tf.reduce_sum(g) for g in grads])

		optimizer = tf.train.AdamOptimizer()
		self.train_op = optimizer.apply_gradients(
			zip(grads, train_vars),
			global_step=self.global_step)

	def model_vars():
		return tf.trainable_variables()
	def run_epoch(self, session, is_training=False, verbose=False):
		start_time = time.time()
		epoch_metrics = {}
		fetches = {
			"loss": self.metrics["loss"],
			"grad_sum": self.metrics["grad_sum"],
			"final_state": self.metrics["final_state"],
			"model_prob": self.metrics["model_prob"]
		}


		if is_training:
			if verbose:
				print("\nTraining...")
			fetches["train_op"] = self.train_op
			keep_prob = self.config.keep_prob
			phase_train = True
		else:
			phase_train = False
			keep_prob = 1
			if verbose:
				print("\nEvaluating...")

		# itr = self.data_reader.itr_generator(data)
		i, total_loss, grad_sum, total_words = 0, 0.0, 0.0, 0
		state = session.run(self.initial_state)
		iterator = data_reader.itr_generator(self.config, self.data)
		for item in iterator:
			feed_dict = {}
			feed_dict[self.x.name] = item["inputs"]
			feed_dict[self.y.name] = item["outputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			for j,(c,h) in enumerate(self.initial_state):
				feed_dict[c] = state[j].c
				feed_dict[h] = state[j].h
			# feed_dict[self.initial_state.c] = state.c
			# feed_dict[self.initial_state.h] = state.h

			vals = session.run(fetches, feed_dict)
			state = vals["final_state"]
			total_loss += vals["loss"]
			grad_sum += vals["grad_sum"]
			total_words += self.batch_size*self.num_steps
			i += 1
			percent_complete = (i * 100.0) / item["num_batches"]
			perplexity = np.exp(total_loss / total_words)

			if verbose:
				print(
				"% Complete :", percent_complete,
				"model : perplexity :", round(perplexity, 3),
				)

		return epoch_metrics

	def predict(self, session, vocabulary,seed_text,verbose=False, numwords = 1000):
		start_time = time.time()
		epoch_metrics = {}
		fetches = {
			# "loss": self.metrics["loss"],
			# "grad_sum": self.metrics["grad_sum"],
			"final_state": self.metrics["final_state"],
			"model_prob": self.metrics["model_prob"],

		}
		
		print("Predicting Text")
		output_vec = seed_text
		total_vec = seed_text
		total_vec = total_vec.tolist()
		# print(total_vec)
		# itr = self.data_reader.itr_generator(data)
		i, total_loss, grad_sum, total_words = 0, 0.0, 0.0, 0
		state = session.run(self.initial_state_for_predict)
		for i in range(numwords):
			feed_dict = {}
			feed_dict[self.predict_x.name] = output_vec
			# feed_dict[self.keep_prob.name] = keep_prob
			# feed_dict[self.phase_train.name] = phase_train
			for j,(c,h) in enumerate(self.initial_state_for_predict):
				feed_dict[c] = state[j].c
				feed_dict[h] = state[j].h
			# feed_dict[self.initial_state.c] = state.c
			# feed_dict[self.initial_state.h] = state.h

			vals = session.run(fetches, feed_dict)
			state = vals["final_state"]
			# total_loss += vals["loss"]
			# grad_sum += vals["grad_sum"]
			predicted_vals = vals["model_prob"]
			predicted_vec = predicted_vals[-1,:]
			predicted_vec_val = np.argmax(predicted_vec,0)
			output_vec = np.roll(output_vec,self.seed_text_len-1)
			output_vec[0,self.seed_text_len-1] = predicted_vec_val
			total_vec[0].append(predicted_vec_val)
		# print(total_vec[0])
		if verbose:
			total_vec = [vocabulary[x] for x in total_vec[0]]
			print(total_vec)
			# print(
			# 		" %s " % vocabulary[predicted_vec_val]
			# 	)

		return epoch_metrics
	








