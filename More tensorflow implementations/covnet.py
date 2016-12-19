from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
from six.moves import range

with open('notMNIST.pickle', 'rb') as f:
	data = pickle.load(f)

num_classes = 10
image_size = 28
batch_size  = 100

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

train_dataset = data['train_dataset']
train_labels = data['train_labels']
train_dataset, train_labels = randomize(train_dataset, train_labels)
train_dataset = train_dataset.reshape(train_dataset.shape[0],28,28,1)
train_labels = (np.arange(num_classes) == train_labels[:,None]).astype(np.float32)
test_dataset = data['test_dataset']
test_labels = data['test_labels']
test_dataset, test_labels = randomize(test_dataset, test_labels)
test_dataset = test_dataset.reshape(test_dataset.shape[0],28,28,1)
test_labels = (np.arange(num_classes) == test_labels[:,None]).astype(np.float32)
valid_dataset = data['valid_dataset']
valid_labels = data['valid_labels']
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0],28,28,1)
valid_labels = (np.arange(num_classes) == valid_labels[:,None]).astype(np.float32)

# print(train_dataset.shape[0])

graph = tf.Graph()
with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(None, image_size,image_size,1))
	# tf_train_dataset = tf.reshape(tf_train_dataset,[-1,image_size,image_size,1])
	tf_train_labels = tf.placeholder(tf.int32,shape=(None,num_classes))
	# tf_valid_dataset = tf.contant(valid_dataset)
	# tf_valid_labels = tf.contant(valid_labels)
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(tf_train_dataset, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_fin = tf.matmul(h_fc1,W_fc2)+b_fc2
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, tf_train_labels))
	global_step = tf.Variable(0)  # count the number of steps taken.
	learning_rate = tf.train.exponential_decay(0.15, global_step,10000,0.96)
	optimizer = tf.train.GradientDescentOptimizer(0.15).minimize(cross_entropy,global_step=global_step)
	correct_prediction = tf.equal(tf.argmax(y_fin,1), tf.argmax(tf_train_labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_steps = 10000
start = 0
end = batch_size
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialised')
	for step in range(num_steps):
		if(end > 200000):
			start = 0
			end = batch_size
		train_data = train_dataset[start:end,:,:,:]
		train_lab = train_labels[start:end,:]
		feed_dict = {tf_train_dataset : train_data, tf_train_labels : train_lab,keep_prob : 0.50}
		_, l = session.run([optimizer, cross_entropy],feed_dict=feed_dict)
		if (step % 500 == 0):
	  		print('Loss at step %d: %f' % (step, l))
	  		feed_dict = {tf_train_dataset:valid_dataset,tf_train_labels:valid_labels}
	  		predictions = session.run([accuracy],feed_dict =feed_dict)
	  		print(predictions)
		start = start+batch_size
		end = end+batch_size
	feed_dict = {tf_train_dataset:test_dataset,tf_train_labels:test_labels}
	predictions = session.run([accuracy],feed_dict = feed_dict)
	print(predictions)
	

