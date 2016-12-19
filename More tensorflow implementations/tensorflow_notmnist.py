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


url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

# def download_progress_hook(count, blockSize, totalSize):
# 	global last_percent_reported
# 	percent = int(count * blockSize * 100 / totalSize)

# 	if last_percent_reported != percent:
# 		if percent % 5 == 0:
# 			sys.stdout.write("%s%%" % percent)
# 			sys.stdout.flush()
# 		else:
# 			sys.stdout.write(".")
# 			sys.stdout.flush()
  
# 	last_percent_reported = percent
		
# def maybe_download(filename, expected_bytes, force=False):
# 	if force or not os.path.exists(filename):
# 		print('Attempting to download:', filename) 
# 		filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
# 		print('\nDownload Complete!')
# 	statinfo = os.stat(filename)
# 	if statinfo.st_size == expected_bytes:
# 		print('Found and verified', filename)
# 	else:
# 		raise Exception(
# 		'Failed to verify ' + filename + '. Can you get to it with a browser?')
# 	return filename

# train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
# test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
# np.random.seed(133)

# def maybe_extract(filename, force=False):
# 	root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
# 	if os.path.isdir(root) and not force:
# 		print('%s already present - Skipping extraction of %s.' % (root, filename))
# 	else:
# 		print('Extracting data for %s. This may take a while. Please wait.' % root)
# 		tar = tarfile.open(filename)
# 		sys.stdout.flush()
# 		tar.extractall()
# 		tar.close()
# 	data_folders = [
# 		os.path.join(root, d) for d in sorted(os.listdir(root))
# 		if os.path.isdir(os.path.join(root, d))]
# 	if len(data_folders) != num_classes:
# 		raise Exception(
# 		'Expected %d folders, one per class. Found %d instead.' % (
# 			num_classes, len(data_folders)))
# 	print(data_folders)
# 	return data_folders
  
# train_folders = maybe_extract(train_filename)
# test_folders = maybe_extract(test_filename)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

# def load_letter(folder, min_num_images):
#   image_files = os.listdir(folder)
#   dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
#                          dtype=np.float32)
#   print(folder)
#   num_images = 0
#   for image in image_files:
#     image_file = os.path.join(folder, image)
#     try:
#       image_data = (ndimage.imread(image_file).astype(float) - 
#                     pixel_depth / 2) / pixel_depth
#       if image_data.shape != (image_size, image_size):
#         raise Exception('Unexpected image shape: %s' % str(image_data.shape))
#       dataset[num_images, :, :] = image_data
#       num_images = num_images + 1
#     except IOError as e:
#       print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
	
#   dataset = dataset[0:num_images, :, :]
#   if num_images < min_num_images:
#     raise Exception('Many fewer images than expected: %d < %d' %
#                     (num_images, min_num_images))
	
#   print('Full dataset tensor:', dataset.shape)
#   print('Mean:', np.mean(dataset))
#   print('Standard deviation:', np.std(dataset))
#   return dataset
		
# def maybe_pickle(data_folders, min_num_images_per_class, force=False):
#   dataset_names = []
#   for folder in data_folders:
#     set_filename = folder + '.pickle'
#     dataset_names.append(set_filename)
#     if os.path.exists(set_filename) and not force:
#       # You may override by setting force=True.
#       print('%s already present - Skipping pickling.' % set_filename)
#     else:
#       print('Pickling %s.' % set_filename)
#       dataset = load_letter(folder, min_num_images_per_class)
#       try:
#         with open(set_filename, 'wb') as f:
#           pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
#       except Exception as e:
#         print('Unable to save data to', set_filename, ':', e)
  
#   return dataset_names

# train_datasets = maybe_pickle(train_folders, 45000)
# test_datasets = maybe_pickle(test_folders, 1800)
with open('notMNIST.pickle', 'rb') as f:
	data = pickle.load(f)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset = data['train_dataset']
train_labels = data['train_labels']
train_dataset, train_labels = randomize(train_dataset, train_labels)
train_dataset = train_dataset.reshape(train_dataset.shape[0],784)
train_labels = (np.arange(num_classes) == train_labels[:,None]).astype(np.float32)
test_dataset = data['test_dataset']
test_labels = data['test_labels']
test_dataset, test_labels = randomize(test_dataset, test_labels)
test_dataset = test_dataset.reshape(test_dataset.shape[0],784)
test_labels = (np.arange(num_classes) == test_labels[:,None]).astype(np.float32)
valid_dataset = data['valid_dataset']
valid_labels = data['valid_labels']
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0],784)
valid_labels = (np.arange(num_classes) == valid_labels[:,None]).astype(np.float32)

print(train_dataset.shape)

batch_size = 200
hidden_layer_size_1 = 1024

graph = tf.Graph()
with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.int32,shape=(batch_size,num_classes))
	tf_test_dataset = tf.constant(test_dataset)
	tf_test_labels = tf.constant(test_labels)
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_valid_labels = tf.constant(valid_labels)
	keep_prob = tf.placeholder(tf.float32)
	w_1 = tf.Variable(tf.truncated_normal([image_size*image_size,hidden_layer_size_1],stddev = 0.01))
	b_1 = tf.Variable(tf.zeros([hidden_layer_size_1]))
	l_1 = tf.nn.relu(tf.matmul(tf_train_dataset,w_1)+b_1)
	# l_1 = tf.nn.dropout(l_1,keep_prob)
	w_2 = tf.Variable(tf.truncated_normal([hidden_layer_size_1,hidden_layer_size_1],stddev = 0.01))
	b_2 = tf.Variable(tf.zeros([hidden_layer_size_1]))
	l_2 = tf.nn.relu(tf.matmul(l_1,w_2)+b_2)
	# l_2 = tf.nn.dropout(l_2,keep_prob)
	w_3 = tf.Variable(tf.truncated_normal([hidden_layer_size_1,num_classes],stddev = 0.01))
	b_3 = tf.Variable(tf.zeros([num_classes]))
	l_3 = tf.matmul(l_2,w_3)+b_3
	beta = tf.placeholder(tf.float32)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l_3,tf_train_labels)+beta*tf.nn.l2_loss(w_1)+beta*tf.nn.l2_loss(b_1)+beta*tf.nn.l2_loss(w_2)+beta*tf.nn.l2_loss(b_2)+beta*tf.nn.l2_loss(w_3)+beta*tf.nn.l2_loss(b_3))
	global_step = tf.Variable(0)  # count the number of steps taken.
	learning_rate = tf.train.exponential_decay(0.15, global_step,10000,0.96)
	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss,global_step=global_step)
	train_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_train_dataset,w_1)+b_1),w_2)+b_2),w_3)+b_3)
	valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,w_1)+b_1),w_2)+b_2),w_3)+b_3)
	test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,w_1)+b_1),w_2)+b_2),w_3)+b_3)

num_steps = 10000
def accuracy(prediction,labels):
	return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))/prediction.shape[0])

start = 0
end = batch_size
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialised')
	for step in range(num_steps):
		if(end > train_dataset.shape[0]):
			start = 0
			end = batch_size
		train_data = train_dataset[start:end,:]
		train_lab = train_labels[start:end,:]
		feed_dict = {tf_train_dataset : train_data, tf_train_labels : train_lab,keep_prob : 0.50,beta : 0}
		_, l, predictions = session.run([optimizer, loss, train_prediction],feed_dict=feed_dict)
		if (step % 500 == 0):
	  		print('Loss at step %d: %f' % (step, l))
	  		print('Training accuracy: %.1f%%' % accuracy(predictions, train_lab))
	  		print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
		start = start+batch_size
		end = end+batch_size
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
