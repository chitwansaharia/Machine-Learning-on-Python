import tensorflow as tf
from create_features import create_feature_list
import csv

train_x,train_y,test_x,test_y,check_x = create_feature_list('train.csv','test.csv')
# print(train_x.shape,train_y.shape,test_x.shape)
# train_x = train_x[0:1000]
# train_y = train_y[0:1000]
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes])),}


	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	hm_epochs = 50
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				epoch_x = train_x[start:end]
				epoch_y = train_y[start:end]
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
				i = i+batch_size
			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

		correct = tf.argmax(prediction, 1)
		correct1 = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct1, 'float'))
		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
		a = correct.eval({x:check_x})
		print('Result:',len(a))
		final_list = []
		final_list.append(('ImageId','Label'))
		j=1
		while j <= len(a):
			final_list.append((j,a[j-1]))
			j = j+1
		with open('final.csv', 'w') as mycsvfile:
			thedatawriter = csv.writer(mycsvfile)
			for row in final_list:
				thedatawriter.writerow(row)




train_neural_network(x)