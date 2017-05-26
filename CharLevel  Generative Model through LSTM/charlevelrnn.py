import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")
sys.path.insert(1,"/u/sahariac/Theano/")
import theano
import theano.tensor as T
import sys
import argparse
import numpy as np
import lasagne
from lasagne.nonlinearities import (sigmoid,linear, )
from lasagne.layers import (ReshapeLayer, DropoutLayer,DenseLayer,
                            ConcatLayer, ElemwiseSumLayer, GaussianNoiseLayer,LSTMLayer,get_output )
import pickle
import os

# Reading data from the input file 

input_data = open("input.txt").read()
input_data_final = []
rev_charmap = dict(enumerate(set(input_data)))
charmap = {v: k for k, v in rev_charmap.iteritems()}
input_data_final = list(map(lambda x: charmap[x],input_data))
# print(input_data_final[0:100])

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))


BATCH_SIZE = 100
SEQ_LEN = 100
VOCABULARY_SIZE = len(set(input_data))
LSTM_DEPTH = 2
NUM_HIDDEN = 100
NUM_EPOCHS = 0

seed_text = input_data[100:100+SEQ_LEN]
print(len(seed_text))

data_size = len(input_data_final)
print(data_size)


def produce_mini_batch(train_data,offset):
	global BATCH_SIZE,SEQ_LEN,VOCABULARY_SIZE
	pointer = offset
	input_minibatch = np.zeros((BATCH_SIZE,SEQ_LEN,VOCABULARY_SIZE))
	output_minibatch = np.zeros((BATCH_SIZE,VOCABULARY_SIZE))
	for i in range(BATCH_SIZE):
		for j in range(SEQ_LEN):
			input_minibatch[i][j][train_data[pointer+j]] = 1
		output_minibatch[i][train_data[pointer+SEQ_LEN]] = 1
		pointer += 1
	return input_minibatch,output_minibatch

def produce_data_point_2(generated_text,offset):
	global SEQ_LEN,VOCABULARY_SIZE
	pointer = offset
	input_minibatch = np.zeros((1,SEQ_LEN,VOCABULARY_SIZE))
	for j in range(SEQ_LEN):
		input_minibatch[0][j][generated_text[pointer+j]] = 1
	return input_minibatch


input_layer = lasagne.layers.InputLayer((None,None,VOCABULARY_SIZE))
hidden_layer = lasagne.layers.LSTMLayer(input_layer, NUM_HIDDEN,nonlinearity=lasagne.nonlinearities.tanh)
hidden_layer = lasagne.layers.LSTMLayer(hidden_layer,NUM_HIDDEN,nonlinearity=lasagne.nonlinearities.tanh,only_return_final = True)

# reshape_layer  = lasagne.layers.ReshapeLayer(hidden_layer,(-1,NUM_HIDDEN))
output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=VOCABULARY_SIZE, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

final_outputs = lasagne.layers.get_output(output_layer)
true_values = T.dmatrix()


# f = theano.function([input_layer.input_var],final_outputs)
# input_temp,target_temp = produce_mini_batch(input_data_final,0)
# print(f(input_temp))
# print(target_temp)

cost = T.nnet.categorical_crossentropy(final_outputs,true_values).mean()

# f = theano.function([input_layer.input_var,true_values],cost)
# input_temp,target_temp = produce_mini_batch(input_data_final,0)
# print(f(input_temp,target_temp))


all_params = lasagne.layers.get_all_params(output_layer,trainable=True)
# pickle.dump(all_params, open("modelweights.pkl", 'wb'))


updates = lasagne.updates.adagrad(cost,all_params,0.03)

train_function = theano.function([input_layer.input_var,true_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([input_layer.input_var, true_values], cost, allow_input_downcast=True)
probs = theano.function([input_layer.input_var],final_outputs,allow_input_downcast=True)

if(os.path.isfile("modelweights.pkl")):
	print("Hello")
	with open("modelweights.pkl","rb") as f:
		layer_params = pickle.load(f)
	print("Printing Parameters")
	# print(layer_params)
	lasagne.layers.set_all_param_values(output_layer,layer_params, trainable=True)

def predict():
	print("Predicting New Text !!")
	generated_text = seed_text
	generated_text_vector = list(map(lambda x: charmap[x],generated_text))
	pointer = 0
	N = 1000
	for i in range(N):
		vector = produce_data_point_2(generated_text_vector,pointer)
		predictions = probs(vector)
		predicted_char = np.argmax(predictions[0])
		# print(predicted_char)
		generated_text += rev_charmap[predicted_char]
		# print(generated_text)
		generated_text_vector.append(predicted_char)
		pointer+=1
	print(generated_text)

predict()

for i in range (NUM_EPOCHS): 
	predict()
	pointer = 0
	costs = []
	batch_number = 0
	while pointer + SEQ_LEN +100 < data_size:
		input_x,target_data = produce_mini_batch(input_data_final,pointer)
		pointer = pointer + BATCH_SIZE
		cost = train_function(input_x,target_data)
		costs.append(train_function(input_x,target_data))
		print("Error at Epoch ",i+1,"and Batch Number ",batch_number," is equal to ",cost)
		batch_number+=1
	print("Mean cost at Epoch ",i+1,"is equal to ",np.mean(costs))
	all_params = lasagne.layers.get_all_param_values(output_layer,trainable=True)
	pickle.dump(all_params, open("modelweights.pkl", 'wb'))


import sys
