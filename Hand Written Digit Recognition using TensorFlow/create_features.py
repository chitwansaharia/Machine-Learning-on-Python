import numpy as np
import random
import csv
def create_feature_list(train,test,test_size=0.1):
	train_x,train_y,check_x= [],[],[]
	with open(train, 'r') as f:
		reader = csv.reader(f)
		reader = list(reader)
		reader = reader[1:]
		for row in reader:
			train_x.append(row[1:])
			temp_list = np.zeros(10);
			temp_list[row[0]] = 1
			train_y.append(temp_list)
	
	with open(test,'r') as f:
		reader = list(csv.reader(f))
		reader = reader[1:]
		for row in reader:
			check_x.append(row)
	
	feature_x = np.array(train_x)
	feature_y = np.array(train_y)
	testing_size = int(test_size*len(feature_x))

	train_x = list(feature_x[:-testing_size])
	train_y = list(feature_y[:-testing_size])
	test_x = list(feature_x[-testing_size:])
	test_y = list(feature_y[-testing_size:])
	return train_x,train_y,test_x,test_y,check_x


