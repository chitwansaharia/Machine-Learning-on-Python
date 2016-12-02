import numpy;
from sigmoid import *;
def costfunc(Theta1,Theta2,input_level_size,hidden_level_size,output_level_size,m,X,y,lmbda):
	cost = 0
	Theta1_grad = numpy.empty([Theta1.shape[0],Theta1.shape[1]], dtype=int)
	Theta2_grad = numpy.empty([Theta2.shape[0],Theta2.shape[1]], dtype=int)
	temp = X*(Theta1.transpose());
	sigmoid_vec = numpy.vectorize(sigmoid)
	temp = sigmoid_vec(temp)
	temp = numpy.append(numpy.ones([len(temp),1]),temp,1)
	temp = temp*(Theta2.transpose())
	temp = sigmoid_vec(temp)
	y_new = numpy.zeros([m,10])
	i=0
	for y_ele in y:
		y_temp[i,y_ele] = 1
		i = i+1
	log_vec = numpy.vectorize(math.log)
	cost = numpy.multiply(log_vec(temp),y_new) + numpy.multiply(log_vec(1-temp),(1-y_new));
	cost = cost*(-1/m)
	cost = numpy.sum(cost)





