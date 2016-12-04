# Necessary imports
import csv
import numpy
from sigmoid import *;
from siggrad import *;

# Reading training data from csv

# The data is a training set of 28*28 images of hand written digits
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    train_data = list(reader)
train_data = train_data[1:]

# Converting data in processable format
y = list(map((lambda x: x[0]),train_data))
X = list(map((lambda x: [1.0,] + x[1:]),train_data))
X = numpy.asmatrix(X)
X = X.astype('float64');
y = numpy.asmatrix(y)
y = y.astype('int');
print(y)
sigmoid_vec = numpy.vectorize(sigmoid)
siggrad_vec = numpy.vectorize(siggrad)

# Some Global Variables
input_level_size = 784;
hidden_layer_size = 25;
output_layer_size = 10;
m = X.shape[0]
Theta_main = numpy.random.rand(1,25*785+10*26)
Theta_main = numpy.matrix(Theta_main)
lmbda = 10;
alpha = 10;

# Cost function for Neural Network That returns both the cost and the gradient of all the Theta parameters
# The Theta passed as argument is unrolled into the single vector for optimise function to work
def costfunc(Theta):
	Theta = numpy.matrix(Theta)
	Theta1 = Theta[0,0:25*785].reshape(25,785)
	Theta2 = Theta[0,25*785:].reshape(10,26)
	cost = 0
	temp = X*(Theta1.transpose());
	temp = sigmoid_vec(temp)
	temp = numpy.append(numpy.ones([len(temp),1]),temp,1)
	temp = temp*(Theta2.transpose())
	temp = sigmoid_vec(temp)
	y_new = numpy.zeros([m,10])
	i=0
	for y_ele in y:
		y_new[i,y_ele] = 1
		i = i+1
	log_vec = numpy.vectorize(math.log)
	cost = numpy.multiply(log_vec(temp),y_new) + numpy.multiply(log_vec(1-temp),(1-y_new));
	cost = cost*(-1/m)
	cost = numpy.sum(cost)
	t1 = Theta1[:,1:]
	t2 = Theta2[:,1:]
	t1 = numpy.multiply(t1,t1);
	t2 = numpy.multiply(t2,t2);
	t1 = numpy.sum(t1)
	t2 = numpy.sum(t2)
	t = t1+t2;
	t = t*(lmbda)/(2*m);
	cost = cost+t
	Theta1_grad = numpy.zeros([25,785])
	Theta2_grad = numpy.zeros([10,26])
	for iter in range(0,m):
		a1 = X[iter,:]
		z2 = a1*Theta1.transpose();
		a2 = sigmoid_vec(z2);
		a2 = numpy.append(numpy.ones([len(a2),1]),a2,1)
		z3 = a2*Theta2.transpose();
		a3 = sigmoid_vec(z3);
		del3 = a3 - y_new[iter,:];
		del2 = Theta2.transpose()*del3.transpose();
		del2 = del2[1:,:];
		del2 = numpy.multiply(del2,siggrad_vec(z2.transpose()));
		a2 = a2.transpose();
		Theta1_grad = Theta1_grad+del2*a1;
		Theta2_grad = Theta2_grad+del3.transpose()*a2.transpose();
	Theta1_grad = Theta1_grad/m;
	Theta2_grad = Theta2_grad/m;
	temp1 = Theta1*lmbda/m;
	temp2 = Theta2*lmbda/m;
	temp1[:,0] = 0;
	temp2[:,0] = 0;
	Theta1_grad = Theta1_grad + temp1;
	Theta2_grad = Theta2_grad + temp2;
	temp = Theta1_grad.reshape(1,25*785)
	temp1 = Theta2_grad.reshape(1,10*26)
	temp = numpy.append(temp,temp1,1)
	temp = numpy.append(numpy.matrix(cost),temp,1)
	return temp
# Plane gradient descent algorithm for minimizing the cost function.

for i in range(1,300):
	temp = costfunc(Theta_main)
	print(temp[0,0])
	Theta_main = Theta_main - alpha*(temp[0,1:])

# The predict function to check on the test set of data
def predict(Theta,X):
	X = numpy.matrix(X);
	Theta = numpy.matrix(Theta)
	Theta1 = Theta[0,0:25*785].reshape(25,785)
	Theta2 = Theta[0,25*785:].reshape(10,26)
	result = numpy.zeros([1,m])
	for i in range(0,X.shape[0]):
		temp = X[i,:]*Theta1.transpose();
		temp = numpy.append(numpy.matrix(1),temp,1)
		temp = temp*(Theta2.transpose())
		print(temp)
		result[0,i]  = numpy.argmax(temp, axis=1)
	return result

