import numpy;
from sigmoid import *;
from siggrad import *;
def costfunc(Theta1,Theta2,input_level_size,hidden_level_size,output_level_size,m,X,y,lmbda):
	cost = 0
	Theta1_grad = numpy.zeros([Theta1.shape[0],Theta1.shape[1]])
	Theta2_grad = numpy.zeros([Theta2.shape[0],Theta2.shape[1]])
	temp = X*(Theta1.transpose());
	sigmoid_vec = numpy.vectorize(sigmoid)
	siggrad_vec = numpy.vectorize(siggrad)
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
	t1 = Theta1[:,1:]
	t2 = Theta2[:,1:]
	t1 = numpy.multiply(t1,t1);
	t2 = numpy.multiply(t2,t2);
	t1 = numpy.sum(t1)
	t2 = numpy.sum(t2)
	t = t1+t2;
	t = t*(lmbda)/(2*m);
	cost = cost+t

	#############################################################
	for iter in range(0,m):
		a1 = X[iter,:]
		z2 = a1*Theta1.transpose();
		a2 = sigmoid_vec(z2);
		a2 = numpy.append(numpy.ones([len(a2),1]),a2,1)
		z3 = a2*Theta2.transpose();
		a3 = sigmoid_vec(z3);
		del3 = a3 - y_new(iter,:);
		del2 = Theta2.transpose()*del3.transpose();
		del2 = del2[2:,:];
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
	# grad = [Theta1_grad(:) ; Theta2_grad(:)];






