import math;
from sigmoid import *;
def siggrad(x):
	return sigmoid(x).*(1-sigmoid(x));