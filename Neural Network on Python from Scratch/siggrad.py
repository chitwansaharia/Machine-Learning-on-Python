import math;
import numpy;
from sigmoid import *;
def siggrad(x):
	return numpy.multiply(sigmoid(x),(1-sigmoid(x)));