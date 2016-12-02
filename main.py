# Reading the data in csv
import csv
import numpy
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    train_data = list(reader)
train_data = train_data[1:]
y = list(map((lambda x: x[0]),train_data))
X = list(map((lambda x: [1,] + x[1:]),train_data))
X = numpy.asmatrix(X);
y = numpy.asmatrix(y);

# Some Global Variables
input_level_size = 784;
hidden_layer_size = 25;
output_layer_size = 10;
m = X.shape[0]
Theta1 = numpy.empty([25,785], dtype=int)
Theta2 = numpy.empty([10,26], dtype=int)
