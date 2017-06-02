# Machine-Learning-on-Python

## Neural Network on Python

A neural network on python built from scratch for training 28*28 images of hand written digits.The model uses gradient 
descent method for cost function optimisation.It consists of 3-layer network with 25 units in the hidden layer.The network
is not yet optimised for training huge data sets.However it works fine on small training sets.

## Word Level Language Modeling using Tensorflow

To train the code on ptb.train.txt (in the data directory) execute the following command :
$ python train_model.py --model=small/medium/large --action=train

To generate new text from a given seed inside seed.txt (in the data directory) execute the following command :
$ python train_model.py --action=generate

