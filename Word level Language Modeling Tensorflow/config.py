class SmallConfig(object):
    init_scale = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 300
    input_size = 300
    keep_prob = 1.0
    batch_size = 64
    vocab_size = 10000

class MediumConfig(object):
	init_scale = 0.3
	max_grad_norm = 10
	num_layers = 3
	num_steps = 30
	hidden_size = 600
	input_size = 300
	keep_prob = 0.5
	batch_size = 100
	vocab_size = 10000

class LargeConfig(object):	
	init_scale = 0.5
	max_grad_norm = 15
	num_layers = 5
	num_steps = 50
	hidden_size = 800
	input_size = 300
	keep_prob = 0.4
	batch_size = 200
	vocab_size = 10000

