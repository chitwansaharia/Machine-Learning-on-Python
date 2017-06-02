import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")

import tensorflow as tf
import collections
import os
import numpy as np


def _read_words(filename):
	with tf.gfile.GFile(filename, "r") as f:
		return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	id_to_word = {v: k for k, v in word_to_id.iteritems()}

	return word_to_id,id_to_word

def file_to_word_ids(filename,word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id ]

def ptb_raw_data(data_path=None):
	train_data_path = os.path.join(data_path,"ptb.train.txt")
	valid_data_path = os.path.join(data_path,"ptb.valid.txt")
	test_data_path = os.path.join(data_path,"ptb.test.txt")

	word_to_id,id_to_word = _build_vocab(train_data_path)
	train_data = file_to_word_ids(train_data_path,word_to_id)
	valid_data = file_to_word_ids(valid_data_path,word_to_id)
	test_data = file_to_word_ids(test_data_path,word_to_id)

	vocabulary = len(word_to_id)
	data = {}
	data["train_data"] = train_data
	data["test_data"] = test_data
	data["valid_data"] = valid_data
	data["vocab_len"] = vocabulary
	data["id_to_word"] = id_to_word
	data["word_to_id"] = word_to_id
	return data

def itr_generator(config,raw_data, name=None):
    batch_size = config.batch_size
    num_steps = config.num_steps
    raw_data = np.array(raw_data)
    data_len = raw_data.size
    batch_len = data_len // batch_size
    data = np.reshape(raw_data[: batch_size * batch_len],
                      [batch_size, batch_len])

    wts = np.ones((batch_size, num_steps), dtype=np.float32)
    epoch_size = (batch_len - 1) // num_steps

    # input_data = Input_data()
    # input_data.num_batches = epoch_size
    
    epoch_list = np.arange(epoch_size)
    np.random.shuffle(np.arange(epoch_size))
    input_data = {}
    input_data["num_batches"] = epoch_size
    for i in epoch_list:
        s = i * num_steps
        e = s + num_steps

        input_data["inputs"] = data[:, s:e]
        input_data["outputs"] = data[:, s + 1:e + 1]
        yield input_data
