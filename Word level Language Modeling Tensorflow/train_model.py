import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")

import tensorflow as tf
import data_reader
import model
import config
import os
import numpy as np

flags = tf.flags
flags.DEFINE_string("model","small","Possible values are small,medium,large")
flags.DEFINE_string("action","train","Possible values are train,generate")
FLAGS = flags.FLAGS


def get_config():
	print("Using ", FLAGS.model, " configuration")
	if FLAGS.model == "small":
		return config.SmallConfig()
	if FLAGS.model == "medium":
		return config.MediumConfig()
	if FLAGS.model == "large":
		return config.LargeConfig()

def get_action():
	print("Executing %s Action" % FLAGS.action)
	if FLAGS.action == "train":
		return True
	else:
		return False


raw_data = data_reader.ptb_raw_data('data')
seed_text = np.array(data_reader.file_to_word_ids('data/seed.txt',raw_data["word_to_id"]))
seed_text_len = len(seed_text)
seed_text = np.expand_dims(seed_text,0)
print(seed_text.shape)
config = get_config()
config.vocab_size = raw_data["vocab_len"]
config.is_train = get_action()
with tf.Graph().as_default():
	initializer = tf.random_uniform_initializer(-config.init_scale,
												config.init_scale)
	model = model.mymodel(config,raw_data["train_data"],seed_text_len=seed_text_len)
	saver = tf.train.Saver()
	sv = tf.train.Supervisor()
	with sv.managed_session() as session:
		# print("Restoring Model")
		# saver.restore(session, 'weights/best_model.ckpt')
		if config.is_train:
			for i in range(100):
				model.run_epoch(session,is_training = True,
									 verbose=True)
				save_path = saver.save(session, 'weights/best_model.ckpt')
				print("Saving Model in %s" % save_path)
		else:
			model.predict(session,verbose=True,seed_text = seed_text,vocabulary=raw_data["id_to_word"])


