#!/usr/bin/env python3

from data import *
import tensorflow as tf
import numpy as np
import sys
from tensorflow.python import debug as tf_debug

class SparseEncodingModel:

	# Number of neurons per input layer and per output class
	input_features = 160
	output_classes = 1
	log_directory = "logs/"

	def __init__(self, input_data_file = "preprocessed_data/preprocessed_data.csv", 
			train_percent = 90, learning_rate = .000005, dropout_rate = 0.05, 
			batch_size = 512, seed = 1, verbose = False, debug = False):
		self.learning_rate = learning_rate
		self._train_percent = train_percent
		self._debug = debug
		self.data = ProteaseData(self._train_percent, input_data = input_data_file)
		self.batch_size = batch_size
		self.summary_ops = []

		# define placeholders. x is the input data, y are the output labels ("truth" data)

		self.x = tf.placeholder(tf.float32, [None, SparseEncodingModel.input_features], name = 'x-input')
		self.y = tf.placeholder(tf.float32, [None, SparseEncodingModel.output_classes], name = 'y-input')

		# Hidden Layer #1
		self.hidden_layer_1 = tf.layers.dense(self.x, 15, activation = tf.nn.leaky_relu,
			name = 'HiddenLayer1')
		self.dropout_layer_1 = tf.layers.dropout(self.hidden_layer_1, dropout_rate, name = 'Dropout1')
		self.norm_layer_1 = tf.contrib.layers.layer_norm(self.dropout_layer_1)

		# Hidden Layer #2
		self.hidden_layer_2 = tf.layers.dense(self.norm_layer_1, 10, activation = tf.nn.leaky_relu,
			name = 'HiddenLayer2')
		self.dropout_layer_2 = tf.layers.dropout(self.hidden_layer_2, dropout_rate, name = 'Dropout2')
		self.norm_layer_2 = tf.contrib.layers.layer_norm(self.dropout_layer_2)

		# Hidden Layer #3
		self.hidden_layer_3 = tf.layers.dense(self.norm_layer_2, 5, activation = tf.nn.leaky_relu,
			name = 'HiddenLayer3')
		self.dropout_layer_3 = tf.layers.dropout(self.hidden_layer_3, dropout_rate, name = 'Dropout3')
		self.norm_layer_3 = tf.contrib.layers.layer_norm(self.dropout_layer_3)

		# Hidden Layer #4
		self.hidden_layer_4 = tf.layers.dense(self.norm_layer_3, 3, activation = tf.nn.leaky_relu,
			name = 'HiddenLayer4')
		self.dropout_layer_4 = tf.layers.dropout(self.hidden_layer_4, dropout_rate, name = 'Dropout4')
		self.norm_layer_4 = tf.contrib.layers.layer_norm(self.dropout_layer_4)

		# Output Layers
		self.logits = tf.layers.dense(self.norm_layer_4, 1, name = 'Logits')

		self.probabilities = tf.nn.sigmoid(self.logits, name ='Probabilities')
		self.predictions = tf.round(self.probabilities, name = 'Predictions')
		self.prediction_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.y)),
			name = 'Accuracy')

		with tf.name_scope('Loss'):
			self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
				multi_class_labels = self.y, 
				logits = self.logits), name = 'Loss')

		self.summary_ops += tf.summary.histogram("CostHistogram", self.loss)


		# Define optimizer

		self._optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, name =\
				"optimizer").minimize(self.loss)

		self.merged_summary_ops = tf.summary.merge_all()

	def get_training_batch(self, batch_num, batch_size):
		batch_end_index = (batch_num + 1) * batch_size

		if (batch_num + 1) * batch_size > self.data.train_features.shape[0]:
			batch_end_index = data.train_features.shape[0] - batch_num * batch_size

		batch_input = self.data.train_features[batch_num * batch_size:batch_end_index, :]
		batch_labels = self.data.train_labels[batch_num * batch_size:batch_end_index, :]

		return (batch_input, batch_labels)

	def eval_loss(self, evaluation_set):
		evaluation_features, evaluation_labels = self.data.datasets[evaluation_set]

		return self.sess.run(self.loss,
			feed_dict = {
				self.x: evaluation_features, 
				self.y: evaluation_labels
			})

	def eval_accuracy(self, evaluation_set):
		evaluation_features, evaluation_labels = self.data.datasets[evaluation_set]

		return self.sess.run(self.prediction_accuracy,
			feed_dict = {
				self.x: evaluation_features, 
				self.y: evaluation_labels
			})


	def initialize_and_train(self, num_epochs = 100, eval_interval = 200):

		self.sess = tf.Session()
			
		if self._debug:
			self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

		self.log_writer = tf.summary.FileWriter(SparseEncodingModel.log_directory, self.sess.graph, flush_secs = .001)
		self.train_writer = tf.summary.FileWriter('train_logs', self.sess.graph)

		init_global = tf.global_variables_initializer()
		init_local = tf.local_variables_initializer()
		self.sess.run(init_global)
		self.sess.run(init_local)

		print("\nUntrained SparseEncodingModel:")
		print("Training set loss: " + str(self.eval_loss('train')))
		print("Training set accuracy: " + str(self.eval_accuracy('train')))
		print("Dev set loss: " + str(self.eval_loss('dev')))
		print("Dev set accuracy: " + str(self.eval_accuracy('dev')))



		for epoch in range(num_epochs):
			
			num_batches = self.data.train_features.shape[0] // self.batch_size
			for i in range(num_batches):

				batch_features, batch_labels = self.get_training_batch(i, self.batch_size)

				summary, _= self.sess.run([self.merged_summary_ops, self._optimizer],
					feed_dict = {self.x: batch_features, self.y: batch_labels})
				self.train_writer.add_summary(summary, epoch)

			if (epoch + 1) % eval_interval == 0:

				print("\nEpoch " + str(epoch + 1) + ":")
				print("Training set loss: " + str(self.eval_loss('train')))
				print("Training set accuracy: " + str(self.eval_accuracy('train')))
				print("Dev set loss: " + str(self.eval_loss('dev')))
				print("Dev set accuracy: " + str(self.eval_accuracy('dev')))
				
			self.data.shuffle_training_examples()
		
		print("\n--Final trained model--\n")
		print("Epoch " + str(num_epochs) + ":")
		print("Training set loss: " + str(self.eval_loss('train')))
		print("Training set accuracy: " + str(self.eval_accuracy('train')))
		print("Dev set loss: " + str(self.eval_loss('dev')))
		print("Dev set accuracy: " + str(self.eval_accuracy('dev')))
		print("Test set loss: " + str(self.eval_loss('test')))
		print("Test set accuracy: " + str(self.eval_accuracy('test')))






	

if __name__ == "__main__":

	protease_NN = SparseEncodingModel(verbose = True, debug = False, dropout_rate = 0.01)
	protease_NN.initialize_and_train(num_epochs = 20000)