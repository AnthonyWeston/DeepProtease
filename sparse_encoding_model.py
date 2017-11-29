#!/usr/bin/env python3

from data import ProteaseData
import tensorflow as tf
from tensorflow.python import debug as tf_debug


class SparseEncodingModel:

	# Number of neurons per input layer and per output class
	input_features = 160
	output_classes = 1
	log_directory = "logs/"

	"""
	TODO
	-create python script sparse_encoding_main.py for main program
	-move training to sparse_encoding_main.py
	-define input_data_file, batch_size, log_directory, train_percent in main method
	-Put the following in the main method:
	"""
	def __init__(self, data, name = "Model", learning_rate = .000005, dropout_rate = 0.05, 
		seed = 0 , verbose = False, debug = False):
		self.learning_rate = learning_rate
		self.dropout_rate = dropout_rate
		self.data = data
		self.seed = seed
		self.name = name
		
		self.graph = tf.Graph()
		with self.graph.as_default():
				
			self.sess = tf.Session(graph = self.graph)
		
			if debug:
				self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	
		
			with tf.variable_scope(self.name):
				self.kernel_initializer = tf.contrib.layers.xavier_initializer(seed = seed)
				self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
	
				# define placeholders. x is the input data, y are the output labels ("truth" data)
				self.x = tf.placeholder(tf.float32, [None, SparseEncodingModel.input_features], 
					name = self.name + '_x-input')
				self.y = tf.placeholder(tf.float32, [None, SparseEncodingModel.output_classes], 
					name = self.name + '_y-input')	
	
				# Hidden Layer #1
				self.hidden_layer_1 = tf.layers.dense(self.x, 10, activation = tf.nn.leaky_relu,
					name = self.name + '_HiddenLayer1', kernel_initializer = self.kernel_initializer,
					kernel_regularizer = self.regularizer,
					bias_regularizer = self.regularizer)
				self.dropout_layer_1 = tf.layers.dropout(self.hidden_layer_1, dropout_rate, 
					name = self.name + '_Dropout1')
	
				# Hidden Layer #2
				self.hidden_layer_2 = tf.layers.dense(self.dropout_layer_1, 8, activation = tf.nn.leaky_relu,
					name = self.name + '_HiddenLayer2', kernel_initializer = self.kernel_initializer,
					kernel_regularizer = self.regularizer,
					bias_regularizer = self.regularizer)
				self.dropout_layer_2 = tf.layers.dropout(self.hidden_layer_2, dropout_rate, 
					name = self.name + '_Dropout2')
	
				# Hidden Layer #3
				self.hidden_layer_3 = tf.layers.dense(self.dropout_layer_2, 5, activation = tf.nn.leaky_relu,
					name = self.name + '_HiddenLayer3', kernel_initializer = self.kernel_initializer,
					kernel_regularizer = self.regularizer,
					bias_regularizer = self.regularizer)
				self.dropout_layer_3 = tf.layers.dropout(self.hidden_layer_3, dropout_rate, 
					name = self.name + '_Dropout3')
	
				# Hidden Layer #4
				self.hidden_layer_4 = tf.layers.dense(self.dropout_layer_3, 3, activation = tf.nn.leaky_relu,
					name = self.name + '_HiddenLayer4', kernel_initializer = self.kernel_initializer,
					kernel_regularizer = self.regularizer,
					bias_regularizer = self.regularizer)
				self.dropout_layer_4 = tf.layers.dropout(self.hidden_layer_4, dropout_rate, 
					name = self.name + '_Dropout4')
	
	
				# Output Layers
				
				self.logits = tf.layers.dense(self.dropout_layer_4, 1, name = self.name + '_Logits')
				self.probabilities = tf.nn.sigmoid(self.logits, name = self.name + '_Probabilities')
				self.predictions = tf.round(self.probabilities, name = self.name + '_Predictions')
				self.prediction_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.y)),
					name = self.name + '_Accuracy')
				self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels = self.y, 
					logits = self.logits), name = self.name + '_Loss')
				self.rmse = tf.sqrt(tf.reduce_mean((self.y - self.probabilities) ** 2),
					name = self.name + '_RMSE')
				# Define optimizer
	
				self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, 
					name = self.name + '_Optimizer').minimize(self.loss)
	
				# Define summary operations
	
				self.summary_ops = []
	
				self.summary_ops += tf.summary.scalar(self.name + '_Loss', self.loss)
				self.summary_ops += tf.summary.scalar(self.name + '_Accuracy', self.prediction_accuracy)
				self.summary_ops += tf.summary.scalar(self.name + '_RMSE', self.rmse)
	
				self.merged_summary_ops = tf.summary.merge_all()
				
				# Define summary writers
	
				self.train_writer = tf.summary.FileWriter(SparseEncodingModel.log_directory, self.graph,
					filename_suffix = '.' + self.name + '_train')
				self.test_writer = tf.summary.FileWriter(SparseEncodingModel.log_directory, self.graph,
					filename_suffix = '.' + self.name + '_test')
				
				init_global = tf.global_variables_initializer()
				init_local = tf.local_variables_initializer()
				self.sess.run(init_global)
				self.sess.run(init_local)

	def evaluate_metric(self, evaluation_set, metric):
		evaluation_features, evaluation_labels = self.data.datasets[evaluation_set]
		return self.sess.run(metric,
			feed_dict = {
				self.x: evaluation_features, 
				self.y: evaluation_labels
			})

	def run_summary_ops(self, epoch_number = 0):
		
		feed_dict = {self.x: self.data.train_features, self.y: self.data.train_labels}
		
		train_summary = self.sess.run(self.merged_summary_ops, feed_dict = feed_dict)
		self.train_writer.add_summary(train_summary, epoch_number)


		test_summary = self.sess.run(self.merged_summary_ops, feed_dict = feed_dict)
		self.test_writer.add_summary(test_summary, epoch_number)

	def print_model_metrics(self):
		print(self.name + ":\n")
		print('Training set loss: ' + str(self.evaluate_metric('train', self.loss)))
		print('Training set accuracy: ' + str(self.evaluate_metric('train', self.prediction_accuracy)))
		print('Training set RMSE: '+ str(self.evaluate_metric('train', self.rmse)))
		print('Test set loss: ' + str(self.evaluate_metric('test', self.loss)))
		print('Test set accuracy: ' + str(self.evaluate_metric('test', self.prediction_accuracy)))
		print('Test set RMSE: '+ str(self.evaluate_metric('test', self.rmse)))
		print('')


if __name__ == "__main__":

	data = ProteaseData('sparse', "preprocessed_data/preprocessed_data_sparse.csv", train_percent = 90)
	with tf.Session() as sess:
		model = SparseEncodingModel(data)
