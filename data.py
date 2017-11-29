#!/usr/bin/env python3

import numpy as np;

class ProteaseData:

	num_features_map = {'sparse': 160, 'numeric':8}
	num_labels = 1


	def __init__(self, encoding_type, input_data, train_percent = 90, 
		batch_size = 512, verbose = False, seed = 0):

		self.num_features = ProteaseData.num_features_map[encoding_type]
		self.batch_size = batch_size
		self.train_percent = train_percent

		'''
		Shuffle the training examples, using seed as the initial seed
		for numpy's random number generator so we have reproducible results. Then transpose
		the matrix back to get it in the right shape.
		'''
		all_data = np.genfromtxt(input_data, delimiter = ',', skip_header = 0)
		np.random.seed(seed)
		np.random.shuffle(all_data)


		'''
		Each training/test example is now a column in the matrix.
		'''

		self.train_size = int(self.train_percent / 100 * all_data.shape[0])

		if verbose:
			print(all_data.shape)
			print('Training set size: ' + str(self.train_size))

		self._train_data, self._test_data = np.array_split(all_data, (self.train_size,), axis = 0)

		
		'''
		self.train_features and self.train_labels are views of self.train_data, and the views will
		change when the underlying data is shuffled. This applies similary to and test 
		features and labels, although the underlying data shouldn't be shuffled.
		'''


		self.train_features = self._train_data[:, :self.num_features]
		self.train_labels = self._train_data[:, self.num_features:]
		self.test_features = self._test_data[:, :self.num_features]
		self.test_labels = self._test_data[:, self.num_features:]

		# Tuples of training and evaluation features and labels
		self.datasets = {
			'train': (self.train_features, self.train_labels),
			'test': (self.test_features, self.test_labels)
		}

		if verbose:
			self.print_shapes()

	def shuffle_training_examples(self, seed = 0):
		np.random.seed(seed)
		np.random.shuffle(self._train_data)

	def get_training_batch(self, batch_num):
		batch_end_index = (batch_num + 1) * self.batch_size

		if (batch_num + 1) * self.batch_size > self.train_features.shape[0]:
			batch_end_index = self.train_features.shape[0] - batch_num * self.batch_size

		batch_input = self.train_features[batch_num * self.batch_size:batch_end_index, :]
		batch_labels = self.train_labels[batch_num * self.batch_size:batch_end_index, :]

		return (batch_input, batch_labels)
	
	def print_shapes(self):
		print('Shape of training features: ' + str(self.train_features.shape))
		print('Shape of training labels: ' + str(self.train_labels.shape) + '\n')
	
		print('Shape of test features: ' + str(self.test_features.shape))
		print('Shape of test labels: ' + str(self.test_labels.shape) + '\n')

		
if __name__ == '__main__':

	dataset = ProteaseData('sparse', 'preprocessed_data/preprocessed_data_sparse.csv', verbose = True, seed = 1)
	dataset.shuffle_training_examples()

	print('Training Features:')
	print(str(dataset.train_features))
	print('Training Labels:')
	print(str(dataset.train_labels) + '\n')

	print('Test Features:')
	print(str(dataset.test_features))
	print('Test Labels:')
	print(str(dataset.test_labels) + '\n')

	dataset.print_shapes()
