#!/usr/bin/env python3

import numpy as np;

class ProteaseData:

	num_features = 160
	num_labels = 1


	def __init__(self, train_percent, input_data = "preprocessed_data/preprocessed_data.csv", 
		verbose = True, seed = 0):

		self.train_percent = int(train_percent)

		"""
		Shuffle the training examples, using seed as the initial seed
		for numpy's random number generator so we have reproducible results. Then transpose
		the matrix back to get it in the right shape.
		"""
		all_data = np.genfromtxt(input_data, delimiter = ",", skip_header = 0)
		np.random.seed(seed)
		np.random.shuffle(all_data)


		"""
		Each training/dev/test example is now a column in the matrix.
		"""

		self.train_size = int(self.train_percent / 100 * all_data.shape[0])

		if verbose:
			print(all_data.shape)
			print("Training set size: " + str(self.train_size))

		self.dev_size = (all_data.shape[0] - self.train_size) // 2
		self.test_size = all_data.shape[0] - self.dev_size - self.train_size

		split_indices = [self.train_size, self.train_size + self.dev_size]

		self._train_data, self._dev_data, self._test_data = np.array_split(all_data, split_indices, axis = 0)

		
		"""
		self.train_features and self.train_labels are views of self.train_data, and the views will
		change when the underlying data is shuffled. This applies similary to the dev and test 
		features and labels, although the underlying data shouldn't be shuffled.
		"""


		self.train_features = self._train_data[:, :ProteaseData.num_features]
		self.train_labels = self._train_data[:, ProteaseData.num_features:]
		self.dev_features = self._dev_data[:, :ProteaseData.num_features]
		self.dev_labels = self._dev_data[:, ProteaseData.num_features:]
		self.test_features = self._test_data[:, :ProteaseData.num_features]
		self.test_labels = self._test_data[:, ProteaseData.num_features:]

		# Tuples of training and evaluation features and labels
		self.datasets = {
			'train': (self.train_features, self.train_labels),
			'dev': (self.dev_features, self.dev_labels),
			'test': (self.test_features, self.test_labels)
		}

		if verbose:
			print("Shape of training features: " + str(self.train_features.shape))
			print("Shape of training labels: " + str(self.train_labels.shape) + "\n")

			print("Shape of dev features: " + str(self.dev_features.shape))
			print("Shape of dev labels: " + str(self.dev_labels.shape) + "\n")

			print("Shape of test features: " + str(self.test_features.shape))
			print("Shape of test labels: " + str(self.test_labels.shape) + "\n")

	def shuffle_training_examples(self, seed = 0):
		np.random.seed(seed)
		np.random.shuffle(self._train_data)

		
if __name__ == "__main__":

	dataset = ProteaseData(60, "data/data_test.csv", verbose = True, seed = 1)
	dataset.shuffle_training_examples()

	print("Training Features:")
	print(str(dataset.train_features))
	print("Training Labels:")
	print(str(dataset.train_labels) + '\n')

	print("Dev Features:")
	print(str(dataset.dev_features))
	print("Dev Labels:")
	print(str(dataset.dev_labels) + '\n')	

	print("Test Features:")
	print(str(dataset.test_features))
	print("Test Labels:")
	print(str(dataset.test_labels) + '\n')

	print("Training Dataset Size: " + str(dataset.train_size))
	print("Dev Dataset Size: " + str(dataset.dev_size))
	print("Test Dataset Size: " + str(dataset.test_size))
