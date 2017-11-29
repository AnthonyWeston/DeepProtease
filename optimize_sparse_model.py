#!/usr/bin/env python3

from sparse_encoding_model import *
from data import *
import tensorflow as tf
import argparse

eval_interval = 250

def train_models(models, num_epochs):

	print('\nUntrained Models:')
	print_models_metrics(models)

	arbitrary_model = list(models.values())[0]
	data = arbitrary_model.data

	for epoch in range(num_epochs):

		num_batches = data.train_features.shape[0] // data.batch_size
		for i in range(num_batches):

			batch_features, batch_labels = data.get_training_batch(i)
			
			for _, model in models.items():

				model.sess.run(model.optimizer,
					feed_dict = {model.x: batch_features, model.y: batch_labels})

		for _, model in models.items():
			model.run_summary_ops()

		if (epoch + 1) % eval_interval == 0:
			print('\nEpoch ' + str(epoch + 1) + ':\n')
			print_models_metrics(models)
			
		model.data.shuffle_training_examples()
		
	print('\n--Final trained models--\n')
	print('Epoch ' + str(num_epochs) + ':\n')
	print_models_metrics(models)

def print_models_metrics(models):

	for _, model in models.items():
		model.print_model_metrics()

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', metavar='Number of models to test', type = int)
	parser.add_argument('-e', metavar='Number of epochs to train', type = int)
	parser.add_argument('--debug', action = 'store_true')
	args = parser.parse_args()

	num_models = args.n
	num_epochs = args.e

	data = ProteaseData('sparse', 'preprocessed_data/preprocessed_data_sparse.csv')



	models = {'Model' + str(i): SparseEncodingModel(data, name = 'Model' + str(i)) for i in range(num_models)}

	train_models(models, num_epochs)





if __name__ == '__main__':
	main()