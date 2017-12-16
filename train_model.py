#!/usr/bin/env python3

from sparse_encoding_model import *
from data import *
import tensorflow as tf
import argparse

eval_interval = 250

def train_model(model, num_epochs):

    print('\nUntrained Model:')
    model.print_metrics()


    for epoch in range(num_epochs):

        num_batches = model.data.train_features.shape[0] // model.data.batch_size
        for i in range(num_batches):

            batch_features, batch_labels = model.data.get_training_batch(i)
            model.sess.run(model.optimizer,
                feed_dict = {model.x: batch_features, model.y: batch_labels})

        model.run_summary_ops(epoch + 1)

        if (epoch + 1) % eval_interval == 0:
            print('\nEpoch ' + str(epoch + 1) + ':\n')
            model.print_metrics()
            
        model.data.shuffle_training_examples()
        
    print('\n--Final trained model--\n')
    print('Epoch ' + str(num_epochs) + ':\n')
    model.print_metrics()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', metavar='Number of model to test', type = int)
    parser.add_argument('-e', metavar='Number of epochs to train', type = int)
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    num_models = args.n
    num_epochs = args.e

    data = ProteaseData(90, "preprocessed_data/preprocessed_data.csv")

    model = SparseEncodingModel()

    train_model(model, 25000)





if __name__ == '__main__':
    main()