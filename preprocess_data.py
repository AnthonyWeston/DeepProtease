#! /usr/bin/python3

import argparse
import pandas as pd 



def preprocess_data():

	"""Take an input file containing HIV protease cleavage data and preprocess it into
	a format we can use as input for a neural network"""

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', nargs = 1, metavar = 'InputFileName', required = True, 
		help = 'Specifies the input data file')
	parser.add_argument('-o', nargs = 1, metavar = 'OutputFileName', required = True,
		help = 'Specifies the output data file name')
	parser.add_argument('-mode', nargs = 1, metavar = 'OutputMode', required = True,
		choices = ['replace', 'append'], help = 'Specifies whether to overwrite or \
		replace the destination file')
	args = parser.parse_args()






if __name__ == '__main__':

	preprocess_data()