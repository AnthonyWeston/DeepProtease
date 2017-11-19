#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod
import argparse
import pandas as pd 


class AbstractPreprocessor:

	"""
	Base class for classes used to preprocess HIV-1 protease cleavage data
	"""

	__metaclass__ = ABCMeta

	# A mapping of one letter amino acid codes to integers by alphabetical order

	data_mapping = {
	'A': 0,
	'C': 1,
	'D': 2,
	'E': 3,
	'F': 4,
	'G': 5,
	'H': 6,
	'I': 7,
	'K': 8,
	'L': 9,
	'M': 10,
	'N': 11,
	'P': 12,
	'Q': 13,
	'R': 14,
	'S': 15,
	'T': 16,
	'V': 17,
	'W': 18,
	'Y': 19,
	}

	@abstractmethod
	def __init__(self):
		# Parse arguments first

		parser = argparse.ArgumentParser()
		parser.add_argument('-i', metavar = 'Input Filename', required = True, 
			help = 'Specifies the input data file')
		parser.add_argument('-o', metavar = 'Output Filename', required = True,
			help = 'Specifies the output data file name')
		parser.add_argument('-mode', metavar = 'Output Mode', required = True,
			choices = ['replace', 'append'], help = 'Specifies whether to overwrite or \
			replace the destination file')
		args = parser.parse_args()

		self.input_filename = args.i
		self.output_filename = args.o
		self.output_mode = args.mode[0]



	@abstractmethod
	def preprocess_data(self):

		"""
		Take an input file containing HIV protease cleavage data and begin preprocessing.
		Input format example from file: DQKPLAQR,-1
		Output example row of dataframe (First row is column labels, second is data): 
		   0   1   2   3   4   5   6   7  label
		0  2  13   8  12   9   0  13  14     0

		This is the format that gets handed off to the base class when they call
		super.preprocess_data()
		Note: in the label column, -1 is converted to 0.
		"""


		"""Parse the input file. The format is a series of 8 amino acid characters, followed by
		a comma, followed by 1 or -1"""

		input_data = pd.read_table(self.input_filename, delimiter = ',', names = ['sequence', 'label'], 
			dtype = {'sequence': str, 'label': int} )

		sequence_data = input_data['sequence']
		self.label_data = input_data['label']
		self.label_data = self.label_data.replace({-1, 0})


		sequence_data = sequence_data.apply(lambda x:'|'.join(list(x)))
		sequence_data = sequence_data.str.split('|', 8, expand = True)
		self.sequence_data = sequence_data

		self.sequence_data = self.sequence_data.applymap(
			lambda x: AbstractPreprocessor.data_mapping[x])