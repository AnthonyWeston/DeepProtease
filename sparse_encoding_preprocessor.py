#!/usr/bin/env python3

import pandas as pd
import numpy as np
from abstract_preprocessor import *

class SparseEncodingPreprocessor(AbstractPreprocessor):

	"""
	Extends the base class for classes used to preprocess HIV-1 protease cleavage data.
	"""

	def __init__(self):
		super().__init__()

	def preprocess_data(self):
		super().preprocess_data()

		"""
		Complete preprocessing of HIV protease cleavage data. Each column except label
		is expanded into eight one-hot arrays of length 20 with a 1 in the position
		corresponding to the identity of the amino acid. The final dataframe has 161
		columns. The data is output to a file specified in the -o argument of the 
		program.
		"""

		self.sequence_data = self.sequence_data.applymap(
				lambda x: SparseEncodingPreprocessor.one_hot_string(20, x))

		final_sequence_data = pd.DataFrame()
		for i in range(8):
			dataframe_part = self.sequence_data[i].str.split('|', 20, expand = True)
			dataframe_part.columns = [i * 20 + j for j in range(20)]
			
			final_sequence_data = pd.concat((final_sequence_data, dataframe_part), axis = 1)

		self.labeled_sequence_data = final_sequence_data
		self.labeled_sequence_data['labels'] = self.label_data

		print(self.labeled_sequence_data)

		open_mode = 'w' if self.output_mode == 'replace' else 'a'
		output_file = open(self.output_filename, open_mode)
		self.labeled_sequence_data.to_csv(output_file, index = False)
	

	def one_hot_string(length, one_index):
		return '|'.join(['0' if i != one_index else '1' for i in range(length)])


if __name__ == "__main__":

	"""
	Scripts used for troubleshooting during development
	"""

	print(SparseEncodingPreprocessor.one_hot_string(20, 0))