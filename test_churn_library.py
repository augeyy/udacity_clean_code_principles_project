import os
import logging

import pytest

import churn_library as cl



@pytest.fixture()
def path():
	return "./data/bank_data.csv"


class TestImportData:
	"""
	A class to test for the `cl.import_data` function
	"""

	def test_import_success(self, path):
		'''
		test data import - this example is completed for you to assist with the other test functions
		'''
		df = cl.import_data(path)

		assert df.shape[0] > 0
		assert df.shape[1] > 0

	
	def test_import_file_not_found(self):
		path = "./this/path/does/not/exists"

		with pytest.raises(FileNotFoundError):
			cl.import_data(path)

# def test_eda(perform_eda):
# 	'''
# 	test perform eda function
# 	'''


# def test_encoder_helper(encoder_helper):
# 	'''
# 	test encoder helper
# 	'''


# def test_perform_feature_engineering(perform_feature_engineering):
# 	'''
# 	test perform_feature_engineering
# 	'''


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''


# if __name__ == "__main__":
# 	pass








