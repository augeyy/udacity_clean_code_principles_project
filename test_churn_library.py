import os
import logging

import pytest

import numpy as np
import pandas as pd

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


class TestMakeChurnColumn:
	"""
	A class to test for the `cl.make_target_column` function
	"""

	@pytest.fixture
	def input_df(self):
		return pd.DataFrame(data={
			"Attrition_Flag": ["Existing Customer", "Attrited Customer"]
		})

	def test_success(self, input_df):
		expected_df = pd.DataFrame(data={
			"Attrition_Flag": ["Existing Customer", "Attrited Customer"],
			"Churn": [0, 1]
		})

		df = cl.make_churn_column(input_df)

		assert "Churn" in df.columns
		assert np.array_equal(df["Churn"].values, np.array([0, 1]))

	def test_column_not_here(self, input_df):
		# Modify input_df
		input_df.drop(columns=["Attrition_Flag"], inplace=True)

		with pytest.raises(ValueError):
			cl.make_churn_column(input_df)

	def test_column_values_invalid(self, input_df):
		# Modify input_df
		input_df["Attrition_Flag"] = input_df["Attrition_Flag"].apply(lambda x:
			"aaa" if x == "Existing Customer" else "Attrited Customer"
		)

		with pytest.raises(ValueError):
			cl.make_churn_column(input_df)

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








