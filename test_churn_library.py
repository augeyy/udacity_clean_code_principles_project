import os
import logging

import pytest

import numpy as np
import pandas as pd

import churn_library as cl


class TestImportData:
	"""
	A class to test for the `cl.import_data` function
	"""

	@pytest.fixture
	def path(self):
		return "./data/bank_data.csv"

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


class TestPerformEda:
	"""
	A class to test for the `cl.perform_eda` function
	"""

	@pytest.fixture
	def input_df(self):
		return pd.DataFrame(data={
			"Churn": [1, 0, 1],
			"Customer_Age": [30, 31, 50],
			"Marital_Status": ["Single", "Married", "Married"],
			"Total_Trans_Ct": [50, 50, 70]
		})

	def test_success_images_folder_already_exists(self, input_df, tmp_path):
		pth = tmp_path

		cl.perform_eda(input_df, str(pth))
		assert len([f for f in (pth / "images").iterdir()]) == 5

	def test_success_images_folder_not_exists(self, input_df, tmp_path):
		pth = tmp_path

		os.makedirs(os.path.join(pth, "images"))

		cl.perform_eda(input_df, str(pth))
		assert len([f for f in (pth / "images").iterdir()]) == 5

	def test_missing_col_in_df(self, input_df, tmp_path):
		pth = tmp_path

		# Drop one of expected column
		input_df.drop(columns=["Customer_Age"], inplace=True)

		with pytest.raises(ValueError):
			cl.perform_eda(input_df, str(pth))


class TestEncoderHelper:
	"""
	A class to test for the `cl.encoder_helper` function
	"""

	@pytest.fixture
	def input_df(self):
		return pd.DataFrame(data={
			"Churn": \
				[1, 1, 1, 0, 0, 0],
			"Customer_Age": \
				[30, 31, 40, 29, 30, 50],
			"Gender": \
				["M", "M", "M", "M", "F", "F"],
			"Marital_Status": \
				["Single", "Divorced", "Divorced", "Married", "Married", "Single"]
		})

	def test_success(self, input_df):

		new_df = cl.encoder_helper(
			input_df,
			["Gender", "Marital_Status"],
			"Churn"
		)

		assert len(new_df.columns) == 6
		pd.testing.assert_series_equal(
			new_df["Gender_Churn"],
			pd.Series([0.75, 0.75, 0.75, 0.75, 0, 0], name="Gender_Churn")
		)
		pd.testing.assert_series_equal(
			new_df["Marital_Status_Churn"],
			pd.Series([0.5, 1, 1, 0, 0, 0.5], name="Marital_Status_Churn")
		)

	def test_missing_response_column(self, input_df):

		with pytest.raises(ValueError):
			new_df = cl.encoder_helper(
				input_df,
				["Gender", "Marital_Status"],
				"This_Col_Does_Not_Exist"
			)

	def test_missing_to_be_encoded_column(self, input_df):

		with pytest.raises(ValueError):
			new_df = cl.encoder_helper(
				input_df,
				["Gender", "This_Col_Does_Not_Exist"],
				"Churn"
			)


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








