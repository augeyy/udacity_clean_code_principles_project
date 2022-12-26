"""
File containing tests for the `churn_library` module

Author: Yohann A. <yohann.augey@gmail.com>
Date: Dec. 2022
"""

import os
import logging
from joblib import load
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import churn_library as cl


class LoggingHandler:
	@property
	def logger(self):
		return logging.getLogger(f"{self.__class__.__name__}")


class TestImportData(LoggingHandler):
	"""A class to test for the `cl.import_data` function"""

	@pytest.fixture
	def path(self):
		return "./data/bank_data.csv"

	def test_import_success(self, path):
		df = cl.import_data(path)
		try:
			assert df.shape[0] > 0
			assert df.shape[1] > 0
			self.logger.info("SUCCESS: read non-empty file")
		except AssertionError:
			self.logger.error(
				"ERROR: file has either empty rows or empty columns")


	def test_import_file_not_found(self):
		path = "./this/file/does/not/exist.csv"

		try:
			with pytest.raises(ValueError):
				cl.import_data(path)
		except:
			self.logger.error("ERROR: file was found")
			raise
		else:
			self.logger.info("SUCCESS: file not found")
				


class TestAddChurnColumnToDf(LoggingHandler):
	"""A class to test for the `cl.add_churn_column_to_df` function"""

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

		df = cl.add_churn_column_to_df(input_df)

		try:
			assert "Churn" in df.columns
			self.logger.info("SUCCESS: `Churn` in columns")
		except AssertionError:
			self.logger.error("ERROR: `Churn` not in columns")

		try:
			assert np.array_equal(df["Churn"].values, np.array([0, 1]))
			self.logger.info("SUCCESS: `Churn` values are the expected ones")
		except AssertionError:
			self.logger.error("ERROR: `Churn` values are not the expected ones")

	def test_column_not_here(self, input_df):
		# Modify input_df
		input_df.drop(columns=["Attrition_Flag"], inplace=True)

		try:
			with pytest.raises(ValueError):
				cl.add_churn_column_to_df(input_df)
		except:
			self.logger.error("ERROR: `Attrition_Flag` column was found")
			raise
		else:
			self.logger.info("SUCCESS: `Attrition_Flag` column not found")

	def test_column_values_invalid(self, input_df):
		# Modify input_df
		input_df["Attrition_Flag"] = input_df["Attrition_Flag"].apply(lambda x:
			"aaa" if x == "Existing Customer" else x
		)

		try:
			with pytest.raises(ValueError):
				cl.add_churn_column_to_df(input_df)
		except:
			self.logger.error("ERROR: no invalid value in `Attrition_Flag`")
		else:
			self.logger.info("SUCCESS: invalid value in `Attrition_Flag`")


class TestPerformEda(LoggingHandler):
	"""A class to test for the `cl.perform_eda` function"""

	@pytest.fixture
	def input_df(self):
		return pd.DataFrame(data={
			"Churn": [1, 0, 1],
			"Customer_Age": [30, 31, 50],
			"Marital_Status": ["Single", "Married", "Married"],
			"Total_Trans_Ct": [50, 50, 70]
		})

	def test_success(self, input_df, tmp_path):
		pth = tmp_path

		cl.perform_eda(input_df, os.path.join(tmp_path, "images"))
		try:
			assert len([f for f in (pth / "images" / "eda").iterdir()]) == 5
			self.logger.info("SUCCESS: correct number of output files")
		except AssertionError:
			self.logger.error("ERROR: incorrect number of output files")

	def test_missing_col_in_df(self, input_df, tmp_path):
		pth = tmp_path

		# Drop one of expected column
		input_df.drop(columns=["Customer_Age"], inplace=True)

		try:
			with pytest.raises(ValueError):
				cl.perform_eda(input_df, str(pth))
		except:
			self.logger.error("ERROR: no missing column in df")
		else:
			self.logger.info("SUCCESS: missing col in df")


class TestEncoderHelper(LoggingHandler):
	"""A class to test for the `cl.encoder_helper` function"""

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

		try:
			assert len(new_df.columns) == 6
			self.logger.info("SUCCESS: correct number of columns in output df")
		except AssertionError:
			self.logger.error("ERROR: incorrect numbe of column in output df")

		expected = [
			["Gender_Churn",  [0.75, 0.75, 0.75, 0.75, 0, 0]],
			["Marital_Status_Churn", [0.5, 1, 1, 0, 0, 0.5]]
		]

		for col_name, col_values in expected:

			try:
				pd.testing.assert_series_equal(
					new_df[col_name],
					pd.Series(col_values, name=col_name)
				)
				self.logger.info(
					"SUCCESS: correct values for `%s` col", col_name)
			except:
				self.logger.error(
					"ERROR: incorrect values for `%s` col", col_name)

	def test_missing_response_column(self, input_df):

		try:
			with pytest.raises(ValueError):
				new_df = cl.encoder_helper(
					input_df,
					["Gender", "Marital_Status"],
					"This_Col_Does_Not_Exist"
				)
		except:
			self.logger.error("ERROR: response col found")
		else:
			self.logger.info("SUCCESS: response col not found")

	def test_missing_to_be_encoded_column(self, input_df):

		try:
			with pytest.raises(ValueError):
				new_df = cl.encoder_helper(
					input_df,
					["Gender", "This_Col_Does_Not_Exist"],
					"Churn"
				)
		except:
			self.logger.error("ERROR: found expected missing col to be encoded")
		else:
			self.logger.info(
				"SUCCESS: expected missing col to be encoded not found")


class TestFeatureEngineering(LoggingHandler):
	"""A class to test for the `cl.perform_feature_engineering` function"""

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
				["Single", "Divorced", "Divorced","Married", "Married", "Single"]
		})
	
	# Mock `cl.encoder_helper`
	@pytest.fixture(autouse=True)
	def patch_encoder_helper(self, monkeypatch, input_df):
		def mock_encoder_helper(*args, **kwargs):
			encode_df = pd.DataFrame(
				data={
					"Gender_Churn": [0.75, 0.75, 0.75, 0.75, 0, 0],
					"Marital_Status_Churn": [0.5, 1, 1, 0, 0, 0.5]
				}
			)
			return pd.concat(
				[input_df, encode_df],
				axis=1
			)

		monkeypatch.setattr(cl, "encoder_helper", mock_encoder_helper)

	# Mock `cl.FEATURE_LIST` global variable
	@pytest.fixture(autouse=True)
	def patch_feature_list(self, monkeypatch):
		monkeypatch.setattr(
			cl,
			"FEATURE_LIST",
			["Customer_Age", "Gender_Churn", "Marital_Status_Churn"]
		)

	@pytest.mark.usefixtures('patch_feature_list')
	def test_success(self, input_df):
		X_train, X_test, y_train, y_test = \
			cl.perform_feature_engineering(
				input_df,
				"Churn"
			)

		output = {
			"X_train": X_train,
			"X_test": X_test,
			"y_train": y_train,
			"y_test": y_test
		}

		expected = [
			["X_train", pd.DataFrame, (4, 3)],
			["X_test", pd.DataFrame, (2, 3)],
			["y_train", pd.Series, (4,)],
			["y_test", pd.Series, (2,)]
		]

		for arr_name, arr_type, shape in expected:
			try:
				assert isinstance(output[arr_name], arr_type) \
					   and output[arr_name].shape == shape
				self.logger.info(
					"SUCCESS: %s is %s and has correct shape",
					arr_name,
					arr_type
				)
			except:
				self.logger.error(
					"ERROR: %s not %s or not shape %s (expected: %s)",
					arr_name,
					arr_type,
					output[arr_name].shape,
					shape
				)


	@pytest.mark.usefixtures('patch_feature_list')
	def test_missing_response_column(self, input_df):
		try:
			with pytest.raises(ValueError):
				cl.perform_feature_engineering(
					input_df,
					"This_Col_Does_Not_Exist"
				)
		except:
			self.logger.error("ERROR: response col found")
		else:
			self.logger.info("SUCCESS: response col not found")

	def test_missing_expected_feature_column(self, input_df, monkeypatch):

		monkeypatch.setattr(
			cl,
			"FEATURE_LIST",
			["Customer_Age", "Gender_Churn", "This_Col_Does_Not_Exist"]
		)

		try:
			with pytest.raises(ValueError):
				cl.perform_feature_engineering(
					input_df,
					"Churn"
				)
		except:
			self.logger.error("ERROR: found expected missing col")
		else:
			self.logger.info("SUCCESS: expected missing col not found")


class TestClassificationReportImage(LoggingHandler):
	"""A class to test for the `cl.classification_report_image` function"""

	@pytest.fixture
	def input_arrs(self):
		y_train = np.array([0, 0, 0, 1, 1])
		y_test = np.array([0, 1, 0])
		y_train_preds_lr = np.array([0, 0, 1, 0, 1])
		y_train_preds_rf = np.array([1, 0, 0, 1, 1])
		y_test_preds_lr = np.array([0, 0, 1])
		y_test_preds_rf = np.array([0, 1, 0])
		return (
			y_train, y_test,
			y_train_preds_lr, y_train_preds_rf,
			y_test_preds_lr, y_test_preds_rf
		)

	def test_success(self, input_arrs, tmp_path):

		(y_train, y_test,
		y_train_preds_lr, y_train_preds_rf,
		y_test_preds_lr, y_test_preds_rf) = \
			input_arrs

		cl.classification_report_image(
			y_train,
			y_test,
			y_train_preds_lr,
			y_train_preds_rf,
			y_test_preds_lr,
			y_test_preds_rf,
			os.path.join(tmp_path, "images")
		)

		try:
			n_outputs = \
				len([f for f in (tmp_path / "images" / "results").iterdir()])
			assert n_outputs == 2
			self.logger.info("SUCCESS: correct number of output files")
		except AssertionError:
			self.logger.error("ERROR: incorrect number of output files")

	def test_input_arr_not_1d(self, input_arrs, tmp_path):

		(y_train, y_test,
		y_train_preds_lr, y_train_preds_rf,
		y_test_preds_lr, y_test_preds_rf) = \
			input_arrs
		
		# Change dimension of one of the input array to cause error
		y_test_preds_lr = np.array([[0, 0, 0], [1, 1, 1]])

		try:
			with pytest.raises(ValueError):
				cl.classification_report_image(
					y_train,
					y_test,
					y_train_preds_lr,
					y_train_preds_rf,
					y_test_preds_lr,
					y_test_preds_rf,
					str(tmp_path)
				)
		except:
			self.logger.error("ERROR: input array is 1D")
		else:
			self.logger.info("SUCCESS: input array is not 1D")

	def test_input_arr_not_same_shape_as_other_arr(self, input_arrs, tmp_path):

		(y_train, y_test,
		y_train_preds_lr, y_train_preds_rf,
		y_test_preds_lr, y_test_preds_rf) = \
			input_arrs
		
		# Change shape of one of the input array to cause error
		y_test_preds_lr = np.array([0, 0, 0, 1, 1, 1])

		try:
			with pytest.raises(ValueError):
				cl.classification_report_image(
					y_train,
					y_test,
					y_train_preds_lr,
					y_train_preds_rf,
					y_test_preds_lr,
					y_test_preds_rf,
					str(tmp_path)
				)
		except:
			self.logger.error(
				"ERROR: input arrays w/ different length not found")
		else:
			self.logger.info(
				"SUCCESS: input arrays w/ different length found")


class TestFeatureImportancePlot(LoggingHandler):
	"""A class to test for the `cl.feature_importance_plot` function"""

	@pytest.fixture
	def X_y_train(self):
		X_train = pd.DataFrame(data={
			"Feature1": [1, 2, 3, 4, 5],
			"Feature2": [0, 0, 1, 1, 1]
		})
		y_train = pd.Series([0, 0, 0, 1, 1])
		return X_train, y_train
	
	@pytest.fixture
	def X_data(self):
		X_data = pd.DataFrame(data={
			"Feature1": [1, 3],
			"Feature2": [0, 1]
		})
		return X_data

	def test_success(self, tmp_path, X_y_train, X_data):
		# Train a simple RF model to be used as mock
		X_train, y_train = X_y_train
		model = RandomForestClassifier().fit(X_train, y_train)

		cl.feature_importance_plot(
			model,
			X_data,
			os.path.join(tmp_path, "images")
		)

		# Make sure that two plots have been made
		try:
			n_outputs = \
				len([f for f in (tmp_path / "images" / "results").iterdir()])
			assert n_outputs == 2
			self.logger.info("SUCCESS: correct number of output files")
		except AssertionError:
			self.logger.error("ERROR: incorrect number of output files")
	
	def test_model_has_no_feat_importances_attr(
		self, tmp_path, X_y_train, X_data
	):
		# Train a simple LR model to be used as mock
		# LR model has no `feature_importances_` attributes
		X_train, y_train = X_y_train
		model = LogisticRegression().fit(X_train, y_train)

		dst_path = str(tmp_path / "results")  # `results` folder does not exist

		try:
			with pytest.raises(ValueError):
				cl.feature_importance_plot(model, X_data, dst_path)
		except:
			self.logger.error("ERROR: model has `feature_importances_` attr")
		else:
			self.logger.info(
				"SUCCESS: model has no `feature_importances_` attr")


class TestTrainModels(LoggingHandler):
	"""A class to test for the `cl.train_models` function"""

	@pytest.fixture
	def X_y_train_test(self):
		X_train = pd.DataFrame(data={
			"Feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
			"Feature2": [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
		})
		y_train = pd.Series([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])

		X_test = pd.DataFrame(data={
			"Feature1": [1, 2, 3],
			"Feature2": [0, 0, 1]
		})
		y_test = pd.Series([0, 0, 0])
		return X_train, X_test, y_train, y_test

	def test_success(self, tmp_path, monkeypatch, X_y_train_test):
		# Mock functions called inside `train_models`
		classification_report_image_mock = Mock()
		feature_importance_plot_mock = Mock()

		monkeypatch.setattr(
			cl,
			"classification_report_image",
			classification_report_image_mock
		)

		monkeypatch.setattr(
			cl,
			"feature_importance_plot",
			feature_importance_plot_mock
		)

		X_train, X_test, y_train, y_test = \
			X_y_train_test

		cl.train_models(
			X_train, X_test,
			y_train, y_test,
			os.path.join(tmp_path, "models"),
			os.path.join(tmp_path, "images")
		)

		# Make sure two model artifacts were saved
		expected = ["logistic_clf.pkl", "rfc_clf.pkl"]
		for artifact_name in expected:
			try:
				_ = load(tmp_path / "models" / "logistic_clf.pkl")
				self.logger.info("SUCCESS: could load %s", artifact_name)
			except:
				self.logger.error("ERROR: could not load %s", artifact_name)
				raise

		# Make sure ROC curves were saved
		try:
			n_outputs = \
				len([f for f in (tmp_path / "images" / "results").iterdir()])
			assert n_outputs == 1
			self.logger.info("SUCCESS: correct number of output file")
		except AssertionError:
			self.logger.error("ERROR: incorrect number of output file")

		# Make sure that other functions were called
		try:
			feature_importance_plot_mock.assert_called()
			classification_report_image_mock.assert_called()
			self.logger.info(
				"SUCCESS: expected functions to be called were called")
		except:
			self.logger.error(
				"ERROR: expected functions to be called were not called")
