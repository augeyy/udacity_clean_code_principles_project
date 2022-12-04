import os
import logging
from joblib import load
from unittest.mock import Mock

import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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


class TestAddChurnColumnToDf:
	"""
	A class to test for the `cl.add_churn_column_to_df` function
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

		df = cl.add_churn_column_to_df(input_df)

		assert "Churn" in df.columns
		assert np.array_equal(df["Churn"].values, np.array([0, 1]))

	def test_column_not_here(self, input_df):
		# Modify input_df
		input_df.drop(columns=["Attrition_Flag"], inplace=True)

		with pytest.raises(ValueError):
			cl.add_churn_column_to_df(input_df)

	def test_column_values_invalid(self, input_df):
		# Modify input_df
		input_df["Attrition_Flag"] = input_df["Attrition_Flag"].apply(lambda x:
			"aaa" if x == "Existing Customer" else "Attrited Customer"
		)

		with pytest.raises(ValueError):
			cl.add_churn_column_to_df(input_df)


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


class TestFeatureEngineering:

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

		assert isinstance(X_train, pd.DataFrame) and X_train.shape == (4, 3)
		assert isinstance(y_train, pd.Series) and y_train.shape == (4,)
		assert isinstance(X_test, pd.DataFrame) and X_test.shape == (2, 3)
		assert isinstance(y_test, pd.Series) and y_test.shape == (2,)

	@pytest.mark.usefixtures('patch_feature_list')
	def test_missing_response_column(self, input_df):
		with pytest.raises(ValueError):
			cl.perform_feature_engineering(
				input_df,
				"This_Col_Does_Not_Exist"
			)

	def test_missing_expected_feature_column(self, input_df, monkeypatch):

		monkeypatch.setattr(
			cl,
			"FEATURE_LIST",
			["Customer_Age", "Gender_Churn", "This_Col_Does_Not_Exist"]
		)

		with pytest.raises(ValueError):
			cl.perform_feature_engineering(
				input_df,
				"Churn"
			)


class TestClassificationReportImage:

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


	def test_success_images_folder_not_exists(self, input_arrs, tmp_path):

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
			str(tmp_path)
		)
		assert len([f for f in (tmp_path / "images").iterdir()]) == 2

	def test_success_images_folder_already_exists(self, input_arrs, tmp_path):

		(tmp_path / "images").mkdir(parents=True)

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
			str(tmp_path)
		)
		assert len([f for f in (tmp_path / "images").iterdir()]) == 2

	def test_input_arr_not_1d(self, input_arrs, tmp_path):

		(y_train, y_test,
		y_train_preds_lr, y_train_preds_rf,
		y_test_preds_lr, y_test_preds_rf) = \
			input_arrs
		
		# Change dimension of one of the input array to cause error
		y_test_preds_lr = np.array([[0, 0, 0], [1, 1, 1]])

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

	def test_input_arr_not_same_shape_as_other_arr(self, input_arrs, tmp_path):

		(y_train, y_test,
		y_train_preds_lr, y_train_preds_rf,
		y_test_preds_lr, y_test_preds_rf) = \
			input_arrs
		
		# Change shape of one of the input array to cause error
		y_test_preds_lr = np.array([0, 0, 0, 1, 1, 1])

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


class TestFeatureImportancePlot:
	"""
	A class to test for the `cl.feature_importance_plot` function
	"""

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


		dst_path = str(tmp_path / "results")  # `results` folder does not exist yet

		cl.feature_importance_plot(model, X_data, dst_path)

		# Make sure that two plots have been made
		assert len([f for f in (tmp_path / "results").iterdir()]) == 2
	
	def test_model_has_no_feat_importances_attr(self, tmp_path, X_y_train, X_data):
		# Train a simple LR model to be used as mock
		# LR model has no `feature_importances_` attributes
		X_train, y_train = X_y_train
		model = LogisticRegression().fit(X_train, y_train)

		dst_path = str(tmp_path / "results")  # `results` folder does not exist yet

		with pytest.raises(ValueError):
			cl.feature_importance_plot(model, X_data, dst_path)


class TestTrainModels:

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

		models_dst_pth = tmp_path / "models"
		images_dst_pth = tmp_path / "images"

		X_train, X_test, y_train, y_test = \
			X_y_train_test

		cl.train_models(
			X_train, X_test,
			y_train, y_test,
			str(models_dst_pth), str(images_dst_pth)
		)

		# Make sure two model artifacts were saved
		assert len([f for f in (tmp_path / "models").iterdir()]) == 2
		try:
			_ = load(tmp_path / "models" / "logistic_clf.pkl")
		except Exception:
			raise ValueError("ERROR: unabled to load lr artifact")
		try:
			_ = load(tmp_path / "models" / "rfc_clf.pkl")
		except Exception:
			raise ValueError("ERROR: unabled to load rf artifact")

		# Make sure ROC curves were saved
		assert len([f for f in (tmp_path / "images").iterdir()]) == 1

		# Make sure that other functions were called
		feature_importance_plot_mock.assert_called()
		classification_report_image_mock.assert_called()
