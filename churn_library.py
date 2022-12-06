"""
A module to perform data science steps for predicting customer churn based on
the final project of the first course "Clean Code Principles" which is part 
of the Machine Learning DevOps Engineer Nanodegree

Author: Yohann A. <yohann.augey@gmail.com>
Date: Dec. 2022
"""
import os
os.environ["QT_QPA_PLATFORM"]="offscreen"
from joblib import dump

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from config import (
    CATEGORICAL_COLS,
    QUANT_COLS,
    FEATURE_LIST,
    PARAM_GRID
)

import logging
logging.basicConfig(
    filename="./logs/churn_library.log",
    level = logging.INFO,
    filemode="w",
    format="[%(filename)s:%(lineno)s - %(funcName)30s()] - %(levelname)s - %(message)s")

# To make sure legend, axis labels, etc fit in the figure window
plt.rcParams['figure.constrained_layout.use'] = True


def import_data(pth):
    """
    Read CSV file as DataFrame

    Parameters
    ----------
    pth : str
        A path to the CSV file to read

    Returns
    -------
    df : pd.DataFrame
        CSV file read as a DataFrame
    """
    try:
        df = pd.read_csv(pth)
        logging.info(f"SUCCESS: Import data from {pth}")
        return df
    except FileNotFoundError as e:
        logging.error("ERROR: file not found at {pth}")
        raise e


def add_churn_column_to_df(df):
    """
    Create `Churn` column based on values of `Attrition_Flag` field

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column `Attrition_Flag`
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containng the new column `Churn` (modify inplace)
    """
    # Check that `Attrition_Flag` exists
    try:
        assert "Attrition_Flag" in df.columns
    except AssertionError:
        logging.error("ERROR: `Attrition_Flag` column not in df columns")
        raise ValueError()
    
    # Check that `Attrition_Flag` values are well-defined
    try:
        expected_values_set = {"Existing Customer", "Attrited Customer"}
        assert set(df["Attrition_Flag"].unique()).issubset(
            expected_values_set
        )
    except:
        logging.error(
            "ERROR: `Attrition_Flag` values must within {expected_values_set})")
        raise ValueError()
    
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    logging.info("SUCCESS: make `Churn` column")

    return df


def perform_eda(df, dst_pth: str = "."):
    """
    Perform EDA on DataFrame and save figures to the `images` folder

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which to perform EDA
    dst_pth : str, default="."
        Folder where to save figures.
        `images` folder will be created in `folder_pth`

    Returns
    -------
    None
    """
    # Check that DataFrame contains expected columns
    try:
        cols_for_eda = \
            ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]
        assert set(cols_for_eda) <= (set(df.columns))
    except AssertionError:
        logging.error(
            f"ERROR: df does not contain all expected columns {cols_for_eda}"
        )
        raise ValueError()

    images_pth = os.path.join(dst_pth, "images")
    if not os.path.exists(images_pth):
        os.makedirs(images_pth)
        logging.info(f"SUCCESS: using new directory @{images_pth}")
    else:
        logging.info(f"SUCCESS: using existing directory @{images_pth}")
    
    # Plot `Churn` histogram
    plt.figure(figsize=(20,10)) 
    df["Churn"].hist()
    plt.xlabel("Churn")
    plt.ylabel("Count")
    fig_fpath = os.path.join(images_pth, "churn_hist.png")
    plt.savefig(fig_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved `Churn` hist @{fig_fpath}")


    # Plot `Customer_Age` histogram
    plt.figure(figsize=(20,10)) 
    df["Customer_Age"].hist()
    plt.xlabel("Customer Age")
    plt.ylabel("Count")
    fig_fpath = os.path.join(images_pth, "customer_age_hist.png")
    plt.savefig(fig_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved `Custormer_Age` hist @{fig_fpath}")

    # Plot `Marital_Status` bar
    plt.figure(figsize=(20,10)) 
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.xlabel("Marital Status")
    plt.ylabel("Frequency")
    fig_fpath = os.path.join(images_pth, "marital_status_bar.png")
    plt.savefig(fig_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved `Marital_Status` bar plot @{fig_fpath}")

    # Plot `Total_Trans_Ct` distribution
    plt.figure(figsize=(20,10)) 
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    fig_fpath = os.path.join(images_pth, "total_trans_ct_distri.png")
    plt.savefig(fig_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved `Total_Trans_Ct` distribution @{fig_fpath}")

    # Plot correlation
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths = 2)
    fig_fpath = os.path.join(images_pth, "correlation.png")
    plt.savefig(fig_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved correlation plot @{fig_fpath}")

    return


def encoder_helper(df, category_lst, response):
    """
    Encode categorical column into a new column with proportion
    of dependant variable for each category

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which to perform encoding

    category_lst : List[str]
        List of columns that contain categorical features

    response : str
        String of response name

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new columns for
    """
    df = df.copy()
    try:
        assert response in df.columns
    except AssertionError:
        logging.error(f"ERROR: df does not contain `{response}` column")
        raise ValueError()
    try:
        assert set(category_lst) <= set(df.columns)
    except:
        missing_cols = list(set(category_lst) - set(df.columns))
        logging.error(f"ERROR: df does not contain {missing_cols} column(s)")
        raise ValueError()

    for category in category_lst:
        prop_dict = df.groupby(category)[response].mean().to_dict()
        df[category + '_' + response] = df[category].map(prop_dict)
        logging.info(f"SUCCESS: encoded `{category}` column!")
    
    return df


def perform_feature_engineering(df, response):
    """
    Perform feature engineering

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which to perform feature engineering
    response : str
        String of response name

    Returns
    -------
    X_train, X_test : pd.DataFrame
        X train and test data
    y_train, y_test : pd.Series
        y test and test data
    """
    df = df.copy()
    try:
        assert response in df.columns
    except AssertionError:
        logging.error(f"ERROR: df does not contain `{response}` column")
        raise ValueError()

    # Encode categorical features
    # NOTE: mean value is calculated on the entire dataset --> leakage
    # Leave it as it is to be consistent with instructions of Udemy project
    df = encoder_helper(
        df,
        CATEGORICAL_COLS,
        response
    )
    logging.info("SUCCESS: encoded all categorical features!")

    try:
        assert set(FEATURE_LIST) <= set(df.columns)
    except:
        missing_cols = list(set(FEATURE_LIST) - set(df.columns))
        logging.error(f"ERROR: df does not contain {missing_cols} column(s)")
        raise ValueError()

    X = df[FEATURE_LIST]
    y = df[response]

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y,
            test_size=0.3,
            random_state=42
        )
    logging.info("SUCCESS: split train and test sets!")

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
    dst_pth: str = "."
):
    """
    Produces classification report for training and testing results
    and stores report as image in `images` folder

    Parameters
    ----------
    y_train : ndarray
        Training response values
    y_test : ndarray
        Test response values
    y_train_preds_lr : ndarray
        Training predictions from logistic regression
    y_train_preds_rf : ndarray
        Training predictions from random forest
    y_test_preds_lr : ndarray
        Test predictions from logistic regression
    y_test_preds_rf : ndarray
        Test predictions from random forest
    folder_pth : str, default="."
        Folder where to save figures.
        `images` folder will be created in `folder_pth`

    Returns
    -------
    None
    """
    # Check shape of arrays
    try:
        assert (
            y_train.ndim == y_train_preds_lr.ndim == y_train_preds_rf.ndim \
            == y_test.ndim == y_test_preds_lr.ndim == y_test_preds_rf.ndim \
            == 1
        )
    except AssertionError:
        logging.error(f"ERROR: arrs must be 1D")
        raise ValueError()

    try:
        assert (
            y_train.shape == y_train_preds_lr.shape == y_train_preds_rf.shape
        )
        assert (
            y_test.shape == y_test_preds_lr.shape == y_test_preds_rf.shape
        )
    except AssertionError:
        logging.error(f"ERROR: train arrs and test arrs must have same shape")
        raise ValueError()

    images_pth = os.path.join(dst_pth, "images")
    if not os.path.exists(images_pth):
        os.makedirs(images_pth)
        logging.info(f"SUCCESS: using new directory @{images_pth}")
    else:
        logging.info(f"SUCCESS: using existing directory @{images_pth}")

    preds_dict = {
        "lr": {
            "train": y_train_preds_lr,
            "test": y_test_preds_lr
        },
        "rf": {
            "train": y_train_preds_rf,
            "test": y_test_preds_rf
        }
    }
    for model_name in preds_dict.keys():
        y_train_preds = preds_dict[model_name]["train"]
        y_test_preds = preds_dict[model_name]["test"]

        plt.rc('figure', figsize=(5, 5))
        plt.text(
            0.01, 1.25, str('Random Forest Train'),
            {'fontsize': 10}, fontproperties = 'monospace'
        )
        plt.text(
            0.01, 0.05, str(classification_report(y_test, y_test_preds)),
            {'fontsize': 10}, fontproperties = 'monospace'
        )
        plt.text(
            0.01, 0.6, str('Random Forest Test'),
            {'fontsize': 10}, fontproperties = 'monospace'
        )
        plt.text(
            0.01, 0.7, str(classification_report(y_train, y_train_preds)),
            {'fontsize': 10}, fontproperties = 'monospace'
        )
        plt.axis('off')
        fpath = os.path.join(images_pth, f"{model_name}_results_train.png")
        plt.savefig(fpath)
        plt.close()

        logging.info(
            f"SUCCESS: wrote {model_name} classfication results @{fpath}"
        )
    return


def feature_importance_plot(model, X_data, dst_pth: str = "."):
    """
    Creates and stores the feature importances in pth

    Parameters
    ----------
    model
        Model object containing `feature_importances_` attributes

    X_data : pd.DataFrame
        DataFrame of X values

    dst_pth : str, default="."
        Path to store the figure

    Returns
    -------
        None
    """
    # Check that model has the attribute `feature_importances_`
    try:
        assert hasattr(model, "feature_importances_")
    except AssertionError:
        logging.error(
            "ERROR: `model` does not have the attribute `feature importances`"
        )
        raise ValueError()

    if not os.path.exists(dst_pth):
        os.makedirs(dst_pth)
        logging.info(f"SUCCESS: creating directory @{dst_pth}")

    ##################
    # Plot SHAP values
    ##################
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    shap_fpath = os.path.join(dst_pth, "shap_values.png")
    plt.savefig(shap_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved shap values plot @{shap_fpath}")

    #########################
    # Plot feature importance
    #########################
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    feat_imp_fpath = os.path.join(dst_pth, "feature_importances.png")
    plt.savefig(feat_imp_fpath)
    plt.close()
    logging.info(f"SUCCESS: saved feature importances plot @{feat_imp_fpath}")

    return



def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        models_dst_pth: str = "./models",
        images_dst_pth: str = "./images",
    ):
    """
    Train, store model results: images + scores, and store models

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        X train and test data
    y_train, y_test : pd.Series
        y train and test data
    models_dst_pth: str, default="./models"
        Folder where to model artifacts
    images_dst_pth: str, default="./images"
        Folder where to save ROC curves plot

    Returns
    -------
    None
    """
    if not os.path.exists(models_dst_pth):
        os.makedirs(models_dst_pth, exist_ok=True)
        logging.info(f"SUCCESS: created new directory @{models_dst_pth}")

    if not os.path.exists(images_dst_pth):
        os.makedirs(images_dst_pth, exist_ok=True)
        logging.info(f"SUCCESS: created new directory @{images_dst_pth}")

    # Grid Search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    #######
    # Train
    #######
    logging.info("SUCCESS: training lr and rf models...")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_
    logging.info("SUCCESS: finished training RF models via Grid Search!")

    lrc.fit(X_train, y_train)
    logging.info("SUCCESS: finished training Logistic Regression model!")

    # Save models
    model_dict = {"logistic": lrc, "rfc": rfc}
    for name, model in model_dict.items():
        fpath = os.path.join(models_dst_pth, f"{name}_clf.pkl")
        dump(model, fpath)
        logging.info(f"SUCCESS: saved {name} model artifact @{fpath}")

    #######
    # Eval
    #######
    logging.info("SUCCESS: making predictions...")
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    logging.info("SUCCESS: finished making predictions!")

    # Save ROC curves
    logging.info("SUCCESS: making ROC curves...")
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    fpath = os.path.join(images_dst_pth, "roc_curves.png")
    plt.savefig(fpath)
    plt.close()
    logging.info(f"SUCCESS: made ROC curves @{fpath}")

    # Model results
    logging.info("SUCCESS: making classification results...")
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    )

    # Shap + Feature importances
    logging.info("SUCCESS: making feature importances plots...")
    feature_importance_plot(cv_rfc.best_estimator_, X_test, images_dst_pth)

    return


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")

    df = add_churn_column_to_df(df)

    perform_eda(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    train_models(
        X_train, X_test, y_train, y_test,
        models_dst_pth="./models", images_dst_pth="./images"
    )
