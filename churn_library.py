"""
A module to perform data science steps for predicting customer churn based on
the final project of the first course "Clean Code Principles" which is part
of the Machine Learning DevOps Engineer Nanodegree

Author: Yohann A. <yohann.augey@gmail.com>
Date: Dec. 2022
"""
import os
from typing import Any, List

from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split

from constants import (
    RANDOM_STATE,
    TEST_SIZE,
    CATEGORICAL_COLS,
    FEATURE_LIST,
    PARAM_GRID
)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# To make sure legend, axis labels, etc fit in the figure window
plt.rcParams['figure.constrained_layout.use'] = True


def import_data(
    path: str
):
    """
    Read CSV file as DataFrame

    Parameters
    ----------
    path : str
        A path to the CSV file to read

    Returns
    -------
    df : pd.DataFrame
        CSV file read as a DataFrame
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise ValueError(f"ERROR: file not found at {path}") from exc
    else:
        return df


def add_churn_column_to_df(
    df: pd.DataFrame
):
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
    if "Attrition_Flag" not in df.columns:
        raise ValueError("ERROR: `Attrition_Flag` column not in df columns")

    # Check that `Attrition_Flag` values are well-defined
    expected_values_set = {"Existing Customer", "Attrited Customer"}
    if not set(df["Attrition_Flag"].unique()).issubset(expected_values_set):
        raise ValueError(
            f"ERROR: `Attrition_Flag` values must within {expected_values_set}"
        )

    df["Churn"] = df["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1
    })

    return df


def perform_eda(
    df: pd.DataFrame,
    dst_path: str = "./images"
):
    """
    Perform EDA on DataFrame and save figures to the `images` folder

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which to perform EDA
    dst_path : str, default="./images"
        Folder where to save figures.
        `eda` folder will be created in `dst_path` if not existing yet

    Returns
    -------
    None
    """
    # Check that DataFrame contains expected columns
    cols_for_eda = \
        ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]
    if not set(cols_for_eda) <= (set(df.columns)):
        raise ValueError(
            f"ERROR: df does not contain all expected columns {cols_for_eda}"
            )

    eda_path = os.path.join(dst_path, "eda")
    os.makedirs(eda_path, exist_ok=True)

    # Plot `Churn` histogram
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.xlabel("Churn")
    plt.ylabel("Count")
    fig_fpath = os.path.join(eda_path, "churn_hist.png")
    plt.savefig(fig_fpath)
    plt.close()
    print(f"saved `Churn` hist {fig_fpath}")

    # Plot `Customer_Age` histogram
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.xlabel("Customer Age")
    plt.ylabel("Count")
    fig_fpath = os.path.join(eda_path, "customer_age_hist.png")
    plt.savefig(fig_fpath)
    plt.close()
    print(f"saved `Custormer_Age` hist @{fig_fpath}")

    # Plot `Marital_Status` bar
    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.xlabel("Marital Status")
    plt.ylabel("Frequency")
    fig_fpath = os.path.join(eda_path, "marital_status_bar.png")
    plt.savefig(fig_fpath)
    plt.close()
    print(f"saved `Marital_Status` bar plot @{fig_fpath}")

    # Plot `Total_Trans_Ct` distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    fig_fpath = os.path.join(eda_path, "total_trans_ct_distri.png")
    plt.savefig(fig_fpath)
    plt.close()
    print(f"saved `Total_Trans_Ct` distribution @{fig_fpath}")

    # Plot correlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    fig_fpath = os.path.join(eda_path, "correlation.png")
    plt.savefig(fig_fpath)
    plt.close()
    print(f"saved correlation plot @{fig_fpath}")


def encoder_helper(
    df: pd.DataFrame,
    category_lst: List[str],
    response: str
):
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
    if response not in df.columns:
        raise ValueError(f"ERROR: df does not contain `{response}` column")

    missing_cols = list(set(category_lst) - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"ERROR: df does not contain {missing_cols} column(s)"
        )

    for category in category_lst:
        prop_dict = df.groupby(category)[response].mean().to_dict()
        df[f"{category}_{response}"] = df[category].map(prop_dict)

    return df


def perform_feature_engineering(
    df: pd.DataFrame,
    response: str
):
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
    if response not in df.columns:
        raise ValueError(f"ERROR: df does not contain `{response}` column")

    # Encode categorical features
    # NOTE: mean value is calculated on the entire dataset --> leakage
    # Leave it as it is to be consistent with instructions of Udemy project
    df = encoder_helper(df, CATEGORICAL_COLS, response)

    missing_cols = list(set(FEATURE_LIST) - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"ERROR: df does not contain {missing_cols} column(s)"
        )

    X = df[FEATURE_LIST]
    y = df[response]

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_preds_lr: np.ndarray,
    y_train_preds_rf: np.ndarray,
    y_test_preds_lr: np.ndarray,
    y_test_preds_rf: np.ndarray,
    dst_path: str = "./images"
):
    """
    Produces classification report for training and testing results
    and stores report as image in `images` folder

    Parameters
    ----------
    y_train : np.ndarray
        Training response values
    y_test : np.ndarray
        Test response values
    y_train_preds_lr : np.ndarray
        Training predictions from logistic regression
    y_train_preds_rf : np.ndarray
        Training predictions from random forest
    y_test_preds_lr : np.ndarray
        Test predictions from logistic regression
    y_test_preds_rf : np.ndarray
        Test predictions from random forest
    dst_path : str, default="./images"
        Folder where to save figures.
        `results` folder will be created in `dst_path` if not existing yet

    Returns
    -------
    None
    """
    # pylint: disable=too-many-arguments

    # Check shape of arrays
    if not (y_train.ndim == y_train_preds_lr.ndim == y_train_preds_rf.ndim
        == y_test.ndim == y_test_preds_lr.ndim == y_test_preds_rf.ndim
        == 1):
        raise ValueError("ERROR: arrs must be 1D")

    if not (y_train.shape == y_train_preds_lr.shape == y_train_preds_rf.shape and
            y_test.shape == y_test_preds_lr.shape == y_test_preds_rf.shape):
        raise ValueError("ERROR: train arrs and test arrs must have same shape")

    results_path = os.path.join(dst_path, "results")
    os.makedirs(results_path, exist_ok=True)

    preds = [
        ["lr", y_train_preds_lr, y_test_preds_lr],
        ["rf", y_train_preds_rf, y_test_preds_rf]
    ]
    for model_name, y_train_preds, y_test_preds in preds:
        plt.rc('figure', figsize=(5, 5))
        plt.text(
            0.01, 1.25, str('Random Forest Train'),
            {'fontsize': 10}, fontproperties='monospace'
        )
        plt.text(
            0.01, 0.05, str(classification_report(y_test, y_test_preds)),
            {'fontsize': 10}, fontproperties='monospace'
        )
        plt.text(
            0.01, 0.6, str('Random Forest Test'),
            {'fontsize': 10}, fontproperties='monospace'
        )
        plt.text(
            0.01, 0.7, str(classification_report(y_train, y_train_preds)),
            {'fontsize': 10}, fontproperties='monospace'
        )
        plt.axis('off')
        fpath = os.path.join(results_path, f"{model_name}_results_train.png")
        plt.savefig(fpath)
        plt.close()

        print(f"saved {model_name} classfication results @{fpath}")


def feature_importance_plot(
    model: Any,
    X_data: pd.DataFrame,
    dst_path: str = "."
):
    """
    Creates and stores the feature importances in path

    Parameters
    ----------
    model: Any
        Model object containing `feature_importances_` attributes

    X_data : pd.DataFrame
        DataFrame of X values

    dst_path : str, default="."
        Folder where to save figures.
        `results` folder will be created in `dst_path` if not existing yet

    Returns
    -------
        None
    """
    # Check that model has the attribute `feature_importances_`
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "ERROR: `model` does not have the attribute `feature importances`"
        )

    results_path = os.path.join(dst_path, "results")
    os.makedirs(results_path, exist_ok=True)

    ##################
    # Plot SHAP values
    ##################
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    shap_fpath = os.path.join(results_path, "shap_values.png")
    plt.savefig(shap_fpath)
    plt.close()
    print(f"saved shap values plot @{shap_fpath}")

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
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    feat_imp_fpath = os.path.join(results_path, "feature_importances.png")
    plt.savefig(feat_imp_fpath)
    plt.close()
    print(f"saved feature importances plot @{feat_imp_fpath}")


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    models_path: str = "./models",
    images_path: str = "./images",
):
    """
    Train, store model results: images + scores, and store models

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        X train and test data
    y_train, y_test : pd.Series
        y train and test data
    models_path: str, default="./models"
        Folder where to model artifacts
    images_path: str, default="./images"
        Folder where to save ROC curves plot

    Returns
    -------
    None
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    os.makedirs(models_path, exist_ok=True)

    results_path = os.path.join(images_path, "results")
    os.makedirs(results_path, exist_ok=True)

    # Grid Search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    #######
    # Train
    #######
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_

    lrc.fit(X_train, y_train)

    # Save models
    models = [
        ["logistic", lrc],
        ["rfc", rfc]
    ]
    for name, model in models:
        fpath = os.path.join(models_path, f"{name}_clf.pkl")
        dump(model, fpath)
        print(f"saved {name} model artifact @{fpath}")

    #######
    # Eval
    #######
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save ROC curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    fpath = os.path.join(results_path, "roc_curves.png")
    plt.savefig(fpath)
    plt.close()
    print(f"saved ROC curves @{fpath}")

    # Model results
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    )

    # Shap + Feature importances
    feature_importance_plot(
        cv_rfc.best_estimator_, X_test, dst_path=images_path
    )


def main():
    """Function used to run the pipeline"""
    print("START PIPELINE")

    print("importing data...")
    df = import_data("./data/bank_data.csv")

    print("adding churn column to dataframe...")
    df = add_churn_column_to_df(df)

    print("performing eda...")
    perform_eda(df)

    print("performing features engineering...")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")

    print("training models...")
    train_models(X_train, X_test, y_train, y_test)

    print("END PIPELINE")


if __name__ == "__main__":
    main()
