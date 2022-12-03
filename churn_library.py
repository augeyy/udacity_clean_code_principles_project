# library doc string


# import libraries
import os
os.environ["QT_QPA_PLATFORM"]="offscreen"

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logging.basicConfig(
    filename="./logs/churn_library.log",
    level = logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s")


# Columns to keep training model features engineering
CATEGORICAL_COLS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

QUANT_COLS = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

FEATURE_LIST = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn', 
    'Income_Category_Churn',
    'Card_Category_Churn'
]


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


def make_churn_column(df):
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
    folder_pth : str, default="."
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
            "ERROR: df does not contain all expected columns {cols_for_eda}"
        )
        raise ValueError()

    images_pth = os.path.join(dst_pth, "images")
    if not os.path.exists(images_pth):
        os.makedirs(images_pth)
        logging.info("SUCCESS: using new directory @{images_pth}")
    else:
        logging.info("SUCCESS: using existing directory @{images_pth}")
    
    # Plot `Churn` histogram
    plt.figure(figsize=(20,10)) 
    df["Churn"].hist()
    fig_fpath = os.path.join(images_pth, "churn_hist.png")
    plt.savefig(fig_fpath)
    logging.info(f"SUCCESS: saved `Churn` hist @{fig_fpath}")


    # Plot `Customer_Age` histogram
    plt.figure(figsize=(20,10)) 
    df["Customer_Age"].hist()
    fig_fpath = os.path.join(images_pth, "customer_age_hist.png")
    plt.savefig(fig_fpath)
    logging.info(f"SUCCESS: saved `Custormer_Age` hist @{fig_fpath}")

    # Plot `Marital_Status` bar
    plt.figure(figsize=(20,10)) 
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    fig_fpath = os.path.join(images_pth, "marital_status_bar.png")
    plt.savefig(fig_fpath)
    logging.info(f"SUCCESS: saved `Marital_Status` bar plot @{fig_fpath}")

    # Plot `Total_Trans_Ct` distribution
    plt.figure(figsize=(20,10)) 
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    fig_fpath = os.path.join(images_pth, "total_trans_ct_distri.png")
    plt.savefig(fig_fpath)
    logging.info("SUCCESS: saved `Total_Trans_Ct` distribution @{fig_fpath}")

    # Plot correlation
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths = 2)
    fig_fpath = os.path.join(images_pth, "correlation.png")
    plt.savefig(fig_fpath)
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
    X_train : ndarray
        X training data
    X_test : ndarray
        X testing data
    y_train : ndarray
        y training data
    y_test : ndarray
        y testing data
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


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
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

    Returns
    -------
    None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores the feature importances in pth

    Parameters
    ----------
    model
        Model object containing `feature_importances_` attributes

    X_data : pd.DataFrame
        DataFrame of X values

    output_pth : str
        Path to store the figure

    Returns
    -------
        None
    """
    pass

def train_models(X_train, X_test, y_train, y_test):
    """
    Train, store model results: images + scores, and store models

    Parameters
    ----------
    X_train : ndarray
        X training data
    X_test : ndarray
        X testing data
    y_train : ndarray
        y training data
    y_test : ndarray
        y testing data

    Returns
    ----------
    None
    """
    pass
