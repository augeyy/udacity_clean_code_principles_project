# library doc string


# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'


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
    pass


def perform_eda(df):
    """
    Perform EDA on DataFrame and save figures to the `images` folder

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which to perform EDA

    Returns
    -------
    None
    """
    pass


def encoder_helper(df, category_lst, response):
    """
    Encode categorical column into a new column with proportion
    of churn for each category

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
    pass


def perform_feature_engineering(df, response):
    """
    Perform feature engineering

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which to perform feature engineering

    response: str
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
