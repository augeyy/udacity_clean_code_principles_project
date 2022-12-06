# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the final project of the first course of the **Udacity Machine Learning DevOps Engineer Nanodegree**.  
In this project, the goal is to write a data science pipeline code to predict credit card customers that are the most likely to churn.  
Focus is put on writing code following best software engineering practices (modular, documented and tested code).

## Files and data description

### Project structure

├── README.md              <- The top-level README for developers using this project.
│
├── data                   <- Data from third party sources.
│   └── bank_data.csv      
│
├── churn_notebook.ipynb   <- Provided notebook used as the reference to write `churn_library` module
│
├── churn_library.py       <- Module to run data science pipeline
│
├── test_churn_library.py  <- File to test the functions contained in `churn_library` module
│
├── models                 <- Trained and serialized models
│
├── images                 <- Generated analysis as png, PDF, etc.
│
├── logs                   <- Log files
│
├── requirements.txt       <- The requirements file for reproducing the prod environment
│
└── requirements_dev.txt   <- The requirements file for reproducing the development environment
                              eg. tests, linter, etc

### Data description

Data used in this project are credit card customers data available on [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code)

## Running Files

### Running the main file

Create a Python virtual environment
```
python3.8 -m venv .venv
```

Activate the virtual environment
```
source .venv/bin/activate
```

Install the dependencies
```
pip install -r requirements.txt
```

Run main program
```
python3.8 churn_library.py
```

### Running the tests

Create a Python virtual environment dedicated to development
```
python3.8 -m venv .venv_dev
```

Activate the virtual environment
```
source .venv_dev/bin/activate
```

Install the dependencies
```
pip install -r requirements_dev.txt
```

Run the tests
```
pytest -s -v
```

