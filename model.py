from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import joblib

from tuning_Hyperprameters  import TuningHyperparameters

class MLModel(TuningHyperparameters):
    """
    A class for building and training machine learning models using tuned hyperparameters.

    Algorithms:
    -----------
    - Support Vector Regression (SVR)
    - Linear Regression
    - Random Forest Regression

    Attributes:
    -----------
    learning_rate : float
        Learning rate for model where applicable (default = 0.02).
    
    model : Any
        The machine learning model to be used.

    Methods:
    --------
    __init__(self, learning_rate: float = 0.02) -> None:
        Initializes the MLModel class with the specified learning rate.
    
    train_model(X: pd.DataFrame, y: pd.Series, algorithm: str) -> None:
        Trains the specified machine learning algorithm with the best hyperparameters.
    
    save_model(model_path: str) -> None:
        Saves the trained machine learning model to a joblib file.
    """
    def __init__(self, learning_rate: float = 0.02) -> None:
        """
        Initialize the MLModel class with the specified learning rate

        Parameters:
        -----------
        learning_rate : float, optional
            Learning rate for model where applicable (default = 0.02).
        """
        super().__init__(learning_rate)
        self.model = None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, algorithm: str) -> None:
        """
        Train the specified machine learning algorithm with the best hyperparameters.

        Parameters:
        -----------
        X : pd.DataFrame
            The features set.
        
        y : pd.Series
            The target variable.

        algorithm : str
            The alogrithm to be used for training. Options: ['svr', 'linear_regression', 'random_forest_regression']
        """
        if algorithm == 'SVR':
            best_params = self.tune_svr(X, y)
            self.model = SVR(**best_params)
        elif algorithm == 'linear_regression':
            best_params = self.tune_linear_regression(X, y)
            self.model = LinearRegression(**best_params)
        elif algorithm == 'random_forest_regression':
            best_params = self.tune_random_forest_regression(X, y)
            self.model = RandomForestRegressor(**best_params)
        else:
            raise ValueError(f'Invalid algorithm: {algorithm}')
        
        self.model.fit(X, y)

    def save_model(self, model_path: str) -> None:
        """
        Saves the trained machine learning model to a joblib file.

        Parameters:
        -----------
        model_path : str
            The path to the joblib file where the model will be saved.
        """
        self.model = joblib.dump(self.model, model_path)