from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pandas as pd
import joblib
from tuning_Hyperprameters import TuningHyperparameters

class MLModel(TuningHyperparameters):
    """
    A class for building and training machine learning models using tuned hyperparameters.

    Algorithms:
    -----------
    - Support Vector Regression (SVR)
    - Linear Regression
    - Random Forest Regression
    - LightGBM
    - XGBoost
    - CatBoost

    Attributes:
    -----------
    
    model : Any
        The machine learning model to be used.

    Methods:
    --------
    __init__(self, learning_rate: float = 0.02) -> None:
        Initializes the MLModel class with the specified learning rate.
    
    train_model(X: pd.DataFrame, y: pd.Series, algorithm: str, tune: bool = True) -> None:
        Trains the specified machine learning algorithm with the best hyperparameters.
    
    save_model(model_path: str) -> None:
        Saves the trained machine learning model to a joblib file.
    """
    def __init__(self) -> None:
        """
        Initialize the MLModel class with the specified learning rate
        """
        super().__init__()
        self.model = None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, algorithm: str, tune: bool = True) -> None:
        """
        Train the specified machine learning algorithm with the best hyperparameters.

        Parameters:
        -----------
        X : pd.DataFrame
            The features set.
        
        y : pd.Series
            The target variable.

        algorithm : str
            The algorithm to be used for training. Options: ['svr', 'linear_regression', 'random_forest_regression', 'lightgbm', 'xgboost', 'catboost']

        tune : bool, optional
            Whether to perform hyperparameter tuning (default = True).
        """
        if tune:
            if algorithm == 'svr':
                best_params = self.tune_svr(X, y)
                self.model = SVR(**best_params)
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'linear_regression':
                best_params = self.tune_linear_regression(X, y)
                self.model = LinearRegression(**best_params)
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'random_forest_regression':
                best_params = self.tune_random_forest_regression(X, y)
                self.model = RandomForestRegressor(**best_params)
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'lightgbm':
                # Assuming there's a method to tune LightGBM parameters
                best_params = self.tune_lightgbm(X, y)
                self.model = lgb.LGBMRegressor(**best_params)
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'xgboost':
                # Assuming there's a method to tune XGBoost parameters
                best_params = self.tune_xgboost(X, y)
                self.model = xgb.XGBRegressor(**best_params)
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'catboost':
                # Assuming there's a method to tune CatBoost parameters
                best_params = self.tune_catboost(X, y)
                self.model = cb.CatBoostRegressor(silent=True, **best_params)
                self.model.fit(X, y)
                return self.model
            else:
                raise ValueError(f'Invalid algorithm: {algorithm}')
        else:
            if algorithm == 'svr':
                self.model = SVR()
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'linear_regression':
                self.model = LinearRegression()
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'random_forest_regression':
                self.model = RandomForestRegressor()
                self.model.fit(X, y)
                return self.model                
            elif algorithm == 'lightgbm':
                self.model = lgb.LGBMRegressor()
                self.model.fit(X, y)
                return self.model
            elif algorithm == 'xgboost':
                self.model = xgb.XGBRegressor()                
                self.model.fit(X, y)
                return self.model                
            elif algorithm == 'catboost':
                self.model = cb.CatBoostRegressor(silent=True)
                self.model.fit(X, y)
                return self.model
            else:
                raise ValueError(f'Invalid algorithm: {algorithm}')
    
    def save_model(self, model,  model_path: str) -> None:
        """
        Saves the trained machine learning model to a joblib file.

        Parameters:
        -----------
        model_path : str
            The path to the joblib file where the model will be saved.
        model: Any
            The model machine learning models
        """
        joblib.dump(model, model_path)

# Example usage:
# model = MLModel()
# model.train_model(X_train, y_train, algorithm='lightgbm', tune=False)
# model.save_model('lgb_model.joblib')

