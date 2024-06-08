from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import pandas as pd

class TuningHyperparameters:
    """
    A class used to perform hyperparameter tuning for various models.

    Models:
    -------
    - Support Vector Regression (SVR)
    - Linear Regression
    - Random Forest Regression

    Attributes:
    -----------
    svr_params_gird: dict
        Hyperparameters for tuning SVR.
    linear_regression_params_grid: dict
        Hyperparameters for tuning Linear Regression.
    random_forest_regression_params_grid: dict
        Hyperparameters for tuning Random Forest Regression.

    Methods:
    --------
    __init__(learning_rate: float = 0.02) -> None:
        Initializes the TuningHyperparameters class with the specified learning rate.

    rune_svr(X: pd.DataFrame, y: pd.Series) -> dict:
        Tunes hyperparameters for Support Vector Regression model.
    
    rune_linear_regression(X: pd.DataFrame, y: pd.Series) -> dict:
        Tunes hyperparameters for Linear Regression model.
    
    rune_random_forest_regression(X: pd.DataFrame, y: pd.Series) -> dict:
        Tunes hyperparameters for Random Forest Regression model.
    """
    def __init__(self, learning_rate: float = 0.02) -> None:
        """
        Initializes the TuningHyperparameters class with the specified learning rate.

        Parameters:
        -----------
        learning_rate : float, optional
            Learning rate for model where applicable (default = 0.02).
        """
        self.learning_rate = learning_rate
        self.svr_param_grid = {
            'C': [0.001, 0.01],
            'kernel': ['linear'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.1, 0.5],
            'tol': [1e-3, 1e-4, 1e-5],
            'epsilon': [0.1, 0.2, 0.5],
            'shrinking': [True, False],
            'cache_size': [200],
            'verbose': [False],
            'max_iter': [1000, -1]
        }

        self.linear_regression_param_grid = {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'n_jobs': [-1, 1, 2, 4],
            'positive': [True, False]
        }

        # self.random_forest_param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'criterion': ['squared_error'],
        #     'max_depth': [None, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'min_weight_fraction_leaf': [0.0, 0.1],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_leaf_nodes': [None, 10, 20],
        #     'min_impurity_decrease': [0.0, 0.1, 0.2],
        #     'bootstrap': [True, False],
        #     'oob_score': [True, False],
        #     'n_jobs': [-1, 1, 2, 4],
        #     'random_state': [42],
        #     'verbose': [0, 1],
        #     'warm_start': [True, False],
        #     'ccp_alpha': [0.0, 0.1, 0.2],
        #     'max_samples': [None, 0.8, 0.9]
        # }
        self.random_forest_param_grid = {
            'n_estimators': [100],
            'criterion': ['squared_error'],
            'max_depth': [10],
            'min_samples_split': [5],
            'min_samples_leaf': [24],
            'min_weight_fraction_leaf': [0.1],
            'max_features': ['log2'],
            'max_leaf_nodes': [10],
            'min_impurity_decrease': [0.1],
            'bootstrap': [True],
            'oob_score': [True],
            'n_jobs': [2],
            'random_state': [42],
            'verbose': [1],
            'warm_start': [True],
            'ccp_alpha': [0.1],
            'max_samples': [0.9]
        }
    
    
    def tune_svr(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Tunes hyperparameters for Support Vector Regression model.

        Parameters:
        -----------
        X : pd.DataFrame
            The features set.

        y : pd.Series
            The target variable.


        Returns:
        --------
        dict
            The best parameters for SVR model.
        """
        svr = SVR()
        svr_grid_search = GridSearchCV(estimator=svr, param_grid=self.svr_param_grid, cv=5, n_jobs=-1, verbose=1)
        svr_grid_search.fit(X, y)
        return svr_grid_search.best_params_
    
    def tune_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Tunes hyperparameters for Linear Regression model.

        Parameters:
        -----------
        X : pd.DataFrame
            The features set.

        y : pd.Series
            The target variable.
        
        Returns:
        --------
        dict
            The best parameters for Linear Regression model.
        """
        linear_regression = LinearRegression()
        linear_regression_grid_search = GridSearchCV(estimator=linear_regression, param_grid=self.linear_regression_param_grid)
        linear_regression_grid_search.fit(X, y)

        return linear_regression_grid_search.best_params_
    
    def tune_random_forest_regression(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Tunes hyperparameters for Random Forest Regression model.

        Parameters:
        -----------
        X : pd.DataFrame
            The features set.

        y : pd.Series
            The target variable.
        
        Returns:
        --------
        dict
            The best parameters for Random Forest Regression model.
        """
        random_forest_regression = RandomForestRegressor()
        random_forest_regression_grid_search = GridSearchCV(estimator=random_forest_regression, param_grid=self.random_forest_param_grid)
        random_forest_regression_grid_search.fit(X, y)

        return random_forest_regression_grid_search.best_params_