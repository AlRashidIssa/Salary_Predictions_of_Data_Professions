from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
import pandas as pd
class Evaluation:
    """
    Class for evaluating machine learning models.

    Attributes:
    -----------
    y_true : pd.Series
        The true values.
    y_pred : pd.Series
        The predicted values.
    r2_score_val : float
        The R-squared score.
    mae_val : float
        The Mean Absolute Error.
    mdae_val : float
        The Median Absolute Error.
    mse_val : float
        The Mean Squared Error.

    Methods:
    --------
    __init__(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        Initializes the Evaluation class with true and predicted values.
    
    r2(self) -> float:
        Calculates and returns the R-squared score.
    
    MAE(self) -> float:
        Calculates and returns the Mean Absolute Error.
    
    MdAE(self) -> float:
        Calculates and returns the Median Absolute Error.
    
    MSE(self) -> float:
        Calculates and returns the Mean Squared Error.
    """

    def __init__(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        """
        Initializes the Evaluation class with true and predicted values.

        Parameters:
        -----------
        y_true : pd.Series
            The true values.
        
        y_pred : pd.Series
            The predicted values.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.r2_score_val = None
        self.mae_val = None
        self.mdae_val = None
        self.mse_val = None

    def r2(self) -> float:
        """
        Calculates and returns the R-squared score.

        Returns:
        --------
        float
            The R-squared score.
        """
        self.r2_score_val = r2_score(self.y_true, self.y_pred)
        return self.r2_score_val

    def MAE(self) -> float:
        """
        Calculates and returns the Mean Absolute Error.

        Returns:
        --------
        float
            The Mean Absolute Error.
        """
        self.mae_val = mean_absolute_error(self.y_true, self.y_pred)
        return self.mae_val

    def MdAE(self) -> float:
        """
        Calculates and returns the Median Absolute Error.

        Returns:
        --------
        float
            The Median Absolute Error.
        """
        self.mdae_val = median_absolute_error(self.y_true, self.y_pred)
        return self.mdae_val

    def MSE(self) -> float:
        """
        Calculates and returns the Mean Squared Error.

        Returns:
        --------
        float
            The Mean Squared Error.
        """
        self.mse_val = mean_squared_error(self.y_true, self.y_pred)
        return self.mse_val
