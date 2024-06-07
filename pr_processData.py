import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class PreprocessData:
    """
    A class used to preprocess the dataset for salary predictions of data professionals.

    Attributes:
    -----------
    df : pd.DataFrame
        The dataset to be preprocessed.
    
    Methods:
    --------
    __init__(df: pd.DataFrame) -> None:
        Initializes the PreprocessData class with dataset.
    
    scaling() -> pd.DataFrame:
        Scales numerical features in the dataset using MinMaxScaler.
    
    one_hot_encode(columns: list) -> pd.DataFrame:
        Applies one-hot encoding to the specified columns.

    label_encode(columns: list) -> pd.DataFrame:
        Applies label encoding to the specified columns.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the PreprocessData class with the dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be preprocessed.
        """
        self.df = df

    def scaling(self) -> pd.DataFrame:
        """
        Scales numerical features in the dataset using MinMaxScaler.

        Returns:
        --------
        pd.DataFrane
            The dataset with scaled numerical features.
        """
        scaler = MinMaxScaler()
        numberical_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numberical_cols] = scaler.fit_transform(self.df[numberical_cols])

        return self.df

    def one_hot_encode(self, columns: list) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified columns.

        Parameters:
        -----------
        columns : list
            The list of columns to be one-hot encoded.
        
        Returns:
        pd.DataFrame
            The dataset with one-hot encoded columns.
        """
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return self.df

    def label_encode(self, columns: list) -> pd.DataFrame:
        """
        Applies label encoding to the specified columns.

        Parameters:
        -----------
        columns : list
            The list of columns to be lable encoded.

        Returns:
        --------
        pd.DataFrame
            The dataset with lable encoded columns.
        """
        label_encoders = {}
        for column in columns:
            label_encoders[columns] = LabelEncoder()
            self.df[columns] = label_encoders[columns].fit_transform(self.df[columns])

        return self.df