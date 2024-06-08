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
        Initializes the PreprocessData class with the dataset.
    
    scaling() -> pd.DataFrame:
        Scales numerical features in the dataset using MinMaxScaler.
    
    one_hot_encode(columns: list) -> pd.DataFrame:
        Applies one-hot encoding to the specified columns.

    label_encode(columns: list) -> pd.DataFrame:
        Applies label encoding to the specified columns.
        
    zero_encode(columns: list) -> pd.DataFrame:
        Encodes specified columns to zero.
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
        pd.DataFrame
            The dataset with scaled numerical features.
        """
        scaler = MinMaxScaler()
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        return self.df

    def one_hot_encode(self, columns: list) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified columns.

        Parameters:
        -----------
        columns : list
            The list of columns to be one-hot encoded.
        
        Returns:
        --------
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
            The list of columns to be label encoded.

        Returns:
        --------
        pd.DataFrame
            The dataset with label encoded columns.
        """
        label_encoders = {}
        for column in columns:
            label_encoders[column] = LabelEncoder()
            self.df[column] = label_encoders[column].fit_transform(self.df[column])
        return self.df

    def zero_encode(self, columns: list) -> pd.DataFrame:
        """
        Encodes specified columns to zero.

        Parameters:
        -----------
        columns : list
            The list of columns to be encoded to zero.

        Returns:
        --------
        pd.DataFrame
            The dataset with specified columns encoded to zero.
        """
        for column in columns:
            self.df[column] = 0
        return self.df