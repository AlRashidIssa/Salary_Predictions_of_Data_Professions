import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

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
        scaler = MinMaxScaler(feature_range=(0, 1))
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df.loc[:, numerical_cols] = scaler.fit_transform(self.df.loc[:, numerical_cols])
        return self.df

    def one_hot_encode(self, columns: list) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified columns using scikit-learn's OneHotEncoder.

        Parameters:
        -----------
        columns : list
            The list of columns to be one-hot encoded.
        
        Returns:
        --------
        pd.DataFrame
            The dataset with one-hot encoded columns.
        """
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        
        # Fit and transform the specified columns
        encoded_data = encoder.fit_transform(self.df[columns])
        
        # Convert the encoded data to a DataFrame
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))
        
        # Drop the original columns and concatenate the new encoded DataFrame
        self.df = pd.concat([self.df.drop(columns=columns), encoded_df], axis=1)
        
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
            self.df.loc[:, column] = label_encoders[column].fit_transform(df.loc[:, column])
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
    def calculate_jo_years(self, doj_col: str, current_date_col: str) -> pd.DataFrame:
        """
        Converts DOJ and CURRENT DATE columns to datetime and calculates the number of years an employee has been with the company.

        Parameters:
        -----------
        doj_col : str
            The column name for Date of Joining.
        current_date_col : str
            The column name for Current Date.
        
        Returns:
        --------
        pd.DataFrame
            The dataset with an additional 'jo_Years' column.
        """
        # Convert columns to datetime
        self.df[doj_col] = pd.to_datetime(self.df[doj_col])
        self.df[current_date_col] = pd.to_datetime(self.df[current_date_col])
        
        # Calculate the number of years with the company
        self.df['jo_Years'] = (self.df[current_date_col] - self.df[doj_col]).dt.days / 365.25
        
        return self.df