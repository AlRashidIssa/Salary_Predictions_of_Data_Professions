import pandas as pd

class CleaningData:
    """
    A class used to clean and preprocess the dataset for salary predictions of data professionals.

    Attributes:
    -----------
    df : pd.DataFrame
        The dataset to be cleaned.

    Methods:
    --------
    __init__(df: pd.DataFrame) -> None:
        Initializes the CleaningData class with the dataset.

    remove_duplicates() -> pd.DataFrame:
        Removes duplicate rows from the dataset.

    handle_outliers(column: str, method: str = 'IQR') -> pd.DataFrame:
        Handles outliers in the specified column using the specified method.

    replace_missing_values(strategy: str = 'mean') -> pd.DataFrame:
        Replaces missing values in the dataset using the specified strategy.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the CleaningData class with the dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be cleaned.
        """
        self.df = df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Removes duplicate rows from the dataset.

        Returns:
        --------
        pd.DataFrame
            The dataset with duplicate rows removed.
        """
        self.df = self.df.drop_duplicates()
        return self.df

    def handle_outliers(self, column: str, method: str = 'IQR') -> pd.DataFrame:
        """
        Handles outliers in the specified column using the specified method.

        Parameters:
        -----------
        column : str
            The column in which to handle outliers.
        method : str, optional
            The method to use for handling outliers (default is 'IQR').

        Returns:
        --------
        pd.DataFrame
            The dataset with outliers handled.
        """
        if method == 'IQR':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        elif method == 'z-score':
            from scipy.stats import zscore
            self.df = self.df[(zscore(self.df[column]) < 3).all(axis=1)]
        return self.df

    def replace_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        """
        Replaces missing values in the dataset using the specified strategy.

        Parameters:
        -----------
        strategy : str, optional
            The strategy to use for replacing missing values (default is 'mean').
            Options: 'mean', 'median', 'mode'

        Returns:
        --------
        pd.DataFrame
            The dataset with missing values replaced.
        """
        if strategy == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif strategy == 'median':
            self.df = self.df.fillna(self.df.median())
        elif strategy == 'mode':
            self.df = self.df.fillna(self.df.mode().iloc[0])
        return self.df

# # Usage example
# if __name__ == "__main__":
#     # Load the dataset
#     df = pd.read_csv('data_professions.csv')

#     # Initialize the CleaningData class
#     cleaner = CleaningData(df)

#     # Remove duplicates
#     df_no_duplicates = cleaner.remove_duplicates()

#     # Handle outliers in the 'SALARY' column using IQR method
#     df_no_outliers = cleaner.handle_outliers(column='SALARY')

#     # Replace missing values using the mean strategy
#     cleaned_df = cleaner.replace_missing_values(strategy='mean')

#     # Display the cleaned data
#     print(cleaned_df.head())
