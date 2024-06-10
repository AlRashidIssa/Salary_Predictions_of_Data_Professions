from pandas import DataFrame
import numpy as np
import pandas as pd
from pr_processData import PreprocessData
from prediction import Prediction

class PredictionDeployment:
    """
    A class to deploy a salary prediction model for data professionals.

    Attributes:
    -----------
    df : DataFrame
        The dataset containing the input data for prediction.
    path_model : str
        The path to the saved prediction model.

    Methods:
    --------
    preprocess() -> DataFrame:
        Preprocesses the input data for prediction.
    
    predict() -> pd.Series:
        Predicts the salary based on the preprocessed input data using the saved model.
    """

    def __init__(self, df: DataFrame, path_model: str) -> None:
        """
        Initializes the PredictionDeployment class with the dataset and model path.

        Parameters:
        -----------
        df : DataFrame
            The dataset containing the input data for prediction.
        path_model : str
            The path to the saved prediction model.
        """
        self.df = df
        self.path_model = path_model

    def preprocess(self) -> DataFrame:
        """
        Preprocesses the input data for prediction.

        Returns:
        --------
        DataFrame
            The preprocessed dataset ready for prediction.
        """
        # Define a dictionary with column names and empty data
        data_set = {
            'AGE': pd.Series(dtype='float64'),
            'LEAVES USED': pd.Series(dtype='float64'),
            'LEAVES REMAINING': pd.Series(dtype='float64'),
            'RATINGS': pd.Series(dtype='float64'),
            'PAST EXP': pd.Series(dtype='float64'),
            'SEX_F': pd.Series(dtype='float64'),
            'SEX_M': pd.Series(dtype='float64'),
            'DESIGNATION_Analyst': pd.Series(dtype='float64'),
            'DESIGNATION_Associate': pd.Series(dtype='float64'),
            'DESIGNATION_Director': pd.Series(dtype='float64'),
            'DESIGNATION_Manager': pd.Series(dtype='float64'),
            'DESIGNATION_Senior Analyst': pd.Series(dtype='float64'),
            'DESIGNATION_Senior Manager': pd.Series(dtype='float64'),
            'UNIT_Finance': pd.Series(dtype='float64'),
            'UNIT_IT': pd.Series(dtype='float64'),
            'UNIT_Management': pd.Series(dtype='float64'),
            'UNIT_Marketing': pd.Series(dtype='float64'),
            'UNIT_Operations': pd.Series(dtype='float64'),
            'UNIT_Web': pd.Series(dtype='float64'),
            'jo_Years': pd.Series(dtype='float64'),
            'total_exp': pd.Series(dtype='float64')
        }

        # Create an empty DataFrame with the specified columns
        df_empty = pd.DataFrame(data_set)

        # Preprocess the data
        preprocessor = PreprocessData(df=self.df)
        preprocessor_df = preprocessor.calculate_jo_years(doj_col='DOJ', current_date_col='CURRENT DATE')
        preprocessor_df = preprocessor.one_hot_encode(columns=['SEX', 'DESIGNATION', 'UNIT'])
        preprocessor_df['total_exp'] = np.log2(preprocessor_df['PAST EXP'] + preprocessor_df['jo_Years'])
        preprocessor_df.dropna(inplace=True)
        preprocessor_df = preprocessor.manual_scaling()

        # Check if any column names are not present in the empty DataFrame
        for col in preprocessor_df.columns:
            if col not in df_empty.columns:
                print("Column '{}' is not present in df_empty".format(col))

        # Concatenate preprocessor_df and df_empty
        df = pd.concat([preprocessor_df, df_empty], ignore_index=True)

        # Drop unnecessary columns from df
        df = df.drop(columns=['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'])

        # Fill NaN values with 0
        df.fillna(value=0, inplace=True)

        return df

    def predict(self) -> pd.Series:
        """
        Predicts the salary based on the preprocessed input data using the saved model.

        Returns:
        --------
        pd.Series
            The predicted salary.
        """
        # Preprocess the input data
        processed_df = self.preprocess()

        # Predict
        prediction = Prediction(processed_df)
        prediction.load_model(self.path_model)
        new_salary = prediction.predict()

        return new_salary

# Usage example
input_user = {
    'FIRST NAME': 'Rasid',
    'LAST NAME': "Youse",
    'SEX': 'M',
    'DOJ': "7-28-2004",
    'CURRENT DATE': "01-07-2016",
    'DESIGNATION': 'Manager',
    'AGE': 50,
    'UNIT': "Finance",
    'LEAVES USED': 12,
    'LEAVES REMAINING': 20,
    'RATINGS': 3000,
    'PAST EXP': 20
}

def unscale_salary(scaled_value, min_salary=40001.0, max_salary=388112.0):
    return scaled_value * (max_salary - min_salary) + min_salary

data = pd.DataFrame(input_user, index=[0])
deployment = PredictionDeployment(df=data, 
                                  path_model="/home/alrashidissa/Desktop/Internship/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib")
scalin_salary = deployment.predict()
new_salary = unscale_salary(scaled_value=scalin_salary)

print(int(new_salary / 100))
