from pandas import DataFrame
from pr_processData import PreprocessData
from prediction import Prediction

import numpy as np
import pandas as pd

class PredictionDeployment():
    def __init__(self, df: DataFrame, path_model: str) -> None:
        self.df = df
        self.path_model = path_model


        # Define a dictionary with column names and empty data
        data_set = {
            'AGE': pd.Series(dtype='float64'),
            'LEAVES USED': pd.Series(dtype='float64'),
            'LEAVES REMAINING': pd.Series(dtype='float64'),
            'RATINGS': pd.Series(dtype='float64'),
            'PAST EXP': pd.Series(dtype='float64'),
            'SEX_F': pd.Series(dtype='int64'),
            'SEX_M': pd.Series(dtype='int64'),
            'DESIGNATION_Analyst': pd.Series(dtype='int64'),
            'DESIGNATION_Associate': pd.Series(dtype='int64'),
            'DESIGNATION_Director': pd.Series(dtype='int64'),
            'DESIGNATION_Manager': pd.Series(dtype='int64'),
            'DESIGNATION_Senior Analyst': pd.Series(dtype='int64'),
            'DESIGNATION_Senior Manager': pd.Series(dtype='int64'),
            'UNIT_Finance': pd.Series(dtype='int64'),
            'UNIT_IT': pd.Series(dtype='int64'),
            'UNIT_Management': pd.Series(dtype='int64'),
            'UNIT_Marketing': pd.Series(dtype='int64'),
            'UNIT_Operations': pd.Series(dtype='int64'),
            'UNIT_Web': pd.Series(dtype='int64'),
            'jo_Years': pd.Series(dtype='float64'),
            'total_exp': pd.Series(dtype='float64')
        }

        # Create an empty DataFrame with the specified columns
        df_empty = pd.DataFrame(data_set)
        
        # Preprocess the data
        preprocessor = PreprocessData(df=self.df)
        self.preprocessor_df = preprocessor.one_hot_encode(columns=['SEX', 'DESIGNATION', 'UNIT'])
        preprocessor_df = preprocessor.calculate_jo_years(doj_col='DOJ', current_date_col='CURRENT DATE')
        self.preprocessor_df['total_exp'] = np.log2(self.preprocessor_df['PAST EXP'] + self.preprocessor_df['jo_Years'])
        self.preprocessor_df = preprocessor.scaling()
        self.preprocessor_df.dropna(inplace=True)
        self.preprocessor_df = self.preprocessor_df.drop(columns=['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'])

        # Check if any column names are not present in the empty DataFrame
        for col in preprocessor_df.columns:
            if col not in df_empty.columns:
                print("Column '{}' is not present in df_empty".format(col))

        # Populating empty DataFrame
        for col in df_empty.columns:
            for col2 in preprocessor_df.columns:
                if col == col2:
                    df_empty[col] = preprocessor_df[col2].index[0]

        self.df_empty = df_empty.fillna(value=0)
        
    def predict(self) -> pd.Series:
        # Predict
        prediction = Prediction(self.df_empty)
        prediction.load_model(self.path_model)
        new_salary = prediction.predict()
        
        return new_salary

# Usage example
input_user = {
    'FIRST NAME': 'Rasid',
    'LAST NAME': "Youse",
    'SEX': 'M',
    'DOJ': "7-28-2014",
    'CURRENT DATE': "01-07-2016",
    'DESIGNATION': 'Manager',
    'AGE': 50,
    'UNIT': "Finance",
    'LEAVES USED': 12,
    'LEAVES REMAINING': 20,
    'RATINGS': 3000,
    'PAST EXP': 30
}

data = pd.DataFrame(input_user, index=[0])
deployment = PredictionDeployment(df=data, 
                                  path_model="/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib")
new_salary, preprocessed_data = deployment.predict()
print(new_salary)
print(preprocessed_data)
