from pandas import DataFrame
from pr_processData import PreprocessData
from prediction import Prediction
import pandas as pd

class PredictionDeployment():
    """
    A class for deployment of prediction models after preprocessing the data.

    Attributes:
    -----------
    df : DataFrame
        The dataset to be used for prediction.
    model_path : str
        The file path to the saved prediction model.

    Methods:
    --------
    __init__(df: DataFrame, model_path: str) -> None:
        Initializes the PredictionDeployment class with the dataset and model path.
    
    predict() -> pd.Series:
        Makes predictions using the loaded model on the preprocessed dataset.
    """
    def __init__(self, df: DataFrame, path_model: str) -> None:
        """
        Initializes the PredictionDeployment class with the dataset and model path.

        Parameters:
        -----------
        df : DataFrame
            The dataset to be used for prediction.
        model_path : str
            The file path to the saved prediction model.
        """
        self.df = df
        self.path_model = path_model

        # Preprocess the data
        preprocessor = PreprocessData(df=df)

        # Ensure that the dataset has the necessary columns for prediction
        expected_columns = ['FIRST NAME', 'LAST NAME', 'DOJ', 'DESIGNATION', 'AGE', 'UNIT', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS', 'PAST EXP']
        for column in expected_columns:
            if column not in df.columns:
                df[column] = ''  # Add missing columns with default value

        # encode specific columns
        preprocessor_df = preprocessor.zero_encode(columns=['FIRST NAME', 'LAST NAME'])
        if 'SEX' in df.columns:
            preprocessor_df = preprocessor.one_hot_encode(columns=['SEX'])
            # Ensure that all one-hot encoded columns are present
            expected_columns = ['SEX_M', 'SEX_F']
            for column in expected_columns:
                if column not in preprocessor_df.columns:
                    preprocessor_df[column] = 0  # Add missing columns with default value
        
        preprocessor_df = preprocessor.label_encode(columns=['DOJ', 'DESIGNATION', 'UNIT'])

        # Scaling all DataFrame
        scaling = PreprocessData(df=preprocessor_df) 
        self.scaling_df = scaling.scaling()

    def predict(self) -> pd.Series:
        """
        Makes predictions using the loaded model on the preprocessed dataset.

        Returns:
        --------
        pd.Series
            The predicted values.
        """
        # Predict
        prediction = Prediction(self.scaling_df)
        prediction.load_model(self.path_model)
        new_salary = prediction.predict()

        return new_salary



