import joblib 
import pandas as pd


class Prediction():
    """
    A class used for loading a machine learning model and making predictions.

    Attributes:
    -----------
    model : Any
        The machine learning model to be used for predictions.
    
    Methods:
    --------
    __init__() -> None:
        Initialize the Prediction class and the parent PreprocessData class.
    
    load_model(model_path: str) -> None:
        Loads the machine learning model from a joblib file.
    
    predict(data: pd.DataFrame) -> pd.Series:
        Makes predictions using the loaded model on the preprocessed data.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the Prediction class and the parent PreprocessData class.

        df : pd.DataFrame
            The data for make predict
        """
        self.model = None
        self.data = df
    
    def load_model(self, mode_path: str) -> None:
        """
        Loads the machine learning model from a joblib file.

        Parameters:
        -----------
        model_path : str
            The path to the joblib file containing the saved model.
        """
        self.model = joblib.load(mode_path)

    def predict(self) -> pd.Series:
        """
        Makes predictions using the loaded model on the preprocessed data.


        Returns:
        ---------
        pd.Series
            The predcitions made by the model.
        """
        # Check if the model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model using the load_model method.")
        
        # Make predictions
        predictions = self.model.predict(self.data)
        return pd.Series(predictions)


# # Usage example
# if __name__ == "__main__":
#     # Initialize the Prediction class
#     predictor = Prediction()

#     # Load the model
#     predictor.load_model('model.joblib')

#     # Load the data to predict on
#     test_data = pd.read_csv('test_data.csv')

#     # Make predictions
#     predictions = predictor.predict(test_data)

#     # Display the predictions
#     print(predictions.head())