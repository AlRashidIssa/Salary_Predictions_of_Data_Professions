import sys
sys.path.append('/home/alrashidissa/Desktop/Internship/Salary_Predictions_of_Data_Professions/')


from ingestData import IngestData
from cleaningData import CleaningData
from pr_processData import PreprocessData
from splitingData import SplitData
from model import MLModel
from evaluation import Evaluation
from prediction import Prediction
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictionModel:
    """
    Salary Prediction Model for data professionals.
    This class handles the complete workflow from data ingestion,
    cleaning, preprocessing, training, tuning, and evaluation of different models.
    """

    def __init__(self, data_source) -> None:
        """
        Initialize the SalaryPredictionModel class.

        Parameters:
        data_source (str): Path to the data source (CSV file).
        """
        self.data_source = data_source
        self.models = {}
        self.metrics = {}
        self.hyperparameters_dict = {}
        self.data = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Load and preprocess the data.
        """
        ingest_data = IngestData()
        data = ingest_data.get_data(self.data_source, source_type='csv')

        data = data.drop(columns=['LAST NAME', 'FIRST NAME'])

        # Step: Clean Data
        cleaning_data = CleaningData(df=data)
        cleaned_df = cleaning_data.remove_duplicates()
        cleaned_df = cleaning_data.replace_missing_values(column='DOJ', strategy='mode')
        cleaned_df = cleaning_data.replace_missing_values(column='AGE', strategy='median')
        cleaned_df = cleaning_data.replace_missing_values(column='LEAVES USED', strategy='mean')
        cleaned_df = cleaning_data.replace_missing_values(column='RATINGS', strategy='median')
        cleaned_df = cleaning_data.replace_missing_values(column='LEAVES REMAINING', strategy='median')

        # Step: Preprocess Data
        preprocessor = PreprocessData(df=cleaned_df)
        preprocessor_df = preprocessor.one_hot_encode(columns=['SEX', 'DESIGNATION', 'UNIT'])
        preprocessor_df = preprocessor.calculate_jo_years('DOJ', 'CURRENT DATE')
        preprocessor_df['total_exp'] = np.log2(preprocessor_df['PAST EXP'] + preprocessor_df['jo_Years'])
        preprocessor_df = preprocessor.scaling()
        preprocessor_df.dropna(inplace=True)
        preprocessor_df = preprocessor_df.drop(columns=['DOJ', 'CURRENT DATE'])

        # Step: Split Data
        split = SplitData()
        self.X_train, self.X_test, self.y_train, self.y_test = split.split(df=preprocessor_df, target='SALARY', test_size=0.2, random_state=42)


    def run_models(self):
        """
        Train and save different models.
        """
        model = MLModel()
        self.models['svr'] = model.train_model(self.X_train, self.y_train, algorithm='svr', tune=False)
        self.models['linear_regression'] = model.train_model(self.X_train, self.y_train, algorithm='linear_regression', tune=False)
        self.models['random_forest'] = model.train_model(self.X_train, self.y_train, algorithm='random_forest_regression', tune=False)
        self.models['lightgbm'] = model.train_model(self.X_train, self.y_train, algorithm='lightgbm', tune=False)
        self.models['catboost'] = model.train_model(self.X_train, self.y_train, algorithm='catboost', tune=False)
        self.models['xgboost'] = model.train_model(self.X_train, self.y_train, algorithm='xgboost', tune=False)

        # Save Models
        for model_name, __ in self.models.items():
            model.save_model(self.models[model_name], f"traind_models/{model_name}_model.joblib")
    
    def hyperparameters(self):
        """
        Tune hyperparameters for different models.
        """
        for model_name, model in self.models.items():
            self.hyperparameters_dict[model_name] = model.get_params()

    def evaluate_models(self):
        """
        Evaluate the trained models and store the metrics.
        """
        for model_name, model in self.models.items():
            predictor = Prediction(self.X_test) # type: ignore
            predictor.load_model(f"traind_models/{model_name}_model.joblib")
            predictions = predictor.predict()

            evaluator = Evaluation(self.y_test, predictions) # type: ignore
            self.metrics[model_name] = {
                'r2': evaluator.r2(),
                'mae': evaluator.MAE(),
                'mdae': evaluator.MdAE(),
                'mse': evaluator.MSE()
            }

    def run_pipeline(self):
        """
        Execute the complete pipeline: loading data, tuning, training, and evaluation.
        """
        self.load_data()
        self.run_models()
        self.evaluate_models()
        self.hyperparameters()


    def get_results(self):
        """
        Get the models, metrics, hyperparameters, and data.

        Returns:
        dict: A dictionary containing models, metrics, hyperparameters, and data.
        """
        return {
            'models': self.models,
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters_dict,
            'data': {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test
            }
        }

if __name__ == "__main__":

    model = SalaryPredictionModel(data_source='/home/alrashidissa/Desktop/Internship/Salary_Predictions_of_Data_Professions/Data/Salary Prediction of Data Professions.csv')
    model.run_pipeline()
    results = model.get_results()
    print(results['metrics'])
