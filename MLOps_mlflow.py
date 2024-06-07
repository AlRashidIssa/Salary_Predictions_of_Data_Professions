from ingestData import IngestData
from cleaningData import CleaningData
from pr_processData import PreprocessData
from splitingData import SplitData
from tuning_Hyperprameters import TuningHyperparameters
from model import MLModel
from evaluation import Evaluation
from prediction import Prediction

import mlflow 
import pandas as pd
import joblib

# Define the main function to encapsulate the entire pipeline
def main(filepath):
    mlflow.start_run()

    # Step 1: Ingest Data
    data_ingestor = IngestData(filepath)
    data = data_ingestor.get_data()  # This method should read data from CSV or SQL database

    # Step 2: Clean Data
    data_cleaner = CleaningData()
    cleaned_data = data_cleaner.clean_data(data)  # Clean the data (e.g., remove duplicates, handle outliers)

    # Step 3: Preprocess Data
    preprocessor = PreprocessData()
    preprocessed_data = preprocessor.fit_transform(cleaned_data)  # Handle missing values, scale, and encode features

    # Step 4: Split Data
    splitter = SplitData()
    X_train, X_test, y_train, y_test = splitter.split(preprocessed_data.drop('SALARY', axis=1), preprocessed_data['SALARY'])  # Split data into training and test sets

    # Step 5: Hyperparameter Tuning
    hyper_tuner = TuningHyperparameters()
    best_params_svr = hyper_tuner.tune_svr(X_train, y_train)
    best_params_lr = hyper_tuner.tune_linear_regression(X_train, y_train)
    best_params_rf = hyper_tuner.tune_random_forest(X_train, y_train)

    # Log hyperparameters
    mlflow.log_params(best_params_svr)
    mlflow.log_params(best_params_lr)
    mlflow.log_params(best_params_rf)

    # Step 6: Train Models
    model = MLModel()
    svr_model = model.train_svr(X_train, y_train, best_params_svr)
    lr_model = model.train_linear_regression(X_train, y_train, best_params_lr)
    rf_model = model.train_random_forest(X_train, y_train, best_params_rf)

    # Step 7: Evaluate Models
    evaluator = Evaluation()

    # For SVR
    y_pred_svr = svr_model.predict(X_test)
    svr_r2 = evaluator.r2(y_test, y_pred_svr)
    svr_mae = evaluator.MAE(y_test, y_pred_svr)
    svr_mdae = evaluator.MdAE(y_test, y_pred_svr)
    svr_mse = evaluator.MSE(y_test, y_pred_svr)

    # Log metrics for SVR
    mlflow.log_metric("svr_r2", svr_r2)
    mlflow.log_metric("svr_mae", svr_mae)
    mlflow.log_metric("svr_mdae", svr_mdae)
    mlflow.log_metric("svr_mse", svr_mse)

    # For Linear Regression
    y_pred_lr = lr_model.predict(X_test)
    lr_r2 = evaluator.r2(y_test, y_pred_lr)
    lr_mae = evaluator.MAE(y_test, y_pred_lr)
    lr_mdae = evaluator.MdAE(y_test, y_pred_lr)
    lr_mse = evaluator.MSE(y_test, y_pred_lr)

    # Log metrics for Linear Regression
    mlflow.log_metric("lr_r2", lr_r2)
    mlflow.log_metric("lr_mae", lr_mae)
    mlflow.log_metric("lr_mdae", lr_mdae)
    mlflow.log_metric("lr_mse", lr_mse)

    # For Random Forest
    y_pred_rf = rf_model.predict(X_test)
    rf_r2 = evaluator.r2(y_test, y_pred_rf)
    rf_mae = evaluator.MAE(y_test, y_pred_rf)
    rf_mdae = evaluator.MdAE(y_test, y_pred_rf)
    rf_mse = evaluator.MSE(y_test, y_pred_rf)

    # Log metrics for Random Forest
    mlflow.log_metric("rf_r2", rf_r2)
    mlflow.log_metric("rf_mae", rf_mae)
    mlflow.log_metric("rf_mdae", rf_mdae)
    mlflow.log_metric("rf_mse", rf_mse)

    # Save the best model
    best_model = rf_model  
    mlflow.sklearn.log_model(best_model, "model")

    mlflow.end_run()

if __name__ == "__main__":
    # Example usage
    main(filepath="path/to/your/data.csv")