import joblib
import numpy as np
from ingestData import IngestData
from cleaningData import CleaningData
from pr_processData import PreprocessData
from splitingData import SplitData
from tuning_Hyperprameters import TuningHyperparameters
from model import MLModel
from evaluation import Evaluation
from prediction import Prediction

import warnings 
warnings.filterwarnings('ignore')

# Step 1: Ingest Data
# NOTE: # This method should read data from CSV or SQL database
data_ingestor = IngestData()
data = data_ingestor.get_data(source="/workspaces/Salary_Predictions_of_Data_Professions/Data/Salary Prediction of Data Professions.csv",
                              source_type='csv')  

# Step 2: Clean Data
# NOTE: # Clean the data (e.g., remove duplicates, handle outliers)exclude
numberical_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(exclude=[np.number]).columns


data_cleaner = CleaningData(df=data)
cleaned_data = data_cleaner.remove_duplicates()
cleaned_data = data_cleaner.handle_outliers(column=numberical_cols, method='IQR')
cleaned_data = data_cleaner.replace_missing_values(strategy='mode')

# Step 3: Preprocess Data
# NOTE: Encoder for categorical features and scaling
preprocessor = PreprocessData(cleaned_data)
preprocessed_data = preprocessor.one_hot_encode(columns=categorical_cols)
preprocessed_data = preprocessor.scaling()

# Step 4: Split Data
# NOTE: Split data into training and test sets
splitter = SplitData()
X_train, X_test, y_train, y_test = splitter.split(preprocessed_data, target='SALARY', test_size=0.2, random_state=42)  

# Step 5: Hyperparameter Tuning
# NOTE: I have combined all the Hyperparameters tuning features of the ML Model class
hyper_tuner = TuningHyperparameters()
best_params_svr = hyper_tuner.tune_svr(X_train, y_train)
best_params_lr = hyper_tuner.tune_linear_regression(X_train, y_train)
best_params_rf = hyper_tuner.tune_random_forest_regression(X_train, y_train)

# Step 6: Train Models
model = MLModel()
# svr_model = model.train_model(X_train, y_train, algorithm='SVR')
lr_model = model.train_model(X_train, y_train, algorithm='linear_regression')
# rf_model = model.train_model(X_train, y_train, algorithm='random_forest_regression')

# Step 7: Evaluate Models
evaluator = Evaluation()

# # For SVR
# y_pred_svr = svr_model.predict(X_test)
# svr_r2 = evaluator.r2(y_test, y_pred_svr)
# svr_mae = evaluator.MAE(y_test, y_pred_svr)
# svr_mdae = evaluator.MdAE(y_test, y_pred_svr)
# svr_mse = evaluator.MSE(y_test, y_pred_svr)

# For Linear Regression
y_pred_lr = lr_model.predict(X_test)
lr_r2 = evaluator.r2(y_test, y_pred_lr)
lr_mae = evaluator.MAE(y_test, y_pred_lr)
lr_mdae = evaluator.MdAE(y_test, y_pred_lr)
lr_mse = evaluator.MSE(y_test, y_pred_lr)

# # For Random Forest
# y_pred_rf = rf_model.predict(X_test)
# rf_r2 = evaluator.r2(y_test, y_pred_rf)
# rf_mae = evaluator.MAE(y_test, y_pred_rf)
# rf_mdae = evaluator.MdAE(y_test, y_pred_rf)
# rf_mse = evaluator.MSE(y_test, y_pred_rf)

# Output evaluation metrics
# print(f"SVR Model - R2: {svr_r2}, MAE: {svr_mae}, MdAE: {svr_mdae}, MSE: {svr_mse}")
print(f"Linear Regression Model - R2: {lr_r2}, MAE: {lr_mae}, MdAE: {lr_mdae}, MSE: {lr_mse}")
# print(f"Random Forest Model - R2: {rf_r2}, MAE: {rf_mae}, MdAE: {rf_mdae}, MSE: {rf_mse}")

# Step 8: Make Predictions
# predictor = Prediction()
# predictor.load_model()  # Load the best model
# new_data = data_ingestor.get_data()  # Get new data for prediction
# predictions = predictor.predict(new_data)  # Generate predictions

# Optional: Save the final model
joblib.dump(lr_model, 'lr_model.joblib')