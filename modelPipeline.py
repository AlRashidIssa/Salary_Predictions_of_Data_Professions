
# Importing custom modules
from ingestData import IngestData
from cleaningData import CleaningData
from pr_processData import PreprocessData
from splitingData import SplitData
from tuning_Hyperprameters import TuningHyperparameters
from model import MLModel
from evaluation import Evaluation
from prediction import Prediction

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Step 1: Ingest Data
data_ingestor = IngestData()
data = data_ingestor.get_data(source="/workspaces/Salary_Predictions_of_Data_Professions/Data/Salary Prediction of Data Professions.csv",
                              source_type='csv')

# Step 2: Clean Data
cleaning_data = CleaningData(df=data)
cleaned_df = cleaning_data.remove_duplicates()
cleaned_df = cleaning_data.replace_missing_values(column='LAST NAME', strategy='mode')
cleaned_df = cleaning_data.replace_missing_values(column='DOJ', strategy='mode')
cleaned_df = cleaning_data.replace_missing_values(column='AGE', strategy='median')
cleaned_df = cleaning_data.replace_missing_values(column='LEAVES USED', strategy='mean')
cleaned_df = cleaning_data.replace_missing_values(column='RATINGS', strategy='median')
cleaned_df = cleaning_data.replace_missing_values(column='LEAVES REMAINING', strategy='median')

# Step 3: Preprocess Data
preprocessor = PreprocessData(df=cleaned_df)
preprocessor_df = preprocessor.zero_encode(columns=['FIRST NAME', 'LAST NAME'])
preprocessor_df = preprocessor.one_hot_encode(columns=['SEX', 'CURRENT DATE'])
preprocessor_df = preprocessor.label_encode(columns=['DOJ', 'DESIGNATION', 'UNIT'])
scaling = PreprocessData(df=preprocessor_df)
scaling_df = preprocessor.scaling()

# Step 4: Split Data
split = SplitData()
X_train, X_test, y_train, y_test = split.split(df=scaling_df, target='SALARY', test_size=0.2, random_state=42)

# Step 5: Hyperparameter Tuning
hyper_tuner = TuningHyperparameters()
best_params_svr = hyper_tuner.tune_svr(X_train, y_train)
best_params_lr = hyper_tuner.tune_linear_regression(X_train, y_train)
best_params_rf = hyper_tuner.tune_random_forest_regression(X_train, y_train)

# Step 6: Train Models
model = MLModel()
svr_model = model.train_model(X_train, y_train, algorithm='SVR')
lr_model = model.train_model(X_train, y_train, algorithm='linear_regression')
rf_model = model.train_model(X_train, y_train, algorithm='random_forest_regression')

# Save Models
model.save_model(svr_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lr_model.joblib")
model.save_model(lr_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/svr_model.joblib")
model.save_model(rf_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/rf_model.joblib")

# Step 7: Make Predictions
predictor_svr = Prediction(X_test)
predictor_lr = Prediction(X_test)
predictor_rf = Prediction(X_test)

predictor_svr.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/svr_model.joblib')
predictor_lr.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lr_model.joblib')
predictor_rf.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/rf_model.joblib')

predictions_svr = predictor_svr.predict()
prediction_lr = predictor_lr.predict()
predictions_rf = predictor_rf.predict()

# Step 8: Evaluate Models
evaluator = Evaluation(y_test, predictions_svr)
svr_r2 = evaluator.r2()
svr_mae = evaluator.MAE()
svr_mdae = evaluator.MdAE()
svr_mse = evaluator.MSE()

evaluator = Evaluation(y_test, prediction_lr)
lr_r2 = evaluator.r2()
lr_mae = evaluator.MAE()
lr_mdae = evaluator.MdAE()
lr_mse = evaluator.MSE()

evaluator = Evaluation(y_test, predictions_rf)
rf_r2 = evaluator.r2()
rf_mae = evaluator.MAE()
rf_mdae = evaluator.MdAE()
rf_mse = evaluator.MSE()

# Output evaluation metrics
print(f"SVR Model - R2: {svr_r2}, MAE: {svr_mae}, MdAE: {svr_mdae}, MSE: {svr_mse}")
print(f"Linear Regression Model - R2: {lr_r2}, MAE: {lr_mae}, MdAE: {lr_mdae}, MSE: {lr_mse}")
print(f"Random Forest Model - R2: {rf_r2}, MAE: {rf_mae}, MdAE: {rf_mdae}, MSE: {rf_mse}")
