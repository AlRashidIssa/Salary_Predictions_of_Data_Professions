# Importing custom modules
from ingestData import IngestData
from cleaningData import CleaningData
from pr_processData import PreprocessData
from splitingData import SplitData
from tuning_Hyperprameters import TuningHyperparameters
from model import MLModel
from evaluation import Evaluation
from prediction import Prediction

import numpy as np
# Supperssing warnings
import warnings
warnings.filterwarnings('ignore')


# Ingest Data
data_ingestor = IngestData()
data = data_ingestor.get_data(source="/workspaces/Salary_Predictions_of_Data_Professions/Data/Salary Prediction of Data Professions.csv", 
                              source_type='csv')

data = data.drop(columns=['LAST NAME', 'FIRST NAME'])

# Step: Clean Data
cleaning_data = CleaningData(df=data)
cleaned_df = cleaning_data.remove_duplicates()
cleaned_df = cleaning_data.replace_missing_values(column='DOJ', strategy='mode')
cleaned_df = cleaning_data.replace_missing_values(column='AGE', strategy='median')
cleaned_df = cleaning_data.replace_missing_values(column='LEAVES USED', strategy='mean')
cleaned_df = cleaning_data.replace_missing_values(column='RATINGS', strategy='median')
cleaned_df = cleaning_data.replace_missing_values(column='LEAVES REMAINING', strategy='median')

# Step 3: Preprocess Data
preprocessor = PreprocessData(df=cleaned_df)
preprocessor_df = preprocessor.one_hot_encode(columns=['SEX', 'DESIGNATION', 'UNIT'])
preprocessor_df = preprocessor.calculate_jo_years('DOJ', 'CURRENT DATE')
preprocessor_df['total_exp'] = np.log2(preprocessor_df['PAST EXP'] + preprocessor_df['jo_Years'])
preprocessor_df = preprocessor.scaling()
preprocessor_df.dropna(inplace=True)

preprocessor_df = preprocessor_df.drop(columns=['DOJ', 'CURRENT DATE'])

# Step 4: Split Data
split = SplitData()
X_train, X_test, y_train, y_test = split.split(df=preprocessor_df, target='SALARY', test_size=0.2, random_state=42)

# Step 6: Train Models
model = MLModel()
svr_model = model.train_model(X_train, y_train, algorithm='svr', tune=False)
lr_model = model.train_model(X_train, y_train, algorithm='linear_regression', tune=False)
rf_model = model.train_model(X_train, y_train, algorithm='random_forest_regression', tune=False)
lgb_model = model.train_model(X_train, y_train, algorithm='lightgbm', tune=False)
cat_model = model.train_model(X_train, y_train, algorithm='catboost', tune=False)
xgb_model = model.train_model(X_train, y_train, algorithm='xgboost', tune=False)

# Save Models
model.save_model(svr_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/svr_model.joblib")
model.save_model(lr_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lr_model.joblib")
model.save_model(rf_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/rf_model.joblib")
model.save_model(lgb_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lgb_model.joblib")
model.save_model(cat_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib")
model.save_model(xgb_model, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/xgb_model.joblib")

# Step 7: Make Predictions
predictor_svr = Prediction(X_test)
predictor_lr = Prediction(X_test)
predictor_rf = Prediction(X_test)
predictor_lgb = Prediction(X_test)
predictor_xgb = Prediction(X_test)
predictor_cat = Prediction(X_test)

predictor_svr.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/svr_model.joblib')
predictor_lr.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lr_model.joblib')
predictor_rf.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/rf_model.joblib')
predictor_lgb.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lgb_model.joblib')
predictor_xgb.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/xgb_model.joblib')
predictor_cat.load_model('/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib')

predictions_svr = predictor_svr.predict()
prediction_lr = predictor_lr.predict()
predictions_rf = predictor_rf.predict()
predictions_lgb = predictor_lgb.predict()
predictions_xgb = predictor_xgb.predict()
predictions_cat = predictor_cat.predict()

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

evaluator = Evaluation(y_test, predictions_lgb)
lgb_r2 = evaluator.r2()
lgb_mae = evaluator.MAE()
lgb_mdae = evaluator.MdAE()
lgb_mse = evaluator.MSE()

evaluator = Evaluation(y_test, predictions_xgb)
xgb_r2 = evaluator.r2()
xgb_mae = evaluator.MAE()
xgb_mdae = evaluator.MdAE()
xgb_mse = evaluator.MSE()

evaluator = Evaluation(y_test, predictions_cat)
cat_r2 = evaluator.r2()
cat_mae = evaluator.MAE()
cat_mdae = evaluator.MdAE()
cat_mse = evaluator.MSE()

# Output evaluation metrics
print(f"SVR Model - R2: {svr_r2}, MAE: {svr_mae}, MdAE: {svr_mdae}, MSE: {svr_mse}")
print(f"Linear Regression Model - R2: {lr_r2}, MAE: {lr_mae}, MdAE: {lr_mdae}, MSE: {lr_mse}")
print(f"Random Forest Model - R2: {rf_r2}, MAE: {rf_mae}, MdAE: {rf_mdae}, MSE: {rf_mse}")
print(f"LightGBM Model - R2: {lgb_r2}, MAE: {lgb_mae}, MdAE: {lgb_mdae}, MSE: {lgb_mse}")
print(f"XGBoost Model - R2: {xgb_r2}, MAE: {xgb_mae}, MdAE: {xgb_mdae}, MSE: {xgb_mse}")
print(f"CatBoost Model - R2: {cat_r2}, MAE: {cat_mae}, MdAE: {cat_mdae}, MSE: {cat_mse}")


# Tuning CatBoost
cat_model_tuning = model.train_model(X_train, y_train, algorithm='catboost', tune=True)
model.save_model(cat_model_tuning, "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib")

cat_model_tuning = Prediction(X_test)
cat_model_tuning.load_model("/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib")

evaluator = Evaluation(y_test, predictions_cat)
catT_r2 = evaluator.r2()
catT_mae = evaluator.MAE()
catT_mdae = evaluator.MdAE()
catT_mse = evaluator.MSE()

print(f"CatBoost Model  Tuning- R2: {catT_r2}, MAE: {catT_mae}, MdAE: {catT_mdae}, MSE: {catT_mse}")

