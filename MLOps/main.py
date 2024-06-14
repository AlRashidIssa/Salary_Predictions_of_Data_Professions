import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from salary_prediction_Model import SalaryPredictionModel
from mlflow_utils import create_mlflow_experiment

def log_params_safely(run, params):
    """Log parameters to MLflow, ensuring no duplicates with different values."""
    existing_params = run.data.params
    for key, value in params.items():
        if key in existing_params and str(existing_params[key]) != str(value):
            print(f"Skipping logging param {key}: {value} as it conflicts with already logged value {existing_params[key]}")
        else:
            mlflow.log_param(key, value)

# Load data for visualizations
dataset_source_url = "Data/Salary Prediction of Data Professions.csv"
data = pd.read_csv(dataset_source_url)
# Create an instance of a PandasDataset
dataset = mlflow.data.from_pandas( # type: ignore
    data, source=dataset_source_url, name="Salary Prediction of Data Professions", targets="SALARY"
)

describt = data.describe()
describt.to_csv("mlflow_artifacts/summary_statistics.csv")  # Save as CSV file
# Categorical description
cat_describt = data.describe(include=['object'])
cat_describt.to_csv("mlflow_artifacts/categorical_description.csv")  # Save as CSV file
# Visualize Distribution of Salaries
fig_hist = plt.figure(figsize=(10, 6))
sns.histplot(data['SALARY'], kde=True) # type: ignore
# Visualize the count of categorical variables
des_plot = plt.figure(figsize=(10, 6))
sns.countplot(y=data['DESIGNATION'])
# Correlation heatmap
numerical_data = data.select_dtypes(include=[np.number]).columns
hot_fig = plt.figure(figsize=(12, 8))
correlation_matrix = data[numerical_data].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# Boxplot: Salary vs. Ratings
box_salary_rating = plt.figure(figsize=(10, 6))
sns.boxplot(x=data['RATINGS'], y=data['SALARY'])
# Scatterplot: Salary vs. Past Experience
scatter_salary_past_exp = plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['PAST EXP'], y=data['SALARY'])
# Scatterplot: Salary vs. Age
scatter_salary_age = plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['AGE'], y=data['SALARY'])

model = SalaryPredictionModel(data_source='Data/Salary Prediction of Data Professions.csv')
model.run_pipeline()
results = model.get_results()
models = results['models']



def main():
    experiment_id = create_mlflow_experiment(
        experiment_name= "Machine-leanrin-Opearation-Prediction-Salary-Prediction",
        artifact_location= "mlflow_artifacts",
        tags={"purpose":"learning"}
    )
    
    with mlflow.start_run(run_name="model_prediction") as run:
        print("RUN ID model_prediction:", run.info.run_id)

        try:

            mlflow.log_input(dataset, context="Data")

            
            mlflow.log_artifact("mlflow_artifacts/summary_statistics.csv")  # Log as artifact
            print("Summary statistics logged.")

            mlflow.log_artifact("mlflow_artifacts/categorical_description.csv")  # Log as artifact
            print("Categorical description logged.")


            mlflow.log_figure(fig_hist, "metrics/DistributionSalaries.png")
            print("Distribution of Salaries visualized and logged.")


            mlflow.log_figure(des_plot, "metrics/DESIGNATION_unique.png")
            print("Count of categorical variables visualized and logged.")


            mlflow.log_figure(hot_fig, "metrics/CorrelationMatrix.png")
            print("Correlation heatmap visualized and logged.")


            mlflow.log_figure(box_salary_rating, "metrics/box_salary_rating.png")
            print("Boxplot of Salary vs. Ratings visualized and logged.")


            mlflow.log_figure(scatter_salary_past_exp, "metrics/scatter_salary_past_exp.png")
            print("Scatterplot of Salary vs. Past Experience visualized and logged.")


            mlflow.log_figure(scatter_salary_age, "metrics/scatter_salary_age.png")
            print("Scatterplot of Salary vs. Age visualized and logged.")

            for model_name, model_obj in models.items():
                with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                    print(f"RUN ID {model_name}:", model_run.info.run_id)
                    hyperparameters = results['hyperparameters'][model_name]
                    metrics = results['metrics'][model_name]
                    log_params_safely(mlflow.active_run(), hyperparameters)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model_obj, f"model_{model_name}")
                    print(f"Model {model_name} training completed and logged.")

        except Exception as e:
            print("An error occurred:", str(e))
if __name__ == "__main__":
    main()
