# Salary Predictions of Data Professions

## Project Overview

This project aims to predict the salaries of data professionals using various machine learning models. The dataset includes various features related to the employees, such as age, gender, job designation, etc. The following machine learning models have been implemented and evaluated:

1. Support Vector Regression (SVR)
2. Linear Regression
3. Random Forest Regression
4. LightGBM
5. XGBoost
6. CatBoost

## Project Steps

1. **Ingest Data**: Load the dataset from a CSV file.
2. **Clean Data**: Remove duplicates and handle missing values.
3. **Preprocess Data**: Encode categorical variables and scale numerical features.
4. **Split Data**: Split the data into training and testing sets.
5. **Hyperparameter Tuning**: Tune hyperparameters for SVR, Linear Regression, and Random Forest models.
6. **Train Models**: Train models including SVR, Linear Regression, Random Forest, LightGBM, XGBoost, and CatBoost.
7. **Make Predictions**: Predict salaries using the trained models.
8. **Evaluate Models**: Evaluate the performance of the models using metrics like R2, MAE, MdAE, and MSE.

## Data

The dataset used in this project is `Salary Prediction of Data Professions.csv`. It includes the following features:

- FIRST NAME
- LAST NAME
- SEX
- DOJ (Date of Joining)
- AGE
- LEAVES USED
- RATINGS
- LEAVES REMAINING
- DESIGNATION
- UNIT
- CURRENT DATE
- SALARY (target variable)

## Model Evaluation

The following evaluation metrics were used to compare the performance of the models:

- R2 (Coefficient of Determination)
- MAE (Mean Absolute Error)
- MdAE (Median Absolute Error)
- MSE (Mean Squared Error)

### Results

| Model                  | R2          | MAE          | MdAE         | MSE          |
|------------------------|-------------|--------------|--------------|--------------|
| **SVR**                | 0.6369      | 0.0431       | 0.0360       | 0.0029       |
| **Linear Regression**  | 0.8980      | 0.0183       | 0.0126       | 0.0008       |
| **Random Forest**      | 0.9235      | 0.0140       | 0.0089       | 0.0006       |
| **LightGBM**           | 0.9143      | 0.0143       | 0.0093       | 0.0007       |
| **XGBoost**            | 0.9327      | 0.0139       | 0.0090       | 0.0005       |
| **CatBoost**           | 0.9389      | 0.0133       | 0.0087       | 0.0005       |
| **CatBoost (Tuned)**   | 0.9389      | 0.0133       | 0.0087       | 0.0005       |

## Usage

To run this project, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/alrashidissa/Salary_Predictions_of_Data_Professions.git
   cd Salary_Predictions_of_Data_Professions
   ```

2. **Install the necessary dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the project script**:
   ```sh
   python main.py
   ```

## File Structure

- `ingestData.py`: Contains the `IngestData` class for loading the dataset.
- `cleaningData.py`: Contains the `CleaningData` class for cleaning the data.
- `pr_processData.py`: Contains the `PreprocessData` class for preprocessing the data.
- `splitingData.py`: Contains the `SplitData` class for splitting the data into training and testing sets.
- `tuning_Hyperprameters.py`: Contains the `TuningHyperparameters` class for hyperparameter tuning.
- `model.py`: Contains the `MLModel` class for training and saving models.
- `evaluation.py`: Contains the `Evaluation` class for evaluating models.
- `prediction.py`: Contains the `Prediction` class for making predictions.
- `main.py`: Main script to run the project pipeline.

## Contributing

Contributions are welcome! Please create a pull request or submit an issue for any changes or suggestions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```