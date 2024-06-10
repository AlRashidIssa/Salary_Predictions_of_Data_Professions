from flask import Flask, render_template, request
from prediction_Deployment import PredictionDeployment  # Corrected import
import pandas as pd

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result.html', methods=['POST'])
def result():
    # Extract form data
    data = {
        'FIRST NAME': str(request.form['first_name']),
        'LAST NAME': str(request.form['last_name']),
        'SEX': str(request.form['sex']),
        'DOJ': str(request.form['doj']),
        'CURRENT DATE': str(request.form['current_date']),
        'DESIGNATION': str(request.form['designation']),
        'AGE': int(request.form['age']),
        'UNIT': str(request.form['unit']),
        'LEAVES USED': float(request.form['leaves_used']),
        'LEAVES REMAINING': float(request.form['leaves_remaining']),
        'RATINGS': float(request.form['ratings']),
        'PAST EXP': float(request.form['past_exp'])
    }

    # Create a DataFrame from the form data
    data_df = pd.DataFrame(data, index=[0])

    def unscale_salary(scaled_value, min_salary=40001.0, max_salary=388112.0):
        return scaled_value * (max_salary - min_salary) + min_salary

    deployment = PredictionDeployment(df=data_df, 
                                      path_model="/home/alrashidissa/Desktop/Internship/Salary_Predictions_of_Data_Professions/saved_model_joblib/cat_model.joblib")
    scalin_salary = deployment.predict()
    new_salary = int(unscale_salary(scaled_value=scalin_salary) / 100)


    # Render result template with data and predicted salary
    return render_template('result.html', data=data, predicted_salary=new_salary)

if __name__ == '__main__':
    app.run(debug=True)
