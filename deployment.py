from flask import Flask, render_template, request
from prediction_Deployment import PredictionDeployment  # Corrected import
import pandas as pd

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result.html', methods=['POST'])  # Corrected route definition
def result():
    # Extract form data
    data = {
        'FIRST NAME': request.form['first_name'],
        'LAST NAME': request.form['last_name'],
        'SEX': request.form['sex'],
        'DOJ': request.form['doj'],
        'CURRENT DATE': request.form['current_date'],
        'DESIGNATION': request.form['designation'],
        'AGE': request.form['age'],
        'UNIT': request.form['unit'],
        'LEAVES USED': request.form['leaves_used'],
        'LEAVES REMAINING': request.form['leaves_remaining'],
        'RATINGS': request.form['ratings'],
        'PAST EXP': request.form['past_exp']
    }

    # Convert data into DataFrame
    df = pd.DataFrame(data, index=[0])

    # Path to the saved model
    model_path = "/workspaces/Salary_Predictions_of_Data_Professions/saved_model_joblib/lr_model.joblib"

    # Initialize PredictionDeployment object
    deployment = PredictionDeployment(df=df, path_model=model_path)  # Corrected class name

    # Make predictions
    predictions = deployment.predict()

    # Print predictions
    print("Predictions:", predictions)
    
    # Render result template with data and predicted salary
    return render_template('result.html', data=data, predicted_salary=predictions)

if __name__ == '__main__':
    app.run(debug=True)
