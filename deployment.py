from flask import Flask, render_template, request, redirect, url_for
from ingestData import IngestData
from pr_processData import PreprocessData
from prediction import Prediction
from database import Database
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Extract form data
    data = {
        'first_name': request.form['first_name'],
        'last_name': request.form['last_name'],
        'sex': request.form['sex'],
        'doj': request.form['doj'],
        'current_date': request.form['current_date'],
        'designation': request.form['designation'],
        'age': request.form['age'],
        'unit': request.form['unit'],
        'leaves_used': request.form['leaves_used'],
        'leaves_remaining': request.form['leaves_remaining'],
        'ratings': request.form['ratings'],
        'past_exp': request.form['past_exp']
    }

    dataset = Database(db_name="database.db")
    dataset.insert_data(data=data)

    data = pd.DataFrame(data=data)

    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    # Fetch and preprocess data
    preprocess = PreprocessData(data)
    data = preprocess.one_hot_encode(columns= categorical_cols)
    data = preprocess.scaling()

    # Make prediction
    predicted = Prediction(data)
    predicted.load_model(mode_path="")
    predicted_salary = predicted.predict(data=data)

    return render_template('result.html', data=data, predicted_salary=predicted_salary)

if __name__ == '__main__':
    app.run(debug=True)
