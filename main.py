from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__) #create Object Of Flask
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Ensure all columns are in the same order as used during model training
columns = ['name', 'company', 'year', 'kms_driven', 'fuel_type']


@app.route('/', methods=['GET', 'POST']) #Entery Point of Our Application
def index(): # index Function is call When Route or Url was hit
    companies = sorted(car['company'].unique()) #select company
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Retrieve data from the form
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Check for empty fields
    if not all([company, car_model, year, fuel_type, driven]):
        return "Error: Please fill out all the fields."

    # Convert to proper data types
    try:
        year = int(year)
        driven = int(driven)
    except ValueError:
        return "Error: 'Year' and 'Kilometers driven' must be numeric."

    # Prepare data for prediction
    data = pd.DataFrame(data=[[car_model, company, year, driven, fuel_type]], columns=columns)

    # Make prediction
    prediction = model.predict(data)
    return str(np.round(prediction[0], 2))




if __name__ == '__main__':
    app.run(debug=True)
