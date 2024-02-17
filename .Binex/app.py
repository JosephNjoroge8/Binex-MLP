from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load label encoders
label_encoder_x = LabelEncoder()
label_encoder_y = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/form')
def form():
    return render_template('userdataform.html')

@app.route('/predict', methods=['POST', 'GET'])
# Define route for form submission
def predict():
    # Load the trained model
    from main import classifier

    # Get form data
    age = int(request.form['age'])
    gender = request.form['gender']
    marital_status = request.form['marital_status']
    dependent = request.form['dependent']
    income = int(request.form['income'])
    education = request.form['education']
    employment = request.form['employment']
    loan_amount = int(request.form['loan_amount'])
    
    # Encode categorical features
    gender_encoded = label_encoder_x.fit_transform([gender])[0]
    marital_status_encoded = label_encoder_x.fit_transform([marital_status])[0]
    dependent_encoded = label_encoder_x.fit_transform([dependent])[0]
    education_encoded = label_encoder_x.fit_transform([education])[0]
    employment_encoded = label_encoder_x.fit_transform([employment])[0]
    
    # Prepare input data for prediction
    user_data = np.array([[age, gender_encoded, marital_status_encoded, dependent_encoded, income, education_encoded, employment_encoded, loan_amount]])
    
    # Make prediction
    prediction = classifier.predict(user_data)
    
    # Decode prediction
    prediction_result = label_encoder_y.inverse_transform(prediction)[0]
    
    # Render prediction result
    return render_template('result.html', prediction_result=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
