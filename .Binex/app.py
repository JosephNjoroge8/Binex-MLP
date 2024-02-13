from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Load the loan eligibility dataset
    data = pd.read_csv('loan_dataset.csv')

    # Preprocess the data
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    # Train the random forest classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Get user input from the HTML form
    gender = request.form.get('gender')
    married = request.form.get('married')
    dependents = request.form.get('dependents')
    education = request.form.get('education')
    self_employed = request.form.get('self_employed')
    applicant_income = float(request.form.get('applicant_income'))
    coapplicant_income = float(request.form.get('coapplicant_income'))
    loan_amount = float(request.form.get('loan_amount'))
    loan_term = float(request.form.get('loan_term'))
    credit_history = float(request.form.get('credit_history'))
    property_area = request.form.get('property_area')

    # Create a new dataframe with the user input
    user_input = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Make predictions on the user input
    prediction = clf.predict(user_input)

    # Return the prediction result to the HTML page
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
