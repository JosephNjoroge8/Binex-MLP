from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/form')
def form():
    return render_template('userdataform.html')

#@app.route('/predict', methods=['POST', 'GET'])

if __name__ == '__main__':
    app.run(debug=True)
