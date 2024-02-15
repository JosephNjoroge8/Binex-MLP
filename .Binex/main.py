import pandas as pd  # Importing pandas library for data manipulation and analysis
import numpy as np  # Importing numpy library for numerical computations
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from scikit-learn for classification
from sklearn.model_selection import train_test_split  # Importing train_test_split function for splitting data into training and testing sets
from sklearn.metrics import accuracy_score  # Importing accuracy_score function for evaluating model performance
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for encoding categorical variables
from sklearn.preprocessing import OrdinalEncoder  # Importing OrdinalEncoder for encoding ordinal categorical variables
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot for data visualization

# Creating an instance of OrdinalEncoder with handle_unknown='use_encoded_value', unknown_value=np.nan
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

# Reading the CSV file 'Training Data.csv' into a pandas DataFrame named 'data'
data = pd.read_csv('C:\\Users\\PC\\Binex_MLP\\.Binex\\Training Data.csv')

# Displaying the first few rows of the DataFrame 'data'
data.head()

# Displaying the information about the DataFrame 'data' including data types and null values
data.info()

# Displaying the shape of the DataFrame 'data' (number of rows and columns)
data.shape

# Selecting features (columns) for the input (x) and target (y) variables
x = data.iloc[:, np.r_[1:5, 7:10, 12:13]].values
y = data.iloc[:, 12].values

# Splitting the data into training and testing sets with 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating an instance of RandomForestClassifier
classifier = RandomForestClassifier()

# Creating an instance of LabelEncoder for encoding categorical features in x_train
label_encoder_x = LabelEncoder()

# Encoding categorical features in x_train using a loop
for i in range(0, 7):
    x_train[:, i] = label_encoder_x.fit_transform(x_train[:, i])

# Creating an instance of LabelEncoder for encoding the target variable y_train
label_encoder_y = LabelEncoder()
y_train = label_encoder_y.fit_transform(y_train)

# Encoding categorical features in x_test using a loop
for i in range(0, 7):
    x_test[:, i] = label_encoder_x.fit_transform(x_test[:, i])

# Encoding the target variable y_test
y_test = label_encoder_y.fit_transform(y_test)

# Fitting the classifier to the training data
classifier.fit(x_train, y_train)

# Making predictions on the testing data
y_pred = classifier.predict(x_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Printing the accuracy of the model
print("Accuracy:", accuracy)

# Creating a numpy array 'user_data' representing user input for prediction
user_data = np.array([[25, 'Male', 'Single', 'Yes', 50000, 'Graduate', 'No', 1]])

# Creating an array 'user_data_encoded' to store encoded user data
user_data_encoded = np.zeros(user_data.shape)

# Encoding categorical features in user_data using a loop
for i in range(0, 7):
    user_data_encoded[0, i] = label_encoder_x.fit_transform([user_data[0, i]])[0]

# Making prediction on user_data_encoded
prediction = classifier.predict(user_data_encoded)

# Displaying prediction result
if prediction[0] == 1:
    print("Congratulations! You are eligible for a loan.")
else:
    print("Sorry, you are not eligible for a loan.")

# Calculating and printing the probability of loan approval
probability_of_approval = classifier.predict_proba(user_data_encoded)[:, 1]
print(f"Probability of loan approval: {probability_of_approval[0]:.2%}")
