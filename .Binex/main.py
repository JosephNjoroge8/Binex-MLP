import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib as plt

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

data = pd.read_csv('Training Data.csv')
data.head()
data.info()
data.shape

x=data.iloc[:,np.r_[1:5,7:10,12:13]].values
y=data.iloc[:,12].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()

from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()

for i in range(0, 7):
    x_train[:,i]= label_encoder_x.fit_transform(x_train[:,i])

y_train= label_encoder_y.fit_transform(y_train)

for i in range(0, 7):
    x_test[:,i]= label_encoder_x.fit_transform(x_test[:,i])

label_encoder_y = LabelEncoder()
y_test= label_encoder_y.fit_transform(y_test)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

user_data = np.array([[25, 'Male', 'Single', 'Yes', 50000, 'Graduate', 'No', 1]])
user_data_encoded = np.zeros(user_data.shape)

for i in range(0, 7):
    user_data_encoded[0, i] = label_encoder_x.fit_transform([user_data[0, i]])[0]

prediction = classifier.predict(user_data_encoded)

if prediction[0] == 1:
    print("Congratulations! You are eligible for a loan.")
else:
    print("Sorry, you are not eligible for a loan.")

probability_of_approval = classifier.predict_proba(user_data_encoded)[:, 1]
print(f"Probability of loan approval: {probability_of_approval[0]:.2%}")
