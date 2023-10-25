import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mail = pd.read_csv('dataset.csv')
mail.dropna(inplace=True)

label_encoding = {'spam': 0, 'ham': 1}
mail['Category'] = mail['Category'].map(label_encoding)

X = mail['Message']
Y = mail['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

model = LogisticRegression(solver='liblinear')
model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data is:', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data is:', accuracy_on_test_data)

input_your_mail = [input("Enter the Email Text:")]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
if prediction[0] == 1:
    print('Non-Spam Mail')
else:
    print('Spam Mail')
