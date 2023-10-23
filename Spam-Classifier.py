import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('mail_data.csv')
print(df)
data=df.where((pd.notnull(df)), " ")
data.head(10)
data.info()
data.shape
data.loc [data['Category'] == 'spam','Category',]=0
data.loc[data['Category'] =='ham','Category',]=1
X=data['Message']
Y=data['Category']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(X_train)
print(X_train_features)
model=LogisticRegression()
model.fit(X_train_features,Y_train)
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy on training data is:',accuracy_on_training_data)

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print('Accuracy on test  data is:',accuracy_on_test_data)

input_your_mail=['This is the second time we tried to contact you. you have won the 25 lakh prize so claim your prize ,just call 08715678900']
input_data_features=feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_features)
print(prediction)
if (prediction[0]==1):
    print('Ham mail')
else:
    print('Spam mail')

Second_input_your_mail = ['You have been selected to stay in 1of 250 top Indian hotels for more info dial at 0092456558']
input_data_features = feature_extraction.transform(Second_input_your_mail)
prediction_second = model.predict(input_data_features)
print(prediction_second)
if prediction_second[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')