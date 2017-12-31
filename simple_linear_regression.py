#Simple Linear Regression--> Notes in Notes if any Make sure you check
#step 1--> Pre-Process the data!
#Oh! yea! baby we have this template i made it! so ctrl+c and ctrl+v


# Data Preprocessing Template
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

#salary is dependent variable
#year_of_expereience number of years is independent
#so,we need to relate these two

#features-->independent is only the first coloumn so remove the second here the last coloumn
X = dataset.iloc[:, :-1].values
#dependent variable is in 2nd coloumn
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# print('Check X_train',X_train)
# print('Check X_test',X_test)
# print('Check y_train',y_train)
# print('Check y_test',y_test)


# Feature Scaling
#YOYO! Simple linear regression libraries take care of Feature scaling so! for now TaTa!
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#now we have to fit the simple linear regression model to the training set
#the model discussed previosly so no worries (in notes)

from sklearn.linear_model import LinearRegression
#regressor is a machine
regressor=LinearRegression()
#now we need to train this regressor machine
#we need to fit the training data to regressor
#with this the model will learn shit!
regressor.fit(X_train,y_train)

#machine made ! Machine Trained ! now what are u waiting for GET GOING START PREDICTING

#y_pred--> predicted dependent variables
y_pred = regressor.predict(X_test)
#you can also input some value and then get expected salary prediction
y_input_pred=regressor.predict(int(input()))
print(y_pred)
print(y_test)
print(y_input_pred)

#











