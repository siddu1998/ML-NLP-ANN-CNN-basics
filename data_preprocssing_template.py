import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

# creating variable for dataset
dataset=pd.read_csv('Data.csv')
#now we need to some how specify the independent and dependent variables
#in our data set which deals with country/age/salary (independet) and purchased(dependent)
#we will get all the data into a variable
X=dataset.iloc[:,:-1].values
#iloc------> i will take all the data from the csv and convert into a matrix
#:,:-1----> take all the rows(this is indicated by : and after comma tells us about the coloumn) except the last one(since that is dependent)
y=dataset.iloc[:,3].values

print('I am the independent variable vector',X)
print('I am the dependent variable vector',y)


#next lecture so now we have to deal with missing data we see we have some missing data in the csv.
#solution 1: remove the observation :-) hahah!
#solution 2: replace the missing data with the mean of that coloumns data
#so now use library to implement solution2


#this library allows preprocessing data
#Imputer is a class that helps deal with missing data
#now we need to make an object of the class

#missing data is given by NaN it might differ just check by printing the dataset
imputer=Imputer(missing_values='NaN',strategy= 'mean',axis=0)
#strategy default mean only
#axis=0 mean of coloumn
#axis=1 mean of row
#now we have to fit the imputer to the data set
imputer.fit(X[:,1:3])#remember here upper_bound is excluded though it is 1:2 we have to write 1:3
#the fit means to associate the imputer with the array
#the below step is to change the X(add missing data)
X[:,1:3] = imputer.transform(X[:,1:3])
print('This is after all the null have been kicked out',X)

#categorical variables are those colouns that can be made into categories
#we deal with equations in ML so categories needed to be encoded into some numbers

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#OneHotEncoder --> is used for dummy-variable

labelencoder_X=LabelEncoder()
#here what we are doing is...
#R.H.S the object is being told to encode the first coloumn i.e here the first category
#L.H.S assign the encoded to X
X[:,0]=labelencoder_X.fit_transform(X[:,0])
print('After Encoded',X)
#Machine learning is all about weight,so our encoding might say that spain>germany which is wrong
#so we have to use some other type of encoding
#We use the concept of dummy varibles( FORGET EVERYTHING ITS EXACTLY LIKE BIT-MAP)

#we have to pass params -->categorical_features[coloumn]
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
print('This after oneHotEncdoer a.k.a Dummy variables',X)
#OneHotEncoder props--> Number of dummy variables number of categories in the column
#if it belongs to a column then 1 is set the rest are set to 0
#Simple :-)

#now to deal with the next category which is the purchased only 2 categories then go for labelencode (Use label encoder in a yes or no case)
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
print('purchased coloumn made encoded',y)

#Data needs to be set into training set and test set
#machine learning means a machine needs to learn
#model need to learn/ Training set allows the model to learn / and then we test the performance on training set

#SPLIT!
from sklearn.cross_validation import train_test_split

#cross-validation is a library which has the train_test_split sexy method

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,train_size=0.8,random_state=0)
#train_test_split as the name says splits the data into parts test/train test_size param--> amount of data to be put as test_size out of 1 | random_state--> generally not required but we can keep the same number while coding in groups so all get same result
print('Check X_train',X_train)
print('Check X_test',X_test)
print('Check y_train',y_train)
print('Check y_test',y_test)

#so the x_train/y_train we devlop some relationship
# and then test the relationship on x_test,y_test

#A lot of machine learning models are based on eucledian models
#so all the coloumsn must be of simillar range
#so for that we need to scale factorize
# generally get all the columns -1 to 1

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#training set needs to be fit and then transformed
X_train=sc_X.fit_transform(X_train)
#test set does not need to be fit it can directly be transformed
X_test=sc_X.transform(X_test)

#Do We need to scale Dummy Variables! well they are alredy 0 or 1 :-)
# but still we can do it i mean there is nothing wrong in doing it but we have to trade the fact me might add weight to caegory which is not always favourable

print('Scaled X_test',X_test)
print('Scaled X_train',X_train)

#next we will be working with regression! What is regression.





#Regression models (both linear and non-linear) are used for predicting
#  a real value, like salary for example. If your independent variable is time,
# then you are forecasting future values,
# otherwise your model is predicting present but unknown values.


#Simple Linear Regression
#Polynomial Regression
#Multiple Linear Regression
#Support Vector for Regression (SVR)
#Decision Tree Classification
#Random Forest Classification


#--SAI SIDDARTHA MARAM --