#Multiple linear regressions
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values




#here one of the dependent variable is of the form text here CITY name but maths cant deal with strings so we have to encode them into dummy vriables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#remeber u alwayas need to convert into labelencoder before onehotencoder Espectially while using text shit
labelencode_X=LabelEncoder()
X[:,3]=labelencode_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#Avoiding the dummy variable trap -->Remember we discussed not carry the dummy variable trap
X=X[:,1:]




# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



#Above we are done till Data preprocessing phase from below the shit starts

#fitting Multiple Linear regression to the trainign set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#fit the regressor into our training set
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)
print(y_pred)
print(y_test)