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
#Now there is some issue with this
# we are considering all the variables
# the problem with this there are some statistic variables which contribute great impact
# and maybe some shit which do not contribute any impact SO LET US GO FOR BACKWARD ELIMINATION yoyo! Baby

import statsmodels.formula.api as sm
#remember our multiple linear regression shitty equation it has b0
#we can make b0-->b0*x0 where x0 =1
# so to get the value of x0 let us add a column in the dataset with all 1's
# some libraies like the linear regresion wala alredy has it
# byt this shiity library called statsmodes doesnt know it
#so we have to do it on our own
#the trick here is to an array of all ones just add the X the reverse is liitle difficult
X=np.append(arr=np.ones((50, 1)).astype(int),values=X,axis=1)#ones function in numpy adds all ones array of specified size
#Now the party begins baby! Backward elmination starts after we have preprocessed the data by adding that 1 coloumn with ones
#so first team of features which cause most impact and throw the unneseacty shit


# step 1----> Select significane level to stay in the model (SL=0.05 example)
#default-->0.05
# step 2 -----> Fit the full  model with all possible predictors

X_opt=X[:,[0,1,2,3,4,5]]
#endog--> the independent variable vector
#exog --> The dependent variable vector (dependent means the optimal wala)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()


# step 3 | Step 4 |------> consider the predictor with the highest P-value. if P>SL, go to STEP 4. otherwise YOUR MODEL IS READY
#print(regressor_OLS.summary())
#Once u print the regressor_OLS.summary()--> we get certain P-values-->the lesser the P value more the significance or impact or i.e that idependent variable is more useful to predict
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
# x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
# x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
# x3             0.8060      0.046     17.369      0.000       0.712       0.900
# x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
# x5             0.0270      0.017      1.574      0.123      -0.008       0.062

#Now look beta, this model assumes SL=0.5(default and recoomender) now x2===0.953-->this is bull shit way above 0.05 so exclude


X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#print(regressor_OLS.summary()) -->further removal

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#print(regressor_OLS.summary())--->Again we have params-->useless

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#print(regressor_OLS.summary())
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#print(regressor_OLS.summary())
#YIPEEEE! we are done now we have found all the determinents


# step 5------> Fit model with this variable (recurse with step 3)

X_train ,X_test ,y_train, y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)
regressor.fit(X_train,y_train)
y_pred_with_multi=regressor.predict(X_test)

#Actual dependent values
print('Actual y values from the dataset')
print(y_test)
#dependent predicted using multi-value regression
print('y values predicted using multiple linear regression model')
print(y_pred)
#dependent predicted using backward elimination
print('y values predicted using backward eliminatio models')
print(y_pred_with_multi)
