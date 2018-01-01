#Polynomial Regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#since this data set is small and we want utmost precision what we do is we will not divide the data and use the whole data
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to the data set
from sklearn.linear_model import  LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#fitting Polynomial Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
#Now we have to modify the independent set to contain their powers
poly_reg=PolynomialFeatures(degree=4)
#transform the X to X_poly
#this poly_reg which we have just created also adds that default first coloumn of 1's a.k.a b0*x0
X_poly=poly_reg.fit_transform(X)

#now we have to fit this shit into our model

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
#Note here the lin_reg2 is still an object of Linear Regression and not Polynomial regression
#the X_Poly is contains only the data set values manipulated but we will need to predict new values

#visualiziation of linear and polynomail

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff using Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
#play and this will show u how badly ur Screwed :-)
plt.show()



#So we want to plot X--> expereince vs predicted salary (y-axis) this can be a new set of data as well

#incase we want to make curve more smooth we have to introduce more points
#we use np.aranfe--> min vale, max vale, gap
#X_grid will cointain points between max and min separated by 0.1
#now we reshape the lenght will be all the points and 1 coloumn
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
#we can pass lin_reg2.predict(X_poly) also but i just generalized it
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff using Polynomial Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting with linear regression model
print('Please enter the expereience')
exp=float(input())
print('The expeced salaray using linear regression')
print(lin_reg.predict(exp))
print(y)
print('the expected salary using polynomial regression')
print(lin_reg2.predict(poly_reg.fit_transform(6.5)))

