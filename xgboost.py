#XGboost

#XGboost is the most powerful tool used in machine learning it gives it super cool performance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#here we have two categorical variables
labelencoder_X_1=LabelEncoder()
labelencoder_X_2=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
#remeber we need to remove one dummy variable to avaoid dummy variable trap
X=X[:,1:]

#splitllting data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,random_state=0)

#Fitting XGBoost to the training set
from xgboost import XGBClassifier()
classifier=XGBClassifier()
classifier.fit(X_train,y_train)



#Predicitions
y_pred=classifier.predict(X_test)
print(y_pred)
#Confusion
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#k-cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())




















