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


#in neural networks it is good to have future scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



#Making the ANN (artificial Nerual network)

#import the keras libraris and packages
#sequential -->to initilize the neural network
#the dense module is used to create layers

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initilizing the ANN
classifier=Sequential()

#adding the first layer and the hidden first layer
#we have 11 individual input variables
#in this case duing hidden layers we will be using retifier function and output we will be using segmoid funtion

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#now we will add hidden layers here we will add one hidden layer only
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#part 3 making the output layer and getting output
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Running the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ann to training set

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#Makin the predictions


y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)#-->if y_pred >0.5 true
print(y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)