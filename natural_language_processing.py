import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#quoting=3 --> ignore " "

#now we have to cleann the texts

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus=[]

for i in range(0,1000):
    #re has tools to clean shit

    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #now we need only the root of the word
    #this is called steaming ps.steam(word)
    # now we need to get back the string
    review=' '.join(review)
    corpus.append(review)


#Bag - of - words - model
# we will take all the unique words and create one coloumn for each word
# and rows will be the reviews
# a cell holds the number of times the word has occured in the review

# this is a classification model quetison
# now its time to create our sparse matrixx (a matrix with a lot of zeros)



from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)

#max_features-> only 1500 words will make our sparse matrices smaller
#CountVecotorizer --> is used to create the sparse matrix
X=cv.fit_transform(corpus).toarray()
#dependent
y=dataset.iloc[:,1].values






# Generally IN NLP WE USE NAIEVE BYES -->We have our naive bayes classifier

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20,random_state=0)



#fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)


#Prediciting the result

y_pred=classifier.predict(X_test)


#Making the confusion Matrix this will give the comparrision between the true and test
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




rv = input("Please enter review")
review = re.sub('[^a-zA-Z]', ' ', rv)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
print(review)
corpus.append(review)
X_new = cv.transform(corpus).toarray()
review = X_new[-1]
predic = classifier.predict([review])
print(predic)



# I think that choice of classification method depends on our goal.
#
# If we want to catch as many positive reviews as possible, I'd go for Random Forest.
#
# If we want to focus on indicating which of given reviews were positive, I'd go for Naive Bayes.
#
# If we need to be as accurate as possible about recognising both positive and negative reviews, I'd go for LogisticRegression.
#
# If all of these goals are equally important, I'd pick Kernel SVM, as it's the most balanced one.
#





