import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reading the data into a dataframe and getting the shape of the data 
#read the data 
d =pd.read_csv('news.csv')

#getting the shape and the head
d.shape
d.head()

#getting the labels from the dataframe 
labels = d.label
labels.head()

#splitting the dataset 
x_train,x_test,y_train,y_test = train_test_split(d['text'],labels ,test_size = 0.2, random_state=7)

#intiliazing a TfidfVectorizer 
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

#fitting and transforming the train set and testing set 
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

tfidf_test = tfidf_vectorizer.transform(x_test)

#initializing a passiveagressive classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#predicting on the testing set and calculating accuracy 
y_predict =pac.predict(tfidf_test)
score = accuracy_score(y_test,y_predict)
print(f'Accuracy: {round(score*100,2)}%')

#printing a confusion matrix 
confusion_matrix(y_test,y_predict, labels=['FAKE','REAL'])
