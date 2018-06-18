# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:16:04 2018

@author: Zakir
"""
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report

#Importing Datasets
titanic = pd.read_csv("titanic_train.csv")
titanic.info()


#Dropping the unnecessary columns as per the provided questions
titanic = titanic.drop (['Name','PassengerId','Ticket','Cabin','Embarked'], axis=1)


#Dropping the "Na" values from age to get unique values in all columns
titanic = titanic.dropna(subset = ['Age'])
titanic.info()

#Assigning the dependant and independant variables
X = titanic.iloc[:, 1:8].values
y = titanic.iloc[:, 0].values

#Label encoding the "SEX" column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X= X[:,1:]

#Splitting train set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#Fitting the model into the regression algorithm
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

#PREDICTING 
y_pred = regressor.predict(X_test)

#Validating records
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
