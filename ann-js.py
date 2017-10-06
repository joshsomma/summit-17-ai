#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday September 26 13:26:22 2017

@author: josh

With thanks and credit to Kirill Eremenko & Hadelin de Ponteves and the SuperDataScience team
"""

# ANN

# data pre-processing

# import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values
                
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
                
# split the data between train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# scale features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# set up the ANN
# import libs
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialise the ANN
classifier = Sequential()

# add the input layer and first hidden layer
classifier.add(Dense(kernel_initializer="uniform",activation="relu",input_dim=11,units=12))

# add the 2nd hidden layer
classifier.add(Dense(kernel_initializer="uniform",activation="relu",units=12))

# add a 3rd hidden layer
classifier.add(Dense(kernel_initializer="uniform",activation="relu",units=12))

# add a 4th hidden layer
classifier.add(Dense(kernel_initializer="uniform",activation="relu",units=6))

# add the output layer
classifier.add(Dense(kernel_initializer="uniform",activation="sigmoid",units=1))

# compile the ANN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# fit the ann to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#calculate % of correct predictions
acc_rate = (1515 + 198)/2000
acc_rate

# predict a single item
"""Predict if the customer with the following information will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_bool = (new_prediction > 0.5)


