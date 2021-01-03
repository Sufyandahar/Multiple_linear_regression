#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#"""
#Created on Thu Dec 17 14:26:34 2020
#
#@author: sufiyan
#"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
# onehotencoding
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Avoiding dummy variables
X = X[:, 1:]

# spliting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# import model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# prediction 
y_pred = regressor.predict(X_test)
print(y_pred)


from sklearn import metrics
R2 = metrics.r2_score(y_test, y_pred)
print(R2)

# RMSE
rmse = np.sqrt(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))


#Save the model
joblib.dump(regressor, 'multiple_regression')

# load the model
knn_model = joblib.load('multiple_regression')