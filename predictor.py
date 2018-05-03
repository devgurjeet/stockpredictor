#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:52:59 2018

@author: gurjeets
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import datetime as dt
import pickle
import time

#script started.
start_time = time.time()
print("Compilation start Time {}".format(start_time)) 

dataset = pd.read_csv("./data/client_data_part.csv")

X = dataset.iloc[:,[0]]
y = dataset.iloc[:,4]


#change date to numeric.
X['Date'] = pd.to_datetime(X['Date'])
X['Date'] = X.Date.values.astype(np.int64)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()


X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#splitting dataset into training and test sets.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


#Creating the regression model.
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C= 1e3, gamma= 0.2 )

# Loading the models.
model_svr_regressor = 'svr_regressor_model_full_data.pickle'
regressor = pickle.load(open(model_svr_regressor, 'rb'))



y_pred = sc_y.inverse_transform(regressor.predict(sc_X.fit_transform(X)))
y_pred = regressor.predict(X)


plt.scatter(X,y, color="red")
plt.scatter(X, y_pred, color="blue")
plt.title("(SVR)")
plt.xlabel("Time")
plt.ylabel("Prices")
plt.grid(True)
plt.show()

#predicting the results for five days.



#print("Prediction Score.")
print(regressor.score(X_test, y_test))

totalTime = time.time() - start_time
print("Total Compilation Time {}".format(totalTime)) 