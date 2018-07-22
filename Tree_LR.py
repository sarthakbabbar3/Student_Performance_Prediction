# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:12:13 2018

@author: SARTHAK BABBAR
"""


import pandas as pd 
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 
style.use('ggplot')

df= pd.read_csv("tree.csv")
df=df[['absences','G1','G2','G3']]

print(df.shape)

df=df.dropna(axis=0,how='any')
print(df.shape)
X=np.array(df.drop(['G3'],1))



y=np.array(df['G3'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

np.savetxt('predictions_tree.csv',clf.predict(X_test), delimiter=',')
np.savetxt('ActualAnswers_tree.csv',y_test, delimiter=',')

reg=LinearRegression()
reg.fit(X,y)
r2 = reg.score(X, y)
print("R2 score is",r2)
