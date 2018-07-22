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
from sklearn.metrics import mean_absolute_error 



style.use('ggplot')

df= pd.read_csv("dataset_1.csv")

#df= pd.read_csv("smotexy.csv")
dp=pd.read_csv("predict.csv")
df=df[['sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']]

print(df.shape)

df=df.dropna(axis=0,how='any')
print(df.shape)
X=np.array(df.drop(['G3'],1))
X_predict=np.array(dp.drop(['G3'],1))
X_predict=preprocessing.scale(X_predict)
X=preprocessing.scale(X)

y=np.array(df['G3'])

clf=LinearRegression()
clf.fit(X,y)

y_pred = clf.predict(X_predict)

np.savetxt('predictions.csv', y_pred , delimiter=',')

