# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:31:50 2018

@author: SARTHAK BABBAR
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 00:24:36 2018

@author: SARTHAK BABBAR
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from pandas.plotting import scatter_matrix 
from sklearn import preprocessing,cross_validation

df= pd.read_csv("smotexy.csv")
#without 0 marks
#df= pd.read_csv("dataset_1.csv")
#with 0 marks
#df= pd.read_csv("dataset_2.csv")

#df=df[['sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']]
#df=df[['sex','age','address','famsize','Pstatus','G1','G2','G3']]


#print(df.shape)
X=np.array(df.drop(['G3'],1))

X=preprocessing.scale(X)

y=np.array(df['G3'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


print("Mean absolute Error is",mean_absolute_error(y_test,y_pred) )

reg=LinearRegression()
reg.fit(X,y)
r2 = reg.score(X_test, y_test)
print("Full R2 score is",r2)

#scatter_matrix(df)


