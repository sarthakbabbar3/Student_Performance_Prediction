# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 00:37:35 2018

@author: SARTHAK BABBAR
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:49:57 2018

@author: SARTHAK BABBAR
"""

import pandas as pd 
import numpy as np
from sklearn import preprocessing, cross_validation, svm , neighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
 
style.use('ggplot')

df= pd.read_csv("smotexy.csv")
#without 0 marks
#df= pd.read_csv("dataset_1.csv")
#with 0 marks
#df= pd.read_csv("dataset_2.csv")

#df=df[['sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']]


X=np.array(df.drop(['G3'],1))

X=preprocessing.scale(X)

y=np.array(df['G3'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)



print("Mean absolute Error is",mean_absolute_error(y_test,y_pred) )

reg=neighbors.KNeighborsClassifier()
reg.fit(X,y)
r2 = reg.score(X_test, y_test)
print("Full R2 score is",r2)

